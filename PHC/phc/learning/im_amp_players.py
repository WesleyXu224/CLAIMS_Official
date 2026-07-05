
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from rl_games.common.player import BasePlayer

import learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs
from datetime import datetime
import copy
import csv
import json
COLLECT_Z = False

class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)
        # import pdb;pdb.set_trace()
        # 新增：配置PNN目标子网络索引（0/1/2，默认0）
        self.target_pnn_index = config.get('target_pnn_index', 0)
        assert self.target_pnn_index in [0, 1, 2,3,4,5,6,7,8,9], f"PNN子网络索引必须是0-9，当前为{self.target_pnn_index}"

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []

        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0
        self.eval_finished = False

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 0.5 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0

        if humanoid_env.collect_dataset:
            self.obs_buf, self.obs_buf_all = [], []
            self.env_actions, self.actions_all = [], []
            self.motion_length_all = []
            self.clean_actions, self.clean_actions_all = [], []
            self.keys_all = []
            self.reset_buf, self.reset_buf_all = [], []

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        print(self.model.a2c_network) 
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return

    def _sanitize_path_component(self, value):
        value = str(value)
        safe = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_", "."):
                safe.append(ch)
            else:
                safe.append("_")
        sanitized = "".join(safe).strip("._")
        return sanitized or "unnamed"

    def _get_release_eval_dir(self, humanoid_env):
        motion_stem = osp.splitext(osp.basename(humanoid_env.cfg.env.motion_file))[0]
        exp_name = self._sanitize_path_component(humanoid_env.cfg.get("exp_name", "default"))
        motion_name = self._sanitize_path_component(motion_stem)
        epoch_name = "latest" if humanoid_env.cfg.epoch == -1 else f"epoch_{int(humanoid_env.cfg.epoch):08d}"
        return osp.abspath(
            osp.join(
                self.config["network_path"],
                "..",
                "..",
                "..",
                "outputs",
                "eval",
                "phc",
                exp_name,
                motion_name,
                epoch_name,
            )
        )

    def _write_release_eval_artifacts(
        self,
        humanoid_env,
        failed_keys,
        success_keys,
        metrics_all_print,
        metrics_succ_print,
    ):
        eval_dir = self._get_release_eval_dir(humanoid_env)
        os.makedirs(eval_dir, exist_ok=True)

        summary = {
            "exp_name": humanoid_env.cfg.get("exp_name", "default"),
            "motion_file": humanoid_env.cfg.env.motion_file,
            "checkpoint_dir": self.config["network_path"],
            "checkpoint_file": osp.join(self.config["network_path"], "Humanoid.pth"),
            "epoch": int(humanoid_env.cfg.epoch),
            "num_envs": int(humanoid_env.num_envs),
            "num_motions": int(humanoid_env._motion_lib._num_unique_motions),
            "success_rate": float(self.success_rate),
            "num_success": int(len(success_keys)),
            "num_failed": int(len(failed_keys)),
            "metrics_all": {k: float(v) for k, v in metrics_all_print.items()},
            "metrics_success_only": {k: float(v) for k, v in metrics_succ_print.items()},
        }

        with open(osp.join(eval_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        metric_rows = [
            ("success_rate", float(self.success_rate), "all"),
        ]
        metric_rows.extend((k, float(v), "all") for k, v in metrics_all_print.items())
        metric_rows.extend((k, float(v), "success_only") for k, v in metrics_succ_print.items())
        with open(osp.join(eval_dir, "metrics.csv"), "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "metric", "value"])
            for metric, value, split in metric_rows:
                writer.writerow([split, metric, value])

        for filename, keys in (
            ("failed_keys.txt", failed_keys),
            ("success_keys.txt", success_keys),
        ):
            with open(osp.join(eval_dir, filename), "w", encoding="utf-8") as f:
                for key in keys:
                    f.write(f"{key}\n")

        joblib.dump(np.array(failed_keys), osp.join(eval_dir, "failed_keys.pkl"))
        joblib.dump(np.array(success_keys), osp.join(eval_dir, "success_keys.pkl"))

        print(f"Release eval artifacts written to: {eval_dir}")

    def _write_per_sample_metrics(self, humanoid_env, pred_pos_all, gt_pos_all, terminate_hist):
        eval_dir = self._get_release_eval_dir(humanoid_env)
        os.makedirs(eval_dir, exist_ok=True)
        per_sample_path = osp.join(eval_dir, "per_sample_metrics.csv")

        header = [
            "sample_index",
            "name",
            "success",
            "num_frames",
            "mpjpe_g",
            "mpjpe_l",
            "mpjpe_pa",
            "accel_dist",
            "vel_dist",
        ]

        with open(per_sample_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)

            for idx, (name, pred, gt) in enumerate(
                zip(humanoid_env._motion_lib._motion_data_keys, pred_pos_all, gt_pos_all)
            ):
                metrics = compute_metrics_lite([pred], [gt])
                metrics_print = {metric: float(np.mean(value)) for metric, value in metrics.items()}
                writer.writerow([
                    idx,
                    name,
                    int(not terminate_hist[idx]),
                    int(pred.shape[0]),
                    metrics_print["mpjpe_g"],
                    metrics_print["mpjpe_l"],
                    metrics_print["mpjpe_pa"],
                    metrics_print["accel_dist"],
                    metrics_print["vel_dist"],
                ])

        print(f"Per-sample metrics written to: {per_sample_path}")

    def _post_step(self, info, done):
        super()._post_step(info)
        
        
        # modify done such that games will exit and reset.
        if flags.im_eval:

            humanoid_env = self.env.task
            
            termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            # termination_state = info["terminate"]
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            if humanoid_env.collect_dataset:
                self.obs_buf.append(info['obs_buf'])
                self.clean_actions.append(info['clean_actions'])
                self.env_actions.append(info['actions'])
                self.reset_buf.append(info['reset_buf'])

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                if COLLECT_Z:
                    all_zs = torch.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.zs_all += all_zs


                if humanoid_env.collect_dataset:
                    all_obs_buf = np.stack(self.obs_buf) # Time, batch, obs
                    all_obs_buf = [all_obs_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.obs_buf_all += all_obs_buf

                    all_clean_actions = np.stack(self.clean_actions) 
                    all_clean_actions = [all_clean_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.clean_actions_all += all_clean_actions
                    
                    all_actions = np.stack(self.env_actions)
                    all_actions = [all_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.actions_all += all_actions

                    all_reset_buf = np.stack(self.reset_buf)
                    all_reset_buf = [all_reset_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.reset_buf_all += all_reset_buf
                    
                    self.keys_all += humanoid_env._motion_lib.curr_motion_keys.tolist()

                    self.motion_length_all += [obs.shape[0] for obs in all_obs_buf]

                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

                    # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                    # humanoid_env._motion_lib.get_motion_num_steps().sum()

                    failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                    if flags.real_traj:
                        pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                        gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                        pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                        gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                    # print(1 - self.terminate_state.sum() / self.terminate_state.shape[0])
                    print(self.config['network_path'])
                    if COLLECT_Z:
                        zs_all = self.zs_all[:humanoid_env._motion_lib._num_unique_motions]
                        zs_dump = {k: zs_all[idx].cpu().numpy() for idx, k in enumerate(humanoid_env._motion_lib._motion_data_keys)}
                        joblib.dump(zs_dump, osp.join(self.config['network_path'], "zs_run.pkl"))

                    if humanoid_env.collect_dataset:
                        motion_file = humanoid_env.cfg.env.motion_file.split('/')[-1].split('.')[0]
                        dump_dir = osp.join(self.config['network_path'], "phc_act", motion_file, f"noise_{humanoid_env.add_action_noise}_{humanoid_env.action_noise_std}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.pkl")
                        os.makedirs(osp.join(self.config['network_path'], "phc_act", motion_file), exist_ok=True)
                        print("Dumping to: ", dump_dir)
                        joblib.dump({
                                "obs": self.obs_buf_all, 
                                "clean_action": self.clean_actions_all, 
                                "env_action": self.actions_all,
                                "key_names": np.array(self.keys_all),
                                "motion_lengths": np.array(self.motion_length_all),
                                "reset": np.concatenate(self.reset_buf_all), 
                                "running_mean": self.running_mean_std.state_dict(),
                                "config": humanoid_env.cfg,
                                }, dump_dir, compress=True)
                        exit()

                    # import ipdb; ipdb.set_trace()

                    joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
                    joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
                    self._write_per_sample_metrics(
                        humanoid_env=humanoid_env,
                        pred_pos_all=pred_pos_all,
                        gt_pos_all=gt_pos_all,
                        terminate_hist=terminate_hist[: humanoid_env._motion_lib._num_unique_motions],
                    )
                    self._write_release_eval_artifacts(
                        humanoid_env=humanoid_env,
                        failed_keys=failed_keys,
                        success_keys=success_keys,
                        metrics_all_print=metrics_all_print,
                        metrics_succ_print=metrics_print,
                    )
                    self.eval_finished = True
                    print("....")
                    return done

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                if humanoid_env.collect_dataset: 
                    self.obs_buf, self.env_actions, self.clean_actions, self.reset_buf, self.keys = [], [], [], [], []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0


            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)

        return done
    def compute_error_vel(self, joints_gt, joints_pred, vis=None):
        vel_gt = joints_gt[1:] - joints_gt[:-1]
        vel_pred = joints_pred[1:] - joints_pred[:-1]
        normed = np.linalg.norm(vel_pred - vel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        return np.mean(normed[new_vis], axis=1)
    
    def compute_error_accel(self, joints_gt, joints_pred, vis=None):
        """
        Computes acceleration error:
            1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
        Note that for each frame that is not visible, three entries in the
        acceleration error should be zero'd out.
        Args:
            joints_gt (Nx14x3).
            joints_pred (Nx14x3).
            vis (N).
        Returns:
            error_accel (N-2).
        """
        # (N-2)x14x3
        accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
        accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

        normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

        if vis is None:
            new_vis = np.ones(len(normed), dtype=bool)
        else:
            invis = np.logical_not(vis)
            invis1 = np.roll(invis, -1)
            invis2 = np.roll(invis, -2)
            new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
            new_vis = np.logical_not(new_invis)

        return np.mean(normed[new_vis], axis=1)
    
    def p_mpjpe(self, predicted, target):
        """
        Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
        often referred to as "Protocol #2" in many papers.
        """
        assert predicted.shape == target.shape

        muX = np.mean(target, axis=1, keepdims=True)
        muY = np.mean(predicted, axis=1, keepdims=True)

        X0 = target - muX
        Y0 = predicted - muY

        normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
        normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

        X0 /= normX
        Y0 /= normY

        H = np.matmul(X0.transpose(0, 2, 1), Y0)
        U, s, Vt = np.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        R = np.matmul(V, U.transpose(0, 2, 1))

        # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
        sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
        V[:, :, -1] *= sign_detR
        s[:, -1] *= sign_detR.flatten()
        R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

        tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

        a = tr * normX / normY  # Scale
        t = muX - a * np.matmul(muY, R)  # Translation

        # Perform rigid transformation on the input
        predicted_aligned = a * np.matmul(predicted, R) + t

        # Return MPJPE
        return np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        import pdb;pdb.set_trace()
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)


                    if COLLECT_Z: z = self.get_z(obs_dict)
                        

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)

                    cr += r
                    steps += 1

                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())
                    if self.eval_finished:
                        break

                    if render:
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)
                        
                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = (
                                    s[:, all_done_indices, :] * 0.0
                                )

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

                if self.eval_finished:
                    break

            if self.eval_finished:
                break

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

        return
    ###########################Muti Prims###########################
    # def get_action(self, obs_dict, is_determenistic=False):
    #     obs = obs_dict["obs"]
    #     # 1. 观测预处理：扩展批次维度（若需要）
    #     if self.has_batch_dimension == False:
    #         obs = unsqueeze_obs(obs)
    #     # 2. 观测归一化（与训练时一致）
    #     obs = self._preproc_obs(obs)

    #     # 3. 调用PNN指定子网络生成动作（核心）
    #     with torch.no_grad():
    #         target_actor = self.model.a2c_network.pnn.actors[self.target_pnn_index]
    #         raw_action = target_actor(obs)  # 子网络输出69维动作

    #     # 4. 动作后处理：压缩维度+裁剪+缩放（适配环境）
    #     current_action = raw_action
    #     if self.has_batch_dimension == False:
    #         current_action = torch.squeeze(current_action.detach())
    #     # if self.clip_actions:
    #     #     current_action = rescale_actions(
    #     #         self.actions_low,
    #     #         self.actions_high,
    #     #         torch.clamp(current_action, -1.0, 1.0),
    #     #     )

    #     return current_action
