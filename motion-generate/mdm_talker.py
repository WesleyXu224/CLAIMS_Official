# This code is based on https://github.com/openai/guided-diffusion

import glob
import os
import sys
import pdb
import os.path as osp
import re
import os
import numpy as np
import torch
sys.path.append(os.getcwd())
import shutil
from data_loaders.humanml.data.dataset import HumanML3D

from utils.fixseed import fixseed
from utils.parser_util import generate_args
# from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

from data_loaders.tensors import collate
from sample.generate import construct_template_variables, save_multiple_samples, load_dataset
from datetime import datetime

from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel, AutoRegressiveSampler
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric, get_target_location, sample_goal
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate

class MDMTalker:
    def __init__(self):
        self.args = args = generate_args()
        fixseed(args.seed)
        out_path = args.output_dir
        # args.model_path = "./mdm/humanml_trans_enc_512/model000200000.pt"
        name = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
        max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
        fps = 12.5 if args.dataset == 'kit' else 20
        self.n_frames = n_frames = min(max_frames, int(args.motion_length*fps))
        is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
        if args.context_len > 0:
            is_using_data = True  # For prefix completion, we need to sample a prefix
        args.text_prompt = "Running around and jump up and down"
        dist_util.setup_dist(args.device)
        if out_path == '':
            out_path = os.path.join(os.path.dirname(args.model_path),
                                    'samples_{}_{}_seed{}'.format(name, niter, args.seed))
            if args.text_prompt != '':
                out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
            elif args.input_text != '':
                out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        args.num_repetitions = 1
        args.num_samples = 1
        # this block must be called BEFORE the dataset is loaded
#############################################
        if args.text_prompt != '':
            texts = [args.text_prompt] * args.num_samples
        elif args.input_text != '':
            assert os.path.exists(args.input_text)
            with open(args.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            args.num_samples = len(texts)
        elif args.dynamic_text_path != '':
            assert os.path.exists(args.dynamic_text_path)
            assert args.autoregressive, "Dynamic text sampling is only supported with autoregressive sampling."
            with open(args.dynamic_text_path, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            n_frames = len(texts) * args.pred_len  # each text prompt is for a single prediction
        elif args.action_name:
            action_text = [args.action_name]
            args.num_samples = 1
        elif args.action_file != '':
            assert os.path.exists(args.action_file)
            with open(args.action_file, 'r') as fr:
                action_text = fr.readlines()
            action_text = [s.replace('\n', '') for s in action_text]
            args.num_samples = len(action_text)
#############################################


        if args.text_prompt != '':
            texts = [args.text_prompt]
            args.num_samples = 1
        elif args.input_text != '':
            assert os.path.exists(args.input_text)
            with open(args.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            args.num_samples = len(texts)
        elif args.action_name:
            action_text = [args.action_name]
            args.num_samples = 1
        elif args.action_file != '':
            assert os.path.exists(args.action_file)
            with open(args.action_file, 'r') as fr:
                action_text = fr.readlines()
            action_text = [s.replace('\n', '') for s in action_text]
            args.num_samples = len(action_text)
    
        args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
        print('Loading dataset...')
        self.data = data = load_dataset(args, max_frames, n_frames)
        total_num_samples = args.num_samples * args.num_repetitions
        

        print("Creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(args, data)

        self.sample_fn = self.diffusion.p_sample_loop
        if args.autoregressive:
            sample_cls = AutoRegressiveSampler(args, self.sample_fn, n_frames)
            self.sample_fn = sample_cls.sample

        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        # load_model_wo_clip(self.model, state_dict)
        load_saved_model(self.model, args.model_path, use_avg=args.use_ema)

        if args.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)   # wrapping model with the classifier-free sampler
        self.model.to(dist_util.dev())
        self.model.eval()  # disable random masking

        self.motion_shape = (args.batch_size, self.model.njoints, self.model.nfeats, n_frames)
        # import pdb;pdb.set_trace()
        if is_using_data:
            iterator = iter(data)
            input_motion, self.model_kwargs = next(iterator)
            input_motion = input_motion.to(dist_util.dev())
            if texts is not None:
                self.model_kwargs['y']['text'] = texts
        else:
            collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
            is_t2m = any([args.input_text, args.text_prompt])
            if is_t2m:
                # t2m
                collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            else:
                # a2m
                action = data.dataset.action_name_to_action(action_text)
                collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                                arg, one_action, one_action_text in zip(collate_args, action, action_text)]
            _, self.model_kwargs = collate(collate_args)
        
        self.model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in self.model_kwargs['y'].items()}
        # import pdb;pdb.set_trace()
        ##############USE_DIP########
        # prefix_data = np.load("/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/prefix_data.npy")[5]
        # prefix_data = torch.from_numpy(prefix_data).float() 
        # prefix_data = prefix_data.to(args.device) 
        # self.model_kwargs['y']['prefix'][0] = prefix_data
        ##############USE_DIP########

        # original_tensor = self.model_kwargs['y']['prefix']  # 形状 [6, 263, 1, 20]
        # self.model_kwargs['y']['prefix']  = torch.cat([
        #     original_tensor[5:6],  # 取索引5的元素（原第6个）
        #     original_tensor[1:5],  # 取索引1到4的元素（原第2到第5个）
        #     original_tensor[0:1]   # 取索引0的元素（原第1个）
        # ])
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        self.init_image = None    
    def save_multiple_samples(out_path, file_templates,  animations, fps, max_frames, no_dir=False):
        
        num_samples_in_out_file = 3
        n_samples = animations.shape[0]
        
        for sample_i in range(0,n_samples,num_samples_in_out_file):
            last_sample_i = min(sample_i+num_samples_in_out_file, n_samples)
            all_sample_save_file = file_templates['all'].format(sample_i, last_sample_i-1)
            if no_dir and n_samples <= num_samples_in_out_file:
                all_sample_save_path = out_path
            else:
                all_sample_save_path = os.path.join(out_path, all_sample_save_file)
                print(f'saving {os.path.split(out_path)[1]}/{all_sample_save_file}')

            clips = clips_array(animations[sample_i:last_sample_i])
            clips.duration = max_frames/fps
            
            # import time
            # start = time.time()
            clips.write_videofile(all_sample_save_path, fps=fps, threads=4, logger=None)#
            # print(f'duration = {time.time()-start}')
            
            for clip in clips.clips: 
                # close internal clips. Does nothing but better use in case one day it will do something
                clip.close()
            clips.close()  # important
    

    def construct_template_variables(unconstrained):
        row_file_template = 'sample{:02d}.mp4'
        all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
        if unconstrained:
            sample_file_template = 'row{:02d}_col{:02d}.mp4'
            sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
            row_file_template = row_file_template.replace('sample', 'row')
            row_print_template = '[{} row #{:02d} | all columns | -> {}]'
            all_file_template = all_file_template.replace('samples', 'rows')
            all_print_template = '[rows {:02d} to {:02d} | -> {}]'
        else:
            sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
            sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
            row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
            all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

        return sample_print_template, row_print_template, all_print_template, \
            sample_file_template, row_file_template, all_file_template

    def generate_motion(self, prompts, out_path = "mdm_out_3_12", num_repetitions = 3):

        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        os.makedirs(out_path, exist_ok=True)
        
        args, model_kwargs, model, diffusion, data= self.args, self.model_kwargs, self.model, self.diffusion, self.data
        model_kwargs['y']['text'] = prompts
        
        fps = 12.5 if args.dataset == 'kit' else 20
        
        all_motions = []
        all_lengths = []
        all_text = []
        
        total_num_samples  = self.n_frames * num_repetitions
        batch_size = num_samples= len(prompts)


        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
        
        if 'text' in model_kwargs['y'].keys():
            # encoding once instead of each iteration saves lots of time
            model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
        
        if args.dynamic_text_path != '':
            # Rearange the text to match the autoregressive sampling - each prompt fits to a single prediction
            # Which is 2 seconds of motion by default
            model_kwargs['y']['text'] = [model_kwargs['y']['text']] * args.num_samples
            if args.text_encoder_type == 'bert':
                model_kwargs['y']['text_embed'] = (model_kwargs['y']['text_embed'][0].unsqueeze(0).repeat(args.num_samples, 1, 1, 1), 
                                                model_kwargs['y']['text_embed'][1].unsqueeze(0).repeat(args.num_samples, 1, 1))
            else:
                raise NotImplementedError('DiP model only supports BERT text encoder at the moment. If you implement this, please send a PR!')
        
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            sample = self.sample_fn(
                model,
                self.motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=self.init_image,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, self.n_frames).bool()
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]
            # import pdb;pdb.set_trace()
            all_motions.append(sample.cpu().numpy())
            _len = model_kwargs['y']['lengths'].cpu().numpy()
            if 'prefix' in model_kwargs['y'].keys():
                _len[:] = sample.shape[-1]

            all_lengths.append(_len)

            print(f"created {len(all_motions) * args.batch_size} samples")
        all_motions = np.concatenate(all_motions, axis=0)
        all_motions = all_motions[:total_num_samples]  
        all_text = all_text[:total_num_samples]
        # all_lengths = all_lengths * batch_size
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
        # import pdb;pdb.set_trace()
    
#####################SAVE VIDEO####################
        # if os.path.exists(out_path):
        #     shutil.rmtree(out_path)
        # os.makedirs(out_path)

        # npy_path = os.path.join(out_path, 'results.npy')
        # print(f"saving results file to [{npy_path}]")
        # np.save(npy_path,
        #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
        #         'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
        # if args.dynamic_text_path != '':
        #     text_file_content = '\n'.join(['#'.join(s) for s in all_text])
        # else:
        #     text_file_content = '\n'.join(all_text)
        # with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        #     fw.write(text_file_content)
        # with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        #     fw.write('\n'.join([str(l) for l in all_lengths]))

        # print(f"saving visualizations to [{out_path}]...")
        # skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        # sample_print_template, row_print_template, all_print_template, \
        # sample_file_template, row_file_template, all_file_template = construct_template_variables(args.unconstrained)
        # max_vis_samples = 6
        # num_vis_samples = min(args.num_samples, max_vis_samples)
        # animations = np.empty(shape=(args.num_samples, args.num_repetitions), dtype=object)
        # max_length = max(all_lengths)

        # for sample_i in range(args.num_samples):
        #     rep_files = []
        #     for rep_i in range(args.num_repetitions):
        #         caption = all_text[rep_i*args.batch_size + sample_i]
        #         if args.dynamic_text_path != '':  # caption per frame
        #             assert type(caption) == list
        #             caption_per_frame = []
        #             for c in caption:
        #                 caption_per_frame += [c] * args.pred_len
        #             caption = caption_per_frame

                
        #         # Trim / freeze motion if needed
        #         length = all_lengths[rep_i*args.batch_size + sample_i]
        #         motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:max_length]
        #         if motion.shape[0] > length:
        #             motion[length:-1] = motion[length-1]  # duplicate the last frame to end of motion, so all motions will be in equal length

        #         save_file = sample_file_template.format(sample_i, rep_i)
        #         animation_save_path = os.path.join(out_path, save_file)
        #         gt_frames = np.arange(args.context_len) if args.context_len > 0 and not args.autoregressive else []
        #         animations[sample_i, rep_i] = plot_3d_motion(animation_save_path, 
        #                                                     skeleton, motion, dataset=args.dataset, title=caption, 
        #                                                     fps=fps, gt_frames=gt_frames)
        #         rep_files.append(animation_save_path)

        # save_multiple_samples(out_path, {'all': all_file_template}, animations, fps, max(list(all_lengths) + [self.n_frames]))

        # abs_path = os.path.abspath(out_path)
        # print(f'[Done] Results are at [{abs_path}]')
#####################SAVE VIDEO####################











        # os.makedirs(out_path)

        # 合并所有 prompts
        combined_prompts = ' '.join(all_text)
        # 清理字符串以适合作为文件名
        clean_filename = re.sub(r'[^a-zA-Z0-9_]', '_', combined_prompts)
        # 确保文件名不会过长，避免一些系统的限制
        max_filename_length = 200  # 可以根据实际情况调整
        if len(clean_filename) > max_filename_length:
            clean_filename = clean_filename[:max_filename_length]
        file_name = f"{clean_filename}results.npy"
        npy_path = os.path.join(out_path,file_name)

        # print(f"saving results file to [{npy_path}]")
        # np.save(npy_path,
        #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
        #         'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
        ##### Convert to full SMPL
        hand_len = 0.08824
        # import pdb;pdb.set_trace()
        mdm_jts = all_motions.transpose(0, 3, 1, 2).reshape(batch_size, -1, 22, 3)
        
        direction = (mdm_jts[...,  -2, :] - mdm_jts[...,  -4, :])
        left = mdm_jts[...,  -2, :] + direction/np.linalg.norm(direction) * hand_len
        direction = (mdm_jts[...,  -1, :] - mdm_jts[...,  -3, :])
        right = mdm_jts[...,  -1, :] + direction/np.linalg.norm(direction) * hand_len
        mdm_jts_smpl_24 = np.concatenate([mdm_jts, left[...,  None, :], right[..., None, :]], axis = -2)
        # import pdb;pdb.set_trace()
        return mdm_jts_smpl_24.squeeze()


if __name__ == "__main__":
    mdm_talker = MDMTalker()
    mdm_talker.generate_motion(["Running round"])