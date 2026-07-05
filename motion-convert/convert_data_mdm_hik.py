import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

# from smpl_sim.utils.config_utils.copycat_config import Config as CC_Config
# from smpl_sim.khrylib.utils import get_body_qposaddr
from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
# from smpl_sim.smpllib.smpl_robot import Robot
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState


# # 配置与初始化
# robot_cfg = {
#     "mesh": False,
#     "model": "smpl",
#     "upright_start": True,
#     "body_params": {},
#     "joint_params": {},
#     "geom_params": {},
#     "actuator_params": {},
# }
# print(robot_cfg)

# smpl_local_robot = LocalRobot(
#     robot_cfg,
#     data_dir="/home/group16/xuws/PULSE/data/smpl",
# )

# # 定义基础路径
# # base_path = "/home/group16/xuws/PHC/motion-diffusion-model/bert_good_split_motions_and_results_4_12_1159"
# # base_path = "/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_2_25_1546_hik"

# # base_path ="/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_3_08_2030_hik"#L3
# # base_path ="/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_3_12_1809_hik"
# # base_path ="/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_3_15_0916_hik"
# # base_path ="/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_3_13_2219_hik"#L4
# # base_path ="/home/group16/xuws/GradedMotionX-V2/mdm_bert_l15/" 
# # base_path ="/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting_hik_full"
# # base_path ="/home/group16/xuws/HumanoidCombatSim/skillmimic/data/motions"
# # base_path ="/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_soccer"
# base_path ="/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions"
# # base_path ="/home/group16/xuws/PHC/motion-diffusion-model"
# # 定义要处理的文件夹列表
# # folders = ["debug_hik"] 
# # folders = ["hik_to_process"] 
# # folders = ["l1_hik","l2_hik","l3_hik","l4_hik","l5_hik"] ccc
# # folders = ["combat_attack_hik","combat_combo_hik","combat_counter_hik","combat_defense_hik","combat_feint_bait_hik","combat_takedown_clinch_hik","combat_transition_escape_hik","combat_leg_hik","combat_movement_add_hik"] 
# # folders = ["selected_hik_files_all_categories"]
# # folders = ["soccer_mdm_hik"]
# folders = ["gmxv2_difficulty_1_prompts.txt_hik", "gmxv2_difficulty_2_prompts.txt_hik"]

# # folders = ["L4"] 
# # for i in range(6):
# #     folders.extend([f"L{i}_ranked_test_prompts", f"L{i}_ranked_train_prompts"])

# # 循环处理每个文件夹
# for folder_name in folders:
#     folder_path = os.path.join(base_path, folder_name)
#     # 检查文件夹是否存在
#     if not os.path.exists(folder_path):
#         print(f"Folder {folder_path} does not exist. Skipping...")
#         continue

#     # 获取排序后的 .npy 文件列表
#     npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))

#     # 初始化数据容器（使用列表存储每个文件的完整数据，保留文件名映射）
#     file_data_list = []
#     for file_path in npy_files:
#         file_name = os.path.basename(file_path).replace(".npy", "")
#         data = np.load(file_path, allow_pickle=True).item()
#         file_data_list.append({"name": file_name, "data": data})

#     # 按文件顺序堆叠数据（确保维度匹配）
#     stacked_thetas = np.vstack([fd["data"]["thetas"] for fd in file_data_list])
#     stacked_root_trans = np.vstack([fd["data"]["root_translation"] for fd in file_data_list])

#     # 构建文件名到数据的映射（强化对应关系）
#     name_to_index = {fd["name"]: i for i, fd in enumerate(file_data_list)}

#     # mdm2amass 处理（显式通过文件名索引数据）
#     amass_data = {}
#     for fd in file_data_list:
#         file_name = fd["name"]
#         idx = name_to_index[file_name]  # 通过文件名获取唯一索引

#         # 提取单文件数据（确保与文件名严格对应）
#         pose_euler = stacked_thetas[idx].reshape(-1, 24, 3)
#         trans = stacked_root_trans[idx]
#         pose_aa = sRot.from_euler('XYZ', pose_euler.reshape(-1, 3), degrees=True).as_rotvec().reshape(-1, 72)

#         # 坐标变换（保持原有逻辑）
#         transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
#         # import pdb;pdb.set_trace()
#         new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
#         pose_aa[:, :3] = new_root
#         trans = trans.dot(transform.as_matrix().T)
#         # import pdb;pdb.set_trace()

#         trans[:, 2] -= (trans[0, 2] - 0.92)
#         # 添加条件判断，如果 trans[:, 2] 最大的值大于 1.4 或者最小的值小于 0.5，则跳过该数据
#         if trans[:, 2].max() > 1.3 or trans[:, 2].min() < 0.5:
#             print(f"Skipping file {file_name} due to trans[:, 2] out of range.")
#             continue

#         amass_data[file_name] = {"pose_aa": pose_aa, "trans": trans, "beta": np.zeros(10)}

#     double = False
#     mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
#     amass_full_motion_dict = {}
#     for file_name in npy_files:  # 按文件顺序遍历
#         file_name_clean = os.path.basename(file_name).replace(".npy", "")
#         if file_name_clean not in amass_data:
#             continue
#         smpl_data_entry = amass_data[file_name_clean]
#         B = smpl_data_entry['pose_aa'].shape[0]

#         start, end = 0, 0

#         pose_aa = smpl_data_entry['pose_aa'].copy()[start:]
#         root_trans = smpl_data_entry['trans'].copy()[start:]
#         B = pose_aa.shape[0]

#         beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
#         if len(beta.shape) == 2:
#             beta = beta[0]

#         gender = smpl_data_entry.get("gender", "neutral")
#         fps = 30.0

#         if isinstance(gender, np.ndarray):
#             gender = gender.item()

#         if isinstance(gender, bytes):
#             gender = gender.decode("utf-8")
#         if gender == "neutral":
#             gender_number = [0]
#         elif gender == "male":
#             gender_number = [1]
#         elif gender == "female":
#             gender_number = [2]
#         else:
#             raise Exception("Gender Not Supported!!")

#         smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
#         batch_size = pose_aa.shape[0]
#         pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
#         pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()

#         num = 1
#         pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

#         gender_number, beta[:], gender = [0], 0, "neutral"
#         print("using neutral model")

#         # smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
#         # smpl_local_robot.write_xml("/home/group16/xuws/human2humanoid/SMPLSim/smpl_sim/data/assets/mjcf/smpl_humanoid_1.xml")
#         skeleton_tree = SkeletonTree.from_mjcf("/home/group16/xuws/human2humanoid/SMPLSim/smpl_sim/data/assets/mjcf/smpl_humanoid_1.xml")
#         # skeleton_tree = SkeletonTree.from_mjcf("//home/group16/xuws/HumanoidCombatSim/skillmimic/data/assets/mjcf/smpl_humanoid_neutral_boxing.xml")

#         root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
#         # import pdb;pdb.set_trace()
#         new_sk_state = SkeletonState.from_rotation_and_root_translation(
#             skeleton_tree,
#             torch.from_numpy(pose_quat),
#             root_trans_offset,
#             is_local=True)

#         if robot_cfg['upright_start']:
#             pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)

#             print("############### filtering!!! ###############")
#             import scipy.ndimage.filters as filters
#             from smpl_sim.utils.transform_utils import quat_correct
#             root_trans_offset = filters.gaussian_filter1d(root_trans_offset, 3, axis=0, mode="nearest")
#             root_trans_offset = torch.from_numpy(root_trans_offset)
#             pose_quat_global = np.stack([quat_correct(pose_quat_global[:, i]) for i in range(pose_quat_global.shape[1])], axis=1)

#             filtered_quats = filters.gaussian_filter1d(pose_quat_global, 2, axis=0, mode="nearest")
#             pose_quat_global = filtered_quats / np.linalg.norm(filtered_quats, axis=-1)[..., None]
#             print("############### filtering!!! ###############")
#             new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
#             pose_quat = new_sk_state.local_rotation.numpy()

#         new_motion_out = {}
#         new_motion_out['pose_quat_global'] = pose_quat_global
#         new_motion_out['pose_quat'] = pose_quat
#         new_motion_out['trans_orig'] = root_trans
#         new_motion_out['root_trans_offset'] = root_trans_offset
#         new_motion_out['beta'] = beta
#         new_motion_out['gender'] = gender
#         new_motion_out['pose_aa'] = pose_aa
#         new_motion_out['fps'] = fps
#         amass_full_motion_dict[file_name_clean] = new_motion_out

#     # # 保存结果到以文件夹名命名的 .pkl 文件
#     # output_path = os.path.join("/home/group16/xuws/GradedMotionX-V2/mdm_bert_l15/", f"vis_l1_l5_hik_{folder_name}.pkl")
#     # joblib.dump(amass_full_motion_dict, output_path)
#     # print(f"Processed {folder_name} and saved to {output_path}")
    

#             # 使用f-string将folder_name变量嵌入路径
#         output_path = os.path.join(
#             f"/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions/{folder_name}/",
#             f"{file_name_clean}.pkl"
#         )

#         # output_path = os.path.join("/home/group16/xuws/roboboxing/SkillMimic/skillmimic/data/motions/bert_fighting_small/{folder_name}/", f"{file_name_clean}.pkl")
#         joblib.dump(new_motion_out, output_path)

#     output_path_full = os.path.join(
#         f"/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions/",
#         f"{folder_name}.pkl"
#     )
#     joblib.dump(amass_full_motion_dict, output_path_full)



import torch
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import os
import sys
import pdb
import os.path as osp
import argparse  # 导入参数解析模块
sys.path.append(os.getcwd())

# from smpl_sim.utils.config_utils.copycat_config import Config as CC_Config
# from smpl_sim.khrylib.utils import get_body_qposaddr
from smpl_sim.smpllib.smpl_mujoco_new import SMPL_BONE_ORDER_NAMES as joint_names
# from smpl_sim.smpllib.smpl_robot import Robot
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
def main():
    # -------------------------- 命令行参数解析 --------------------------
    parser = argparse.ArgumentParser(description='处理运动数据并输出为pkl文件（支持自定义路径输入）')
    # 1. 基础路径（base_path）
    parser.add_argument('--base_path', required=True, 
                        help='基础数据目录（如 /home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions）')
    # 2. 要处理的文件夹列表（folders）
    parser.add_argument('--folders', required=True, nargs='+', 
                        help='要处理的子文件夹列表（空格分隔，如 gmxv2_difficulty_1_prompts.txt_hik gmxv2_difficulty_2_prompts.txt_hik）')
    # 3. 单个文件输出根目录（用于拼接 output_path）
    parser.add_argument('--single_output_root', required=True, 
                        help='单个文件输出的根目录（最终路径为 该目录/文件夹名/文件名.pkl，如 /home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions）')
    # 4. 完整字典输出目录（用于 output_path_full）
    parser.add_argument('--full_output_dir', required=True, 
                        help='完整字典文件输出目录（最终路径为 该目录/文件夹名.pkl，如 /home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions）')
    
    args = parser.parse_args()


    # -------------------------- 配置与初始化 --------------------------
    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    print("机器人配置:", robot_cfg)

    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="/home/group16/xuws/PULSE/data/smpl",  # 若该路径也需动态配置，可再添加一个参数
    )

    # 从命令行参数获取核心路径配置
    base_path = args.base_path
    folders = args.folders
    single_output_root = args.single_output_root
    full_output_dir = args.full_output_dir


    # -------------------------- 循环处理每个文件夹 --------------------------
    for folder_name in folders:
        folder_path = os.path.join(base_path, folder_name)
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"警告：文件夹 {folder_path} 不存在，跳过...")
            continue

        # 获取排序后的 .npy 文件列表
        npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
        if not npy_files:
            print(f"警告：文件夹 {folder_path} 中没有 .npy 文件，跳过...")
            continue

        # 初始化数据容器（存储每个文件的完整数据及文件名）
        file_data_list = []
        for file_path in npy_files:
            file_name = os.path.basename(file_path).replace(".npy", "")
            data = np.load(file_path, allow_pickle=True).item()
            file_data_list.append({"name": file_name, "data": data})
        # import pdb;pdb.set_trace()
        # 按文件顺序堆叠数据（确保维度匹配）
        # stacked_thetas = np.vstack([fd["data"]["thetas"] for fd in file_data_list])
        # stacked_root_trans = np.vstack([fd["data"]["root_translation"] for fd in file_data_list])

        # # 构建文件名到数据的映射（强化对应关系）
        # name_to_index = {fd["name"]: i for i, fd in enumerate(file_data_list)}

        # mdm2amass 处理（显式通过文件名索引数据）
        # amass_data = {}
        # for fd in file_data_list:
        #     file_name = fd["name"]
        #     idx = name_to_index[file_name]  # 通过文件名获取唯一索引

        #     # 提取单文件数据（确保与文件名严格对应）
        #     pose_euler = stacked_thetas[idx].reshape(-1, 24, 3)
        #     trans = stacked_root_trans[idx]
        #     pose_aa = sRot.from_euler('XYZ', pose_euler.reshape(-1, 3), degrees=True).as_rotvec().reshape(-1, 72)

        #     # 坐标变换
        #     transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
        #     new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
        #     pose_aa[:, :3] = new_root
        #     trans = trans.dot(transform.as_matrix().T)

        #     # 过滤异常高度数据
        #     trans[:, 2] -= (trans[0, 2] - 0.92)

        #     if trans[:, 2].max() > 2.5 or trans[:, 2].min() < 0.2:
        #         print(f"跳过文件 {file_name}：trans[:, 2] 超出范围（max={trans[:, 2].max()}, min={trans[:, 2].min()}）")
        #         continue

        #     amass_data[file_name] = {"pose_aa": pose_aa, "trans": trans, "beta": np.zeros(10)}
# -------------------------------------------------------------------------
        # [修改开始] 删除 np.vstack 堆叠部分，改为逐个处理，以支持不同帧数
        # -------------------------------------------------------------------------
        
        # 原代码（已注释）：
        # stacked_thetas = np.vstack([fd["data"]["thetas"] for fd in file_data_list])
        # stacked_root_trans = np.vstack([fd["data"]["root_translation"] for fd in file_data_list])
        # name_to_index = {fd["name"]: i for i, fd in enumerate(file_data_list)}

        # amass_data 处理（直接从 file_data_list 读取）
        amass_data = {}
        for fd in file_data_list:
            file_name = fd["name"]
            
            # [修改点]：直接从当前文件数据中读取，不使用 stack
            # 注意：如果数据维度是 (1, Frames, DoF)，需要先取 [0] 或者 reshape
# 原代码：
            # raw_thetas = fd["data"]["thetas"]
            # raw_trans = fd["data"]["root_translation"]

            # ---------------------------------------------------------------------
            # [修改] 加上 np.array() 强制转换为数组
            # ---------------------------------------------------------------------
            raw_thetas = np.array(fd["data"]["thetas"])
            raw_trans = np.array(fd["data"]["root_translation"])
            
            # 处理 Thetas
            # 确保 reshape 的第一个维度是帧数 (-1)
            pose_euler = raw_thetas.reshape(-1, 24, 3)
            
            # 处理 Trans
            # 如果 trans 带有 batch 维度 (1, Frames, 3)，需降维
            if len(raw_trans.shape) == 3:
                trans = raw_trans[0]
            else:
                trans = raw_trans

            pose_aa = sRot.from_euler('XYZ', pose_euler.reshape(-1, 3), degrees=True).as_rotvec().reshape(-1, 72)

            # 坐标变换
            transform = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False)
            new_root = (transform * sRot.from_rotvec(pose_aa[:, :3])).as_rotvec()
            pose_aa[:, :3] = new_root
            trans = trans.dot(transform.as_matrix().T)

            # 过滤异常高度数据
            # 确保 trans 有数据
            if trans.shape[0] > 0:
                trans[:, 2] -= (trans[0, 2] - 0.92)

                if trans[:, 2].max() > 2.5 or trans[:, 2].min() < 0.2:
                    print(f"跳过文件 {file_name}：trans[:, 2] 超出范围（max={trans[:, 2].max()}, min={trans[:, 2].min()}）")
                    continue

                amass_data[file_name] = {"pose_aa": pose_aa, "trans": trans, "beta": np.zeros(10)}
            else:
                print(f"跳过空文件 {file_name}")
                continue
        
        # -------------------------------------------------------------------------
        # [修改结束]
        # -------------------------------------------------------------------------
        # 处理并保存每个文件的结果
        mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
        amass_full_motion_dict = {}
        new_motion_out_dict = {}
        for file_path in npy_files:  # 按文件顺序遍历
            file_name_clean = os.path.basename(file_path).replace(".npy", "")
            if file_name_clean not in amass_data:
                continue  # 跳过被过滤的文件

            smpl_data_entry = amass_data[file_name_clean]
            B = smpl_data_entry['pose_aa'].shape[0]

            # 提取姿态和位移数据
            pose_aa = smpl_data_entry['pose_aa'].copy()
            root_trans = smpl_data_entry['trans'].copy()
            beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy()
            if len(beta.shape) == 2:
                beta = beta[0]

            # 处理性别信息
            gender = smpl_data_entry.get("gender", "neutral")
            if isinstance(gender, np.ndarray):
                gender = gender.item()
            if isinstance(gender, bytes):
                gender = gender.decode("utf-8")
            gender_number = [0] if gender == "neutral" else [1] if gender == "male" else [2] if gender == "female" else None
            if gender_number is None:
                print(f"跳过文件 {file_name_clean}：不支持的性别 {gender}")
                continue
            print(f"使用中性模型处理文件 {file_name_clean}")

            # 姿态格式转换（适配Mujoco）
            smpl_2_mujoco = [joint_names.index(q) for q in mujoco_joint_names if q in joint_names]
            batch_size = pose_aa.shape[0]
            pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
            pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

            # 加载骨骼树并处理姿态
            skeleton_tree = SkeletonTree.from_mjcf("/home/group16/xuws/human2humanoid/SMPLSim/smpl_sim/data/assets/mjcf/smpl_humanoid_1.xml")

            
            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            # 校正姿态（保持原有逻辑）
            if robot_cfg['upright_start']:
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True
                )
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)

                print(f"############### 过滤文件 {file_name_clean} 的姿态数据 ###############")
                from smpl_sim.utils.transform_utils import quat_correct
                # 平滑根节点位移
                root_trans_offset = torch.from_numpy(ndimage.filters.gaussian_filter1d(root_trans_offset.numpy(), 3, axis=0, mode="nearest"))
                # 校正四元数并平滑
                pose_quat_global = np.stack([quat_correct(pose_quat_global[:, i]) for i in range(pose_quat_global.shape[1])], axis=1)
                filtered_quats = ndimage.filters.gaussian_filter1d(pose_quat_global, 2, axis=0, mode="nearest")
                pose_quat_global = filtered_quats / np.linalg.norm(filtered_quats, axis=-1)[..., None]
                # 更新骨骼状态
                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,
                    torch.from_numpy(pose_quat_global),
                    root_trans_offset,
                    is_local=False
                )
                pose_quat = new_sk_state.local_rotation.numpy()

            # 封装输出数据
            new_motion_out = {
                'pose_quat_global': pose_quat_global,
                'pose_quat': pose_quat,
                'trans_orig': root_trans,
                'root_trans_offset': root_trans_offset,
                'beta': beta,
                'gender': gender,
                'pose_aa': pose_aa,
                'fps': 30.0
            }
            new_motion_out_dict[file_name_clean]=new_motion_out
            amass_full_motion_dict[file_name_clean] = new_motion_out

            # 保存单个文件结果（使用命令行传入的 single_output_root）
            # 输出路径：single_output_root/文件夹名/文件名.pkl
            # single_output_folder = os.path.join(single_output_root, folder_name)
            os.makedirs(single_output_root, exist_ok=True)  # 确保目录存在
            single_output_path = os.path.join(single_output_root, f"{file_name_clean}.pkl")
            joblib.dump(new_motion_out_dict, single_output_path)
            print(f"已保存单个文件结果：{single_output_path}")
            new_motion_out_dict = {}
        # 保存完整字典结果（使用命令行传入的 full_output_dir）
        # 输出路径：full_output_dir/文件夹名.pkl
        os.makedirs(full_output_dir, exist_ok=True)  # 确保目录存在
        full_output_path = os.path.join(full_output_dir, f"{folder_name}.pkl")
        joblib.dump(amass_full_motion_dict, full_output_path)
        print(f"已保存文件夹 {folder_name} 的完整字典结果：{full_output_path}\n")

    print("所有文件夹处理完成！")


if __name__ == "__main__":
    main()
