# import joblib
# import numpy as np
# from tqdm import tqdm

# # 假设 full_motion_dict 已经从你的脚本中生成
# full_motion_dict = joblib.load("/home/group16/xuws/PHC-420/sample_data/filter_motions_and_results_2_25_L4_86.pkl")

# def convert_to_smpl_format(motion_out):
#     smpl_data = {}

#     # Axis-angle pose, shape (T, 72)
#     smpl_data['pose_aa'] = motion_out['pose_aa']  # (T, 72)

#     # Root translation
#     smpl_data['trans'] = motion_out['trans_orig']  # (T, 3)

#     # Shape parameters
#     beta = motion_out['beta']
#     if isinstance(beta, float) or isinstance(beta, int):
#         beta = np.array([beta] * 10)  # fallback
#     elif beta.ndim == 2:
#         beta = beta[0]  # from shape (1, 10)
#     smpl_data['beta'] = beta

#     # Gender string
#     gender = motion_out['gender']
#     if isinstance(gender, bytes):
#         gender = gender.decode("utf-8")
#     smpl_data['gender'] = gender

#     # FPS
#     smpl_data['fps'] = motion_out.get('fps', 30.0)

#     return smpl_data

# # 转换整个数据集
# amass_like_data = {}
# for k, v in tqdm(full_motion_dict.items()):
#     amass_like_data[k] = convert_to_smpl_format(v)

# # 保存为 AMASS 格式
# joblib.dump(amass_like_data, "output/amass_like_data.pkl")
# print("Saved to output/amass_like_data.pkl")

# import numpy as np
# import os
# import joblib
# from tqdm import tqdm
# # /home/group16/xuws/HumanoidCombatSim_/skillmimic/data/motions/olympic/selected_hik_files_all_categories.pkl
# # 加载你的 full_motion_dict
# full_motion_dict = joblib.load("/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions/gmxv2_difficulty_2_prompts2.txt_hik.pkl")  
# output_npz_dir = "/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions/gmxv2_difficulty_2_prompts_motions_npz"
# os.makedirs(output_npz_dir, exist_ok=True)

# def save_amass_npz(key, motion_out):
#     save_path = os.path.join(output_npz_dir, f"{key}.npz")

#     pose_aa = motion_out['pose_aa']        # (T, 72)
#     trans = motion_out['trans_orig']       # (T, 3)
#     betas = motion_out['beta']
#     if isinstance(betas, (float, int)):
#         betas = np.array([betas] * 10)
#     elif betas.ndim == 2:
#         betas = betas[0]
    
#     gender = motion_out['gender']
#     if isinstance(gender, bytes):
#         gender = gender.decode("utf-8")
#     gender = np.string_(gender)  # 保存为 bytes 格式
#     mocap_framerate = motion_out.get('fps', 30.0)
    
#     # dmpls 默认设为 0
#     dmpls = np.zeros((pose_aa.shape[0], 8), dtype=np.float32)

#     np.savez_compressed(save_path,
#         poses=pose_aa,
#         trans=trans,
#         betas=betas,
#         gender=gender,
#         mocap_framerate=mocap_framerate,
#         dmpls=dmpls
#     )

# # 批量保存
# # import pdb;pdb.set_trace()
# for key, motion in tqdm(full_motion_dict.items()):
#     save_amass_npz(key, motion)

# print(f"All motions saved to {output_npz_dir}")

import numpy as np
import os
import joblib
from tqdm import tqdm
import argparse  # 导入参数解析模块


def main():
    # -------------------------- 命令行参数解析 --------------------------
    parser = argparse.ArgumentParser(description='将.pkl格式的运动字典转换为AMASS格式的.npz文件（支持自定义路径输入）')
    # 1. 输入：full_motion_dict 的 .pkl 文件路径
    parser.add_argument('--pkl_path', required=True, 
                        help='待加载的 full_motion_dict .pkl 文件路径（如 /home/.../gmxv2_difficulty_2_prompts2.txt_hik.pkl）')
    # 2. 输出：npz 文件的保存目录
    parser.add_argument('--output_dir', required=True, 
                        help='.npz 文件的输出目录（如 /home/.../gmxv2_difficulty_2_prompts_motions_npz）')
    
    args = parser.parse_args()

    # -------------------------- 路径校验与初始化 --------------------------
    # 校验 .pkl 文件是否存在
    if not os.path.exists(args.pkl_path):
        print(f"错误：.pkl 文件不存在 -> {args.pkl_path}")
        return
    # 创建输出目录（不存在则自动创建）
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"已确认：.pkl 文件路径 -> {args.pkl_path}")
    print(f"已创建/确认：npz 输出目录 -> {args.output_dir}\n")


    # -------------------------- 原有核心逻辑（仅替换路径） --------------------------
    def save_amass_npz(key, motion_out, output_dir):
        """保存单个运动为 AMASS 格式的 .npz 文件（新增 output_dir 参数接收命令行输入的目录）"""
        save_path = os.path.join(output_dir, f"{key}.npz")

        pose_aa = motion_out['pose_aa']        # (T, 72)
        trans = motion_out['trans_orig']       # (T, 3)
        betas = motion_out['beta']
        
        # 处理 betas 格式（保持原逻辑）
        if isinstance(betas, (float, int)):
            betas = np.array([betas] * 10)
        elif betas.ndim == 2:
            betas = betas[0]
        
        # 处理 gender 格式（保持原逻辑）
        gender = motion_out['gender']
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        gender = np.string_(gender)  # 保存为 bytes 格式
        mocap_framerate = motion_out.get('fps', 30.0)
        
        # dmpls 默认设为 0（保持原逻辑）
        dmpls = np.zeros((pose_aa.shape[0], 8), dtype=np.float32)

        # 保存 npz 文件（路径改为命令行输入的 output_dir）
        np.savez_compressed(
            save_path,
            poses=pose_aa,
            trans=trans,
            betas=betas,
            gender=gender,
            mocap_framerate=mocap_framerate,
            dmpls=dmpls
        )

    # 加载 full_motion_dict（路径改为命令行输入的 pkl_path）
    print("正在加载 .pkl 文件...")
    full_motion_dict = joblib.load(args.pkl_path)

    # 批量保存 npz 文件（传入命令行输入的 output_dir）
    print("开始批量转换并保存 .npz 文件...")
    for key, motion in tqdm(full_motion_dict.items(), desc="转换进度"):
        save_amass_npz(key, motion, args.output_dir)

    print(f"\n所有运动文件已保存到：{args.output_dir}")


if __name__ == "__main__":
    main()
