# # This code is based on https://github.com/openai/guided-diffusion
# """
# Generate a large batch of image samples from a model and save them as a large
# numpy array. This can be used to produce samples for FID evaluation.
# """
# from utils.fixseed import fixseed
# import os
# import numpy as np
# import torch
# from utils.parser_util import edit_args
# from utils.model_util import create_model_and_diffusion, load_model_wo_clip
# from utils import dist_util
# from model.cfg_sampler import ClassifierFreeSampleModel
# from data_loaders.get_data import get_dataset_loader
# from data_loaders.humanml.scripts.motion_process import recover_from_ric
# from data_loaders import humanml_utils
# import data_loaders.humanml.utils.paramUtil as paramUtil
# from data_loaders.humanml.utils.plot_script import plot_3d_motion
# import shutil


# def main():
#     args = edit_args()
#     fixseed(args.seed)
#     out_path = args.output_dir
#     name = os.path.basename(os.path.dirname(args.model_path))
#     niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
#     max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
#     fps = 12.5 if args.dataset == 'kit' else 20
#     dist_util.setup_dist(args.device)
#     if out_path == '':
#         out_path = os.path.join(os.path.dirname(args.model_path),
#                                 'edit_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
#         if args.text_condition != '':
#             out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

#     print('Loading dataset...')
#     assert args.num_samples <= args.batch_size, \
#         f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
#     # So why do we need this check? In order to protect GPU from a memory overload in the following line.
#     # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
#     # If it doesn't, and you still want to sample more prompts, run this script with different seeds
#     # (specify through the --seed flag)
#     args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
#     data = get_dataset_loader(name=args.dataset,
#                               batch_size=args.batch_size,
#                               num_frames=max_frames,
#                               split='test',
#                               hml_mode='train')  # in train mode, you get both text and motion.
#     # data.fixed_length = n_frames
#     total_num_samples = args.num_samples * args.num_repetitions

#     print("Creating model and diffusion...")
#     model, diffusion = create_model_and_diffusion(args, data)

#     print(f"Loading checkpoints from [{args.model_path}]...")
#     state_dict = torch.load(args.model_path, map_location='cpu')
#     load_model_wo_clip(model, state_dict)

#     model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
#     model.to(dist_util.dev())
#     model.eval()  # disable random masking

#     iterator = iter(data)
#     input_motions, model_kwargs = next(iterator)
#     input_motions = input_motions.to(dist_util.dev())
#     texts = [args.text_condition] * args.num_samples
#     model_kwargs['y']['text'] = texts
#     if args.text_condition == '':
#         args.guidance_param = 0.  # Force unconditioned generation

#     # add inpainting mask according to args
#     assert max_frames == input_motions.shape[-1]
#     gt_frames_per_sample = {}
#     model_kwargs['y']['inpainted_motion'] = input_motions
#     if args.edit_mode == 'in_between':
#         model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
#                                                                device=input_motions.device)  # True means use gt motion
#         for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
#             start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
#             gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
#             model_kwargs['y']['inpainting_mask'][i, :, :,
#             start_idx: end_idx] = False  # do inpainting in those frames
#     elif args.edit_mode == 'upper_body':
#         model_kwargs['y']['inpainting_mask'] = torch.tensor(humanml_utils.HML_LOWER_BODY_MASK, dtype=torch.bool,
#                                                             device=input_motions.device)  # True is lower body data
#         model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
#             -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

#     all_motions = []
#     all_lengths = []
#     all_text = []

#     for rep_i in range(args.num_repetitions):
#         print(f'### Start sampling [repetitions #{rep_i}]')

#         # add CFG scale to batch
#         model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

#         sample_fn = diffusion.p_sample_loop

#         sample = sample_fn(
#             model,
#             (args.batch_size, model.njoints, model.nfeats, max_frames),
#             clip_denoised=False,
#             model_kwargs=model_kwargs,
#             skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
#             init_image=None,
#             progress=True,
#             dump_steps=None,
#             noise=None,
#             const_noise=False,
#         )


#         # Recover XYZ *positions* from HumanML3D vector representation
#         if model.data_rep == 'hml_vec':
#             n_joints = 22 if sample.shape[1] == 263 else 21
#             sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
#             sample = recover_from_ric(sample, n_joints)
#             sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

#         all_text += model_kwargs['y']['text']
#         all_motions.append(sample.cpu().numpy())
#         all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

#         print(f"created {len(all_motions) * args.batch_size} samples")


#     all_motions = np.concatenate(all_motions, axis=0)
#     all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
#     all_text = all_text[:total_num_samples]
#     all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

#     if os.path.exists(out_path):
#         shutil.rmtree(out_path)
#     os.makedirs(out_path)

#     npy_path = os.path.join(out_path, 'results.npy')
#     print(f"saving results file to [{npy_path}]")
#     np.save(npy_path,
#             {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
#              'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
#     with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
#         fw.write('\n'.join(all_text))
#     with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
#         fw.write('\n'.join([str(l) for l in all_lengths]))

#     print(f"saving visualizations to [{out_path}]...")
#     skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

#     # Recover XYZ *positions* from HumanML3D vector representation
#     if model.data_rep == 'hml_vec':
#         input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
#         input_motions = recover_from_ric(input_motions, n_joints)
#         input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


#     for sample_i in range(args.num_samples):
#         caption = 'Input Motion'
#         length = model_kwargs['y']['lengths'][sample_i]
#         motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
#         save_file = 'input_motion{:02d}.mp4'.format(sample_i)
#         animation_save_path = os.path.join(out_path, save_file)
#         rep_files = [animation_save_path]
#         print(f'[({sample_i}) "{caption}" | -> {save_file}]')
#         plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
#                        dataset=args.dataset, fps=fps, vis_mode='gt',
#                        gt_frames=gt_frames_per_sample.get(sample_i, []))
#         for rep_i in range(args.num_repetitions):
#             caption = all_text[rep_i*args.batch_size + sample_i]
#             if caption == '':
#                 caption = 'Edit [{}] unconditioned'.format(args.edit_mode)
#             else:
#                 caption = 'Edit [{}]: {}'.format(args.edit_mode, caption)
#             length = all_lengths[rep_i*args.batch_size + sample_i]
#             motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
#             save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
#             animation_save_path = os.path.join(out_path, save_file)
#             rep_files.append(animation_save_path)
#             print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
#             plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
#                            dataset=args.dataset, fps=fps, vis_mode=args.edit_mode,
#                            gt_frames=gt_frames_per_sample.get(sample_i, []))
#             # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

#         all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
#         ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
#         hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
#         ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
#         os.system(ffmpeg_rep_cmd)
#         print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

#     abs_path = os.path.abspath(out_path)
#     print(f'[Done] Results are at [{abs_path}]')


# if __name__ == "__main__":
#     main()


# from visualize.motions2hik import motions2hik
# import numpy as np
# data = np.load("/home/group16/xuws/PHC/motion-diffusion-model/bert_motions_and_results_4_12_1159/L1/A_humanoid_robot_achieves_equilibrium_position_slowly__measured_.npy", allow_pickle=True)
# from scipy.spatial.transform import Rotation as sRot
# import pdb;pdb.set_trace()
# mat = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False).as_matrix()
# data = np.matmul(data, mat.dot(mat))

# # offset = - offset_height - gen_mdm_motions[ 0:1, 0:1, 1]
# # gen_mdm_motions[..., 1] += offset
# # gen_mdm_motions[..., [0, 2]] -= gen_mdm_motions[:1, :1, [0, 2]] - mdm_motions[ticker:(ticker+1), :1, [0, 2]]
# data =  data.transpose(1, 2, 0)[:22, :, :][np.newaxis, :, :, :]
# data_dict = motions2hik(data)

# np.save('phc_L1A_humanoid_robot_achieves_equilibrium_position_slowly__measured_.npy', data_dict)

# import os
# import numpy as np
# from visualize.motions2hik import motions2hik
# from scipy.spatial.transform import Rotation as sRot

# # 源文件夹路径
# source_folder = '/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_3_15_0916/L3'

# # 目标文件夹路径
# destination_folder = '/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_3_15_0916_hik/L3'

# # 创建目标文件夹（若不存在）
# if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

# # 获取源文件夹下的所有文件
# files = os.listdir(source_folder)

# # 遍历源文件夹下的所有文件
# for file in files:
#     if file.endswith('.npy'):
#         file_path = os.path.join(source_folder, file)
#         try:
#             # file_path = "/home/group16/xuws/PHC/motion-diffusion-model/bert_motions_and_results_4_12_1159/L4/It_performs_quicker_sideways_walking_.npy"
#             # 加载 .npy 文件
#             data = np.load(file_path, allow_pickle=True)

#             # 进行旋转操作
#             mat = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False).as_matrix()
#             data = np.matmul(data, mat.dot(mat))

#             # 进行数据变换
#             data = data.transpose(1, 2, 0)[:22, :, :][np.newaxis, :, :, :]

#             # 进行 motions2hik 转换
#             data_dict = motions2hik(data)
#             # import pdb;pdb.set_trace()
#             # 生成目标文件路径
#             new_file_name = f'{file}'
#             new_file_path = os.path.join(destination_folder, new_file_name)
#             # new_file_path = "/home/group16/xuws/PHC/motion-diffusion-model/bert_good_split_motions_and_results_4_12_1159/L5_debug/It_performs_quicker_sideways_walking_.npy"
#             # 保存转换后的数据
#             np.save(new_file_path, data_dict)
            
#             print(f"Converted and saved {file} to {new_file_path}")
#         except Exception as e:
#             print(f"Error processing {file}: {e}")

# import os
# import numpy as np
# from visualize.motions2hik import motions2hik
# from scipy.spatial.transform import Rotation as sRot

# # 定义需要处理的文件夹对列表，每个元素为 (源文件夹路径, 目标文件夹路径)
# folder_pairs = [
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_attack',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_attack/combat_attack_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_combo',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_combo/combat_combo_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_counter',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_counter/combat_counter_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_defense',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_defense/combat_defense_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_feint_bait',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_feint_bait/combat_feint_bait_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_ground_technique',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_ground_technique/combat_ground_technique_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_movement',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_movement/combat_movement_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_takedown_clinch',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_takedown_clinch/combat_takedown_clinch_hik'
#     # ),
#     # (
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_transition_escape',
#     #     '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_transition_escape/combat_transition_escape_hik'
#     # )
#     (
#         '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_movement_add',
#         '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/bert_fighting/combat_movement/combat_movement_add_hik'
#     ),
# ]

# def process_folder_pair(source_folder, destination_folder):
#     # 创建目标文件夹（若不存在）
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#         print(f"创建目标文件夹: {destination_folder}")
    
#     # 获取源文件夹下的所有文件
#     files = os.listdir(source_folder)
#     # import                                                                                                      pdb;pdb.set_trace()
#     for file in files:
#         if file.endswith('.npy'):
#             file_path = os.path.join(source_folder, file)
#             try:
#                 # 加载 .npy 文件
#                 data = np.load(file_path, allow_pickle=True)
                
#                 # 旋转操作（绕x轴旋转90度，矩阵为两次旋转的乘积，这里需确认是否需要单次旋转）
#                 mat = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False).as_matrix()
#                 data = np.matmul(data, mat.dot(mat))
                
#                 # 数据变换：调整维度顺序并截取前22个关节，添加批次维度
#                 data = data.transpose(1, 2, 0)[:22, :, :][np.newaxis, :, :, :]
                
#                 # 进行 motions2hik 转换
#                 data_dict = motions2hik(data)
                
#                 # 生成目标文件路径
#                 new_file_path = os.path.join(destination_folder, file)
#                 # 保存转换后的数据
#                 np.save(new_file_path, data_dict)
                
#                 print(f"成功处理文件: {file_path} -> {new_file_path}")
            
#             except Exception as e:
#                 print(f"处理文件 {file} 时出错: {str(e)}")
#                 continue  # 跳过当前出错文件，继续处理下一个

# # 遍历处理所有文件夹对
# for source, destination in folder_pairs:
#     print(f"\n开始处理文件夹对：\n源文件夹: {source}\n目标文件夹: {destination}")
#     process_folder_pair(source, destination)

# print("\n所有文件夹处理完成！")

# import os
# import numpy as np
# from visualize.motions2hik import motions2hik
# from scipy.spatial.transform import Rotation as sRot
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import multiprocessing

# # 定义需要处理的文件夹对列表，每个元素为 (源文件夹路径, 目标文件夹路径)
# folder_pairs = [
#     (
#         '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions/gmxv2_difficulty_2_prompts2.txt',
#         '/home/group16/xuws/MDM_DIP/motion-diffusion-model/visualize/gmxv2_difficulty_2_prompts_motions/gmxv2_difficulty_2_prompts2.txt_hik'
#     ),
# ]
# def process_file(file, source_folder, destination_folder):
#     """处理单个文件的函数"""
#     if file.endswith('.npy'):
#         file_path = os.path.join(source_folder, file)
#         try:
#             # 加载 .npy 文件
#             data = np.load(file_path, allow_pickle=True)
            
#             # 旋转操作（绕x轴旋转90度）
#             mat = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False).as_matrix()
#             data = np.matmul(data, mat.dot(mat))
            
#             # 数据变换：调整维度顺序并截取前22个关节，添加批次维度
#             data = data.transpose(1, 2, 0)[:22, :, :][np.newaxis, :, :, :]
            
#             # 进行 motions2hik 转换
#             data_dict = motions2hik(data)
            
#             # 生成目标文件路径
#             new_file_path = os.path.join(destination_folder, file)
#             # 保存转换后的数据
#             np.save(new_file_path, data_dict)
            
#             return f"成功处理文件: {file_path} -> {new_file_path}"
        
#         except Exception as e:
#             return f"处理文件 {file} 时出错: {str(e)}"
#     return None

# def process_folder_pair(source_folder, destination_folder, max_workers=None):
#     """处理文件夹对，使用多线程并行处理文件"""
#     # 创建目标文件夹（若不存在）
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)
#         print(f"创建目标文件夹: {destination_folder}")
    
#     # 获取源文件夹下的所有文件
#     files = os.listdir(source_folder)
#     npy_files = [f for f in files if f.endswith('.npy')]
#     print(f"发现 {len(npy_files)} 个.npy文件需要处理")
    
#     # 如果未指定线程数，使用CPU核心数作为默认值
#     if max_workers is None:
#         max_workers = multiprocessing.cpu_count()
#         # 限制最大线程数，避免资源耗尽
#         max_workers = min(max_workers, 16)
    
#     # 使用线程池并行处理文件
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # 提交所有任务
#         futures = [executor.submit(process_file, file, source_folder, destination_folder) 
#                   for file in npy_files]
        
#         # 处理结果
#         for future in as_completed(futures):
#             result = future.result()
#             if result:
#                 print(result)

# # 遍历处理所有文件夹对
# for source, destination in folder_pairs:
#     print(f"\n开始处理文件夹对：\n源文件夹: {source}\n目标文件夹: {destination}")
#     process_folder_pair(source, destination)

# print("\n所有文件夹处理完成！")

import os
import numpy as np
import argparse  # 新增：用于处理命令行参数
from visualize.motions2hik import motions2hik
from scipy.spatial.transform import Rotation as sRot
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def process_file(file, source_folder, destination_folder):
    """处理单个文件的函数"""
    if file.endswith('.npy'):
        file_path = os.path.join(source_folder, file)
        try:
            # 加载 .npy 文件
            data = np.load(file_path, allow_pickle=True)
################MDM################
            # 旋转操作（绕x轴旋转90度）
            mat = sRot.from_euler('xyz', np.array([np.pi / 2, 0, 0]), degrees=False).as_matrix()
            data = np.matmul(data, mat.dot(mat))
            
##################T2M Benchmark#############
            # mat = sRot.from_euler('x', 90, degrees=True).as_matrix()
            # data = np.matmul(data, mat)
            # 数据变换：调整维度顺序并截取前22个关节，添加批次维度
            data = data.transpose(1, 2, 0)[:22, :, :][np.newaxis, :, :, :]
            
            # 进行 motions2hik 转换
            data_dict = motions2hik(data)
            
            # 生成目标文件路径
            new_file_path = os.path.join(destination_folder, file)
            # 保存转换后的数据
            np.save(new_file_path, data_dict)
            
            return f"成功处理文件: {file_path} -> {new_file_path}"
        
        except Exception as e:
            return f"处理文件 {file} 时出错: {str(e)}"
    return None

def process_folder_pair(source_folder, destination_folder, max_workers=None):
    """处理文件夹对，使用多线程并行处理文件"""
    # 创建目标文件夹（若不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"创建目标文件夹: {destination_folder}")
    
    # 验证源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"错误：源文件夹不存在 - {source_folder}")
        return
    if not os.path.isdir(source_folder):
        print(f"错误：源路径不是文件夹 - {source_folder}")
        return
    
    # 获取源文件夹下的所有npy文件
    files = os.listdir(source_folder)
    npy_files = [f for f in files if f.endswith('.npy')]
    print(f"发现 {len(npy_files)} 个.npy文件需要处理")
    
    # 如果未指定线程数，使用CPU核心数作为默认值（限制最大16）
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 32)
    
    # 使用线程池并行处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file, source_folder, destination_folder) 
                  for file in npy_files]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理npy文件并转换为hik格式')
    parser.add_argument('source_path', help='源文件夹路径（包含npy文件）')
    parser.add_argument('dest_path', help='目标文件夹路径（保存转换后文件）')
    args = parser.parse_args()
    
    # 从命令行参数获取路径对
    folder_pairs = [
        (args.source_path, args.dest_path)
    ]
    
    # 处理文件夹对
    for source, destination in folder_pairs:
        print(f"\n开始处理文件夹对：\n源文件夹: {source}\n目标文件夹: {destination}")
        process_folder_pair(source, destination)
    
    print("\n所有文件夹处理完成！")
