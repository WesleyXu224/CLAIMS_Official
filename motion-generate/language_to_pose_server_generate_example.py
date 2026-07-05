#!/usr/bin/env python3
import glob
import os
import sys
import pdb
import os.path as osp

# os.chdir("/home/group16/xuws/PHC/motion-diffusion-model")
sys.path.append(os.getcwd())


import cv2
import joblib
import numpy as np
import time


import asyncio
import cv2
import numpy as np
import threading
from scipy.spatial.transform import Rotation as sRot

import time
import torch
from collections import deque
from datetime import datetime
from torchvision import transforms as T
import time

from aiohttp import web
import aiohttp
import jinja2
import json
import scipy.interpolate as interpolate
import subprocess
from io import StringIO
from mdm_talker import MDMTalker
import re
import argparse
# -------------------------- 关键修改：复用原脚本的参数解析器 --------------------------
# 假设原脚本/MDMTalker依赖的模块有一个创建基础解析器的函数（如create_argparser）
# 若原脚本没有，先创建一个基础解析器（确保包含原有的模型参数）
try:
    # 尝试导入原脚本的基础解析器（若存在，如from train import create_argparser）
    from train import create_argparser  # 替换为原脚本实际的解析器导入路径
    parser = create_argparser()
except ImportError:
    # 若原脚本无现成解析器，创建基础解析器（包含原运行时需要的参数）
    import argparse
    parser = argparse.ArgumentParser(description='PHC Language to Pose Server')
    # 添加原脚本必需的基础参数（参考你运行时的报错信息，补充所有原有的必选参数）
    parser.add_argument('--model_path', required=True, help='模型文件路径')
    parser.add_argument('--text_encoder_type', required=True, choices=['clip', 'bert'], help='文本编码器类型')
    parser.add_argument('--cuda', type=str, default='0', help='CUDA设备号')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    # （可根据报错信息补充其他原有的参数，如--dataset、--arch等，确保原模型能正常加载）

# -------------------------- 新增：在原解析器上添加你的自定义参数 --------------------------
parser.add_argument('--prompts_path', required=True, help='提示文件所在目录（如 prompts/gmxv2）')
parser.add_argument('--prompts_file_names', required=True, nargs='+', help='提示文件名列表（空格分隔，如 gmxv2_difficulty_3_prompts.txt）')
parser.add_argument('--motion_save_path', required=True, help='运动结果保存目录（如 gmxv2_difficulty_3_prompts_motions）')


STANDING_POSE = np.array([[[-0.1443, -0.9426, -0.2548],
         [-0.2070, -0.8571, -0.2571],
         [-0.0800, -0.8503, -0.2675],
         [-0.1555, -1.0663, -0.3057],
         [-0.2639, -0.5003, -0.2846],
         [-0.0345, -0.4931, -0.3108],
         [-0.1587, -1.2094, -0.2755],
         [-0.2534, -0.1022, -0.3361],
         [-0.0699, -0.1012, -0.3517],
         [-0.1548, -1.2679, -0.2675],
         [-0.2959, -0.0627, -0.2105],
         [-0.0213, -0.0424, -0.2277],
         [-0.1408, -1.4894, -0.2892],
         [-0.2271, -1.3865, -0.2622],
         [-0.0715, -1.3832, -0.2977],
         [-0.1428, -1.5753, -0.2303],
         [-0.3643, -1.3792, -0.2646],
         [ 0.0509, -1.3730, -0.3271],
         [-0.3861, -1.1423, -0.3032],
         [ 0.0634, -1.1300, -0.3714],
         [-0.4086, -0.9130, -0.2000],
         [ 0.1203, -0.8943, -0.3002],
         [-0.4000, -0.8282, -0.1817],
         [ 0.1207, -0.8087, -0.2787]]]).repeat(5, axis = 0)

def fps_20_to_30(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(24):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x)
        f_y = interpolate.interp1d(x, int_y)
        f_z = interpolate.interp1d(x, int_z)
        
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts

 
    
async def websocket_handler(request):
    print('Websocket connection starting')
    global pose_mat, trans, dt, sim_talker, ws_talkers
    sim_talker = aiohttp.web.WebSocketResponse()
    ws_talkers.append(sim_talker)
    await sim_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in sim_talker:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == "get_pose":
                await sim_talker.send_json({
                    "pose_mat": pose_mat.tolist(),
                    "trans": trans.tolist(),
                    "dt": dt,
                })

    print('Websocket connection closed')
    return sim_talker 

async def pose_getter(request):
    # query env configurations
    global pose_mat, trans, dt, j3d, tracking_res, ticker, reset_offset, reset_buffer, mdm_motions, cycle_motion
    curr_paths = {}
    
    if reset_offset:
        offset = - offset_height - mdm_motions[0, 0, 1]
        mdm_motions[..., 1] += offset
        reset_offset = False
        
    if reset_buffer:
        if buffer > 0:
            mdm_motions = np.concatenate([np.repeat(mdm_motions[0:1], buffer, axis = 0), mdm_motions])
        else:
            mdm_motions = mdm_motions[-buffer:]
            
        reset_buffer = False
    
    if cycle_motion:
        if ticker > len(mdm_motions) - 1:
            ticker = 0
            mdm_motions[..., [0, 2]] = mdm_motions[..., [0, 2]] - (mdm_motions[:1, :1, [0, 2]] -  mdm_motions[-1:, :1, [0, 2]])
            
            
        j3d_curr = mdm_motions[ticker]
        
        
    else:
        j3d_curr = mdm_motions[min(len(mdm_motions)-1, ticker)]
        
    j3d[0] = j3d_curr
    json_resp = {
        "j3d": j3d.tolist(),
        "dt": dt,
    }
    ticker += 1
        
    return web.json_response(json_resp)

def generate_text(prompts = None, output_file_path = "bert_fighting"):
    global offset_height, mdm_talker, buffer, mdm_motions, ticker
############When Inference Motions ###################### 
    # motion_name = "The_person_s_balance_threshold_is_crossed__their_attempts_to_walk_now_just_contribute_to_their_rapid_fall_"  # Here To Change
    # motion_file_path = "/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_2_16/L5/{}.npy".format(motion_name)
    # mdm_motions = np.load(motion_file_path)
    # # 拼接 .txt 文件的完整路径
    # txt_file_path = f"/home/group16/xuws/PHC/motion-diffusion-model/motions_and_results_2_16/L5/{motion_name}.txt"
    # # 以写入模式打开文件，如果文件不存在则创建
    # with open(txt_file_path, 'w') as txt_file:
    #     pass
############When Inference Motions ###################### 
    prompts = prompts.split("\n")
    num_prompt = len(prompts)
    gen_mdm_motions = mdm_talker.generate_motion(prompts,"bert_fighting")
    # import pdb;pdb.set_trace()
    mat = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    gen_mdm_motions = np.matmul(gen_mdm_motions, mat.dot(mat))
    
    offset = - offset_height - gen_mdm_motions[ 0:1, 0:1, 1]
    gen_mdm_motions[..., 1] += offset
    gen_mdm_motions[..., [0, 2]] -= gen_mdm_motions[:1, :1, [0, 2]] - mdm_motions[ticker:(ticker+1), :1, [0, 2]]
    
    # import pdb;pdb.set_trace()
    mdm_motions = fps_20_to_30(gen_mdm_motions)
############When Generate Motions ###################### 
    # 合并所有 prompts
    combined_prompts = ' '.join(prompts)
    # 清理字符串以适合作为文件名
    clean_filename = re.sub(r'[^a-zA-Z0-9_]', '_', combined_prompts)
    # 确保文件名不会过长，避免一些系统的限制
    max_filename_length = 200  # 可以根据实际情况调整
    if len(clean_filename) > max_filename_length:
        clean_filename = clean_filename[:max_filename_length]
    # file_name = clean_filename + ".npy"
    file_name = f"{output_file_path}/{clean_filename}.npy"#120frame?
    # 保存文件
    np.save(file_name, mdm_motions)
    # import pdb;pdb.set_trace()
############When Generate Motions ###################### 
    ticker = 0

async def send_to_clients(post):
    global ws_talkers
    for ws_talker in ws_talkers:
        if not ws_talker is None:
            try:
                print(f"Sending to client: {post}")
                await ws_talker.send_str(post)
            except Exception as e:
                ws_talker.close()
                ws_talkers.remove(ws_talker)
def commandline_input(args):
    global trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, fps
    ##########When Generate Motions Using TXT prompts#############
    # 替换硬编码为命令行参数
    prompts_path = args.prompts_path  # 提示文件目录
    prompts_file_names = args.prompts_file_names  # 提示文件名列表（支持多个）
    motion_save_path = args.motion_save_path  # 运动保存目录
    
    # 创建运动保存根目录（不存在则创建）
    if not os.path.exists(motion_save_path):
        os.makedirs(motion_save_path, exist_ok=True)  
    print(f"使用配置：")
    print(f"- 提示文件目录: {prompts_path}")
    print(f"- 提示文件名: {prompts_file_names}")
    print(f"- 运动保存目录: {motion_save_path}")
    # ##########When Generate Motions Using TXT prompts#############
    # prompts_path = "prompts/gmxv2"
    # # prompts_file_names = ["combat_attack.txt","combat_defense.txt","combat_movement.txt","combat_counter.txt","combat_feint_bait.txt","combat_ground_technique.txt","combat_transition_escape.txt","combat_combo.txt","combat_takedown_clinch.txt"]
    # # prompts_file_names = ["combat_movement_add.txt"]
    # # prompts_file_names = ["combat_leg.txt"]
    # prompts_file_names = ["gmxv2_difficulty_2_prompts2.txt"]
    
    # # prompts_file_names = ["boxing_attack_prompts.txt","boxing_combination_prompts.txt","boxing_defense_prompts.txt","boxing_evade_counter_prompts.txt","boxing_movement_prompts.txt"]
    # motion_save_path = "gmxv2_difficulty_2_prompts_motions"
    # 使用os.makedirs代替os.mkdir
    # import pdb;pdb.set_trace()
    if not os.path.exists(motion_save_path):
        os.makedirs(motion_save_path, exist_ok=True)  
    for prompt_file_name in prompts_file_names:
        base_name = os.path.splitext(prompt_file_name)[0]  # 获取不带后缀的文件名
        prompt_file_name_with_txt = f"{base_name}.txt"     # 拼接 .txt 后缀
        
        # 构建带 .txt 后缀的输入文件路径
        prompts_file_path = os.path.join(prompts_path, prompt_file_name_with_txt)
        
        with open(prompts_file_path, "r") as prompts_file:
            output_file_path = os.path.join(motion_save_path, prompt_file_name)
            if not os.path.exists(output_file_path):
                os.mkdir(output_file_path)
            for command in prompts_file:
                command = command.strip()
                if command != "":
                    print(command)
                    generate_text(command, output_file_path)
    print('All done! Exiting!')
    raise SystemExit(0)
    ##########When Generate Motions Using TXT prompts#############

############When Generate Motions ###################### 
    # while True:
    #     command = input('Type MDM Prompt: ')

    #     command = ""#TODO
    #     if command == 'exit':
    #         print('Exiting!')
    #         raise SystemExit(0)
    #     elif command == '':
    #         print('Empty Command!')
    #     else:
    #         generate_text(command)
############When Generate Motions ###################### 

############When Inference Motions ###################### 
    # generate_text()
############When Inference Motions ######################          

def main(request):
    return {'name': 'Andrew'}


if __name__ == "__main__":
    print("Running PHC Demo")
   
    # 解析命令行参数
    args = parser.parse_args()
    j3d, j2d, trans, dt, ws_talkers, reset_offset, offset_height, sim_talker, num_ppl = np.zeros([5, 24, 3]), None, np.zeros([3]), 1 / 10, [], True, 0.92,  None, 0
    cycle_motion, mdm_motions, ticker, to_metrabs, buffer, reset_buffer = True, np.zeros([120, 24, 3]), 0, sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix(), 120, False
    
    mdm_talker = MDMTalker()
    
    j3d = STANDING_POSE.copy()
    mdm_motions[:] = STANDING_POSE[:1].copy()
    
    frame = None
    superfast = True
    app = web.Application(client_max_size=1024**2)
    app.router.add_route('GET', '/ws', websocket_handler)
    app.router.add_route('GET', '/get_pose', pose_getter)
    app.add_routes([web.get('/', main)])
    
    print("=================================================================")
    print("r: reset offset (use r:0.91), s: start recording, e: end recording, w: write video")
    print("=================================================================")
    threading.Thread(target=commandline_input, args=(args,), daemon=True).start()
    web.run_app(app, port=8080)
    
