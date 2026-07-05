# ~/g1_motion_tracking/src/video_processing/video_processor_alignment.py
"""
专用于对齐实验的关键帧提取与拼接模块
从每段视频中等间隔提取30帧，并以更高画质保存/拼接
"""

import base64
import cv2
import io
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class AlignmentVideoProcessor:
    """
    对齐实验专用的视频处理器
    - 默认等间隔提取30帧
    - 拼接图像以及单帧保存的JPEG画质略高
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        frame_config = config.get("frame_extraction", {})
        self.total_frames = frame_config.get("alignment_total_frames", 60)
        self.target_frame_size = tuple(frame_config.get("frame_size", [240, 180]))

        # 图像质量设置，可在配置中覆盖
        self.frame_jpeg_quality = frame_config.get("alignment_frame_quality", 97)
        self.stitch_jpeg_quality = frame_config.get("alignment_stitch_quality", 98)

        self.logger.info("AlignmentVideoProcessor 初始化完成")
        self.logger.info(f"提取帧数: {self.total_frames}, 单帧JPEG质量: {self.frame_jpeg_quality}, 拼接JPEG质量: {self.stitch_jpeg_quality}")

    def process_video(self, video_path: str, output_dir: str) -> Dict:
        """
        处理视频，提取关键帧并返回拼接结果
        """
        self.logger.info(f"开始处理视频: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        temp_frames_dir = tempfile.mkdtemp(prefix="alignment_frames_")
        self.logger.debug(f"临时帧目录: {temp_frames_dir}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0

        self.logger.info(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}, 时长≈{duration:.2f}秒")

        frame_indices = self._calculate_frame_indices(total_frames, self.total_frames)
        self.logger.info(f"等间隔提取 {len(frame_indices)} 帧")

        extracted_frames: List[str] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.frame_jpeg_quality])
                extracted_frames.append(frame_path)
                self.logger.debug(f"提取帧 {frame_idx}")
            frame_idx += 1

        cap.release()

        if not extracted_frames:
            raise ValueError("未能成功提取任何帧")

        stitched_filename = f"{Path(video_path).stem}_stitched_{len(extracted_frames)}frames.jpg"
        stitched_image_path = os.path.join(output_dir, stitched_filename)
        stitched_image_base64 = self.stitch_frames_horizontal(extracted_frames, save_path=stitched_image_path)

        result = {
            "video_path": video_path,
            "video_name": Path(video_path).stem,
            "output_dir": output_dir,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "extracted_frame_count": len(extracted_frames),
            "extracted_frames": extracted_frames,
            "stitched_image_path": stitched_image_path,
            "stitched_image_base64": stitched_image_base64,
            "temp_frames_dir": temp_frames_dir,
        }
        return result

    def _calculate_frame_indices(self, total_frames: int, target_count: int) -> List[int]:
        if target_count >= total_frames:
            return list(range(total_frames))
        if total_frames <= 0:
            return []
        indices = np.linspace(0, total_frames - 1, target_count, dtype=int)
        return sorted(set(indices.tolist()))

    def stitch_frames_horizontal(self, frame_paths: List[str], save_path: Optional[str] = None) -> str:
        self.logger.info(f"拼接 {len(frame_paths)} 帧（高质量模式）...")

        images = []
        for path in frame_paths:
            try:
                img = Image.open(path)
                img = img.resize(self.target_frame_size, Image.Resampling.LANCZOS)
                images.append(img)
            except Exception as exc:
                self.logger.error(f"加载图像失败 {path}: {exc}")
                placeholder = Image.new("RGB", self.target_frame_size, color=(150, 150, 150))
                images.append(placeholder)

        frame_width, frame_height = self.target_frame_size
        margin = 2
        canvas_width = len(images) * frame_width + (len(images) + 1) * margin
        canvas_height = frame_height + 2 * margin
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

        for idx, img in enumerate(images):
            x = margin + idx * (frame_width + margin)
            canvas.paste(img, (x, margin))

        if save_path:
            canvas.save(save_path, format="JPEG", quality=self.stitch_jpeg_quality)
            self.logger.info(f"拼接图片已保存（高质量）: {save_path}")

        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=self.stitch_jpeg_quality, optimize=True)
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")
        return base64_image

    def cleanup_temp_frames(self, temp_frames_dir: str):
        if temp_frames_dir and os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            self.logger.debug(f"已清理临时目录: {temp_frames_dir}")
