import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


class DifficultyVideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.total_frames = config.get("frame_extraction", {}).get("total_frames", 15)

    def process_video(self, video_path: str, output_dir: str) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        temp_frames_dir = tempfile.mkdtemp(prefix="frames_temp_")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else 0.0
        frame_indices = self._calculate_frame_indices(total_frames, self.total_frames)

        extracted_frames: List[str] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_indices:
                frame_path = os.path.join(temp_frames_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_frames.append(frame_path)
            frame_idx += 1

        cap.release()

        return {
            "video_path": video_path,
            "video_name": Path(video_path).stem,
            "output_dir": output_dir,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "extracted_frame_count": len(extracted_frames),
            "extracted_frames": extracted_frames,
            "temp_frames_dir": temp_frames_dir,
        }

    def _calculate_frame_indices(self, total_frames: int, target_count: int) -> List[int]:
        if target_count >= total_frames:
            return list(range(total_frames))
        return np.linspace(0, total_frames - 1, target_count, dtype=int).tolist()

    def cleanup_temp_frames(self, temp_frames_dir: str):
        if os.path.exists(temp_frames_dir):
            shutil.rmtree(temp_frames_dir)
