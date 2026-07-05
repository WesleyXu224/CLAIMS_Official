import base64
import io
import json
import logging
import os
import re
import time
from typing import Dict, List

from openai import OpenAI
from PIL import Image


class MotionAnalyzerV2:
    def __init__(self, config: dict, provider_name: str):
        self.config = config
        self.provider_name = provider_name
        self.logger = logging.getLogger(__name__)

        provider_cfg = self.config["providers"][provider_name]
        api_key_env = provider_cfg.get("api_key_env", "VLM_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Missing required environment variable: {api_key_env}")

        self.client = OpenAI(
            api_key=api_key,
            base_url=provider_cfg["base_url"],
        )
        self.provider_cfg = provider_cfg
        self.target_frame_size = tuple(self.config.get("frame_extraction", {}).get("frame_size", [240, 180]))

    def stitch_frames_horizontal(self, frame_paths: List[str], save_path: str = None) -> str:
        images = []
        for path in frame_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(self.target_frame_size, Image.Resampling.LANCZOS)
                images.append(img)
            except Exception:
                images.append(Image.new("RGB", self.target_frame_size, color=(128, 128, 128)))

        frame_width, frame_height = self.target_frame_size
        margin = 2
        canvas_width = len(images) * frame_width + (len(images) + 1) * margin
        canvas_height = frame_height + 2 * margin
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))

        for idx, img in enumerate(images):
            x = margin + idx * (frame_width + margin)
            canvas.paste(img, (x, margin))

        if save_path:
            canvas.save(save_path, format="JPEG", quality=95)

        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=90, optimize=True)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def create_analysis_prompt(self, motion_id: str) -> str:
        return f"""You are a professional motion-difficulty analyst.
Analyze these 15 sequential frames from left to right as one motion sequence.
Motion ID: {motion_id}

Score the motion difficulty conservatively on a 1-10 scale.

Guidelines:
- 1-2: very basic single-step or low-skill movement
- 3-4: simple multi-step movement with limited coordination demand
- 5-6: moderate coordination, timing, or balance demand
- 7-8: advanced multi-phase movement, explosive actions, jump/spin, or strong control
- 9-10: highly complex expert-level movement with multiple advanced components

Return only valid JSON in this exact schema:
{{
  "video_name": "string",
  "action_name": "short action summary",
  "difficulty_score": 1,
  "analysis": {{
    "action_sequence": "string",
    "technical_complexity": "string",
    "movement_intensity": "string",
    "balance_requirement": "string",
    "continuity": "string"
  }},
  "scoring_reason": "string",
  "feedback": {{
    "description": "string",
    "key_events": "string",
    "dynamism_description": "string",
    "complexity_description": "string",
    "difficulty_description": "string",
    "increase_dynamism_suggestion": "string",
    "increase_complexity_suggestion": "string",
    "increase_difficulty_suggestion": "string"
  }}
}}

Requirements:
- Output JSON only, with no markdown.
- Keep `difficulty_score` as an integer.
- Keep `action_name` concise.
- Use objective language.
- Keep `analysis` focused on difficulty-relevant attributes.
- Put richer narrative feedback into `feedback`."""

    def _extract_json_from_text(self, text: str) -> Dict:
        brace_stack = []
        start_idx = -1
        for i, char in enumerate(text):
            if char == "{":
                if not brace_stack:
                    start_idx = i
                brace_stack.append(char)
            elif char == "}":
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx != -1:
                        candidate = text[start_idx:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            cleaned = re.sub(r"[\x00-\x1f]", "", candidate)
                            try:
                                return json.loads(cleaned)
                            except json.JSONDecodeError:
                                continue
        raise ValueError("Could not extract valid JSON from model response.")

    def analyze_motion(self, frame_paths: List[str], video_name: str, output_dir: str) -> Dict:
        stitched_path = os.path.join(output_dir, f"{video_name}_stitched_15frames.jpg")
        base64_image = self.stitch_frames_horizontal(frame_paths, save_path=stitched_path)
        motion_id = f"motion_{abs(hash(video_name)) % 10**8:08d}"
        prompt = self.create_analysis_prompt(motion_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.provider_cfg["model"],
            messages=messages,
            max_tokens=self.provider_cfg.get("max_tokens", 3000),
            temperature=self.provider_cfg.get("temperature", 0.0),
        )
        content = response.choices[0].message.content
        result = self._extract_json_from_text(content)
        result["video_name"] = video_name
        return result
