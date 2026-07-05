import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import smplx
import torch


DEFAULT_MODEL_DIRS = [
    os.environ.get("SMPL_MODEL_DIR", ""),
]


def resolve_model_dir(user_model_dir: Optional[str]) -> str:
    if user_model_dir:
        if not os.path.exists(user_model_dir):
            raise FileNotFoundError(f"SMPL model directory not found: {user_model_dir}")
        return user_model_dir

    for model_dir in DEFAULT_MODEL_DIRS:
        if model_dir and os.path.exists(model_dir):
            return model_dir

    raise FileNotFoundError(
        "Could not resolve an SMPL model directory. "
        "Pass --model_dir explicitly or set SMPL_MODEL_DIR."
    )


def load_gender(npz_gender, fallback_gender: str) -> str:
    if npz_gender is None:
        return fallback_gender
    if isinstance(npz_gender, np.ndarray):
        if npz_gender.ndim == 0:
            npz_gender = npz_gender.item()
        elif npz_gender.size > 0:
            npz_gender = npz_gender.reshape(-1)[0]
    if isinstance(npz_gender, bytes):
        npz_gender = npz_gender.decode("utf-8")
    gender = str(npz_gender).strip().lower()
    return gender if gender else fallback_gender


def build_model(model_dir: str, model_type: str, gender: str, device: torch.device):
    if model_type != "smpl":
        return smplx.create(
            model_path=model_dir,
            model_type=model_type,
            gender=gender,
            use_pca=False,
        ).to(device)

    gender_to_file = {
        "neutral": "SMPL_NEUTRAL.pkl",
        "male": "SMPL_MALE.pkl",
        "female": "SMPL_FEMALE.pkl",
    }
    model_file = os.path.join(model_dir, gender_to_file.get(gender, "SMPL_NEUTRAL.pkl"))
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"SMPL model file not found: {model_file}")

    return smplx.create(
        model_path=model_file,
        model_type=model_type,
        gender=gender,
        use_pca=False,
    ).to(device)


def render_single_npz(
    npz_file: str,
    model_dir: str,
    output_video: str,
    frames_dir: Optional[str],
    fps: int,
    model_type: str,
    default_gender: str,
    width: int,
    height: int,
):
    data = np.load(npz_file)
    poses = data["poses"]
    trans = data["trans"]
    betas = data["betas"]
    gender = load_gender(data["gender"] if "gender" in data else None, default_gender)

    if betas.ndim == 2:
        betas = betas[0]

    output_video = str(output_video)
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    if frames_dir is not None:
        os.makedirs(frames_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_dir, model_type, gender, device)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.72, 0.87, 0.98, 1.0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_idx in range(len(poses)):
        pose_frame = torch.tensor(poses[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        transl = torch.tensor(trans[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
        betas_t = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)

        output = model(
            betas=betas_t,
            body_pose=pose_frame[:, 3:],
            global_orient=pose_frame[:, :3],
            transl=transl,
        )

        vertices = output.vertices[0].detach().cpu().numpy()
        faces = model.faces.astype(np.int32)
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices),
            triangles=o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()

        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("smpl_mesh", mesh, material)

        bbox = mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        extent = max(bbox.get_extent())
        cam_pos = center + np.array([0.0, -extent * 2.2, extent * 0.8])
        cam_up = np.array([0.0, 0.0, 1.0])
        renderer.scene.camera.look_at(center, cam_pos, cam_up)

        image = np.asarray(renderer.render_to_image())
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        if frames_dir is not None:
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_path, image_bgr)

        video_writer.write(image_bgr)

    video_writer.release()
    renderer = None

    print(f"Rendered {npz_file} -> {output_video}")


def render_directory(
    input_dir: str,
    model_dir: str,
    output_dir: str,
    frames_root: Optional[str],
    fps: int,
    model_type: str,
    default_gender: str,
    width: int,
    height: int,
):
    input_path = Path(input_dir)
    npz_files = sorted(input_path.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    if frames_root is not None:
        os.makedirs(frames_root, exist_ok=True)

    for npz_file in npz_files:
        stem = npz_file.stem
        output_video = os.path.join(output_dir, f"{stem}.mp4")
        frames_dir = os.path.join(frames_root, stem) if frames_root is not None else None
        render_single_npz(
            npz_file=str(npz_file),
            model_dir=model_dir,
            output_video=output_video,
            frames_dir=frames_dir,
            fps=fps,
            model_type=model_type,
            default_gender=default_gender,
            width=width,
            height=height,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Render AMASS-style npz motions into mp4 videos.")
    parser.add_argument("--input_dir", help="Directory containing .npz motion files.")
    parser.add_argument("--npz_file", help="Single .npz motion file to render.")
    parser.add_argument("--output_dir", help="Directory for output videos when using --input_dir.")
    parser.add_argument("--output_video", help="Output mp4 path when using --npz_file.")
    parser.add_argument("--frames_root", default=None, help="Optional root directory for per-frame PNG outputs.")
    parser.add_argument("--model_dir", default=None, help="SMPL model directory. If omitted, use SMPL_MODEL_DIR when available.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--model_type", default="smpl")
    parser.add_argument("--default_gender", default="neutral")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model_dir)

    if bool(args.input_dir) == bool(args.npz_file):
        raise ValueError("Pass exactly one of --input_dir or --npz_file.")

    if args.input_dir:
        if not args.output_dir:
            raise ValueError("--output_dir is required when using --input_dir.")
        render_directory(
            input_dir=args.input_dir,
            model_dir=model_dir,
            output_dir=args.output_dir,
            frames_root=args.frames_root,
            fps=args.fps,
            model_type=args.model_type,
            default_gender=args.default_gender,
            width=args.width,
            height=args.height,
        )
    else:
        if not args.output_video:
            raise ValueError("--output_video is required when using --npz_file.")
        render_single_npz(
            npz_file=args.npz_file,
            model_dir=model_dir,
            output_video=args.output_video,
            frames_dir=args.frames_root,
            fps=args.fps,
            model_type=args.model_type,
            default_gender=args.default_gender,
            width=args.width,
            height=args.height,
        )


if __name__ == "__main__":
    main()
