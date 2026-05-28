#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import imageio
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_CODE_ROOT = REPO_ROOT / "expression_identity_gen"
if str(WAN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_CODE_ROOT))

from pipelines.wan_video import ModelConfig, WanVideoPipeline


CONDITION_LORA_NAMES = ("q_loras", "k_loras", "v_loras")


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, static, blurry details, subtitles, watermark, low quality, "
    "worst quality, jpeg artifacts, ugly, deformed face, bad anatomy, extra fingers, "
    "bad hands, distorted limbs, duplicated people, messy background"
)

DEFAULT_LORA_PATH = "train_logs/fsdp_rank128_cpu_offload/standin_condition_lora_step_003000.safetensors"


_DEBUG_START = time.perf_counter()


def debug(message: str) -> None:
    elapsed = time.perf_counter() - _DEBUG_START
    print(f"[infer-debug +{elapsed:8.1f}s] {message}", flush=True)


def debug_cuda(message: str, device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    index = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(index) / 1024**3
    reserved = torch.cuda.memory_reserved(index) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(index) / 1024**3
    debug(
        f"{message}: cuda:{index} allocated={allocated:.2f}GB "
        f"reserved={reserved:.2f}GB max_allocated={max_allocated:.2f}GB"
    )


def project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return WAN_CODE_ROOT / candidate


def existing_input_path(path: str | Path) -> Path:
    candidate = project_path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return candidate


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def init_condition_loras(
    dit: torch.nn.Module,
    rank: int,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> list[str]:
    initialized = []
    for block_idx, block in enumerate(dit.blocks):
        attn = block.self_attn
        attn.init_lora(train=False, rank=rank, device=device, dtype=dtype)
        for name in CONDITION_LORA_NAMES:
            initialized.append(f"blocks.{block_idx}.self_attn.{name}")
    if not initialized:
        raise RuntimeError("No Wan self-attention condition LoRA modules were initialized.")
    return initialized


def load_condition_lora_state_dict(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    unexpected = []
    for name, value in state.items():
        parts = name.split(".")
        if len(parts) != 6 or parts[0] != "blocks" or parts[2] != "self_attn":
            unexpected.append(name)
            continue
        block_idx = int(parts[1])
        lora_name = parts[3]
        weight_name = parts[4]
        try:
            target = getattr(getattr(module.blocks[block_idx].self_attn, lora_name), weight_name).weight
        except (AttributeError, IndexError):
            unexpected.append(name)
            continue
        if value.shape != target.shape:
            if value.numel() != target.numel():
                raise ValueError(
                    f"Cannot load {name}: checkpoint shape={tuple(value.shape)} "
                    f"target shape={tuple(target.shape)}"
                )
            value = value.reshape(target.shape)
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))
    if unexpected:
        raise KeyError(f"Unexpected condition LoRA keys in checkpoint: {unexpected[:8]}")


def crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    resized = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    return torchvision.transforms.functional.center_crop(
        resized, (target_height, target_width)
    )


def load_image(path: str | Path, height: int | None = None, width: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if height is not None and width is not None:
        image = crop_and_resize(image, height, width)
    return image


def load_video_frames(
    path: str | Path,
    num_frames: int,
    height: int | None = None,
    width: int | None = None,
) -> list[Image.Image]:
    reader = imageio.get_reader(path)
    frame_count = int(reader.count_frames())
    if frame_count <= 0:
        reader.close()
        raise ValueError(f"No frames found in {path}")
    clip_frames = min(num_frames, frame_count)
    if clip_frames == 1:
        ids = [0]
    else:
        ids = np.linspace(0, frame_count - 1, clip_frames).round().astype(int).tolist()
    frames = []
    for frame_id in ids:
        frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
        if height is not None and width is not None:
            frame = crop_and_resize(frame, height, width)
        frames.append(frame)
    reader.close()
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    return frames


def save_video(frames, save_path: str | Path, fps: int, quality: int) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(save_path), fps=fps, quality=quality)
    for frame in tqdm(frames, desc="Saving video"):
        writer.append_data(np.array(frame))
    writer.close()


def overlay_condition_winner_attention(
    frame: Image.Image,
    heat: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    mask: np.ndarray | None = None,
) -> Image.Image:
    heat_image = Image.fromarray(np.uint8(np.clip(heat, 0.0, 1.0) * 255), mode="L")
    heat_image = heat_image.resize(frame.size, resample=Image.Resampling.BICUBIC)
    heat_array = np.asarray(heat_image).astype(np.float32) / 255.0

    group_image = Image.fromarray(np.uint8(np.clip(groups, 0, 1) * 255), mode="L")
    group_image = group_image.resize(frame.size, resample=Image.Resampling.NEAREST)
    group_array = (np.asarray(group_image) >= 128)

    if mask is None:
        mask_array = np.ones_like(heat_array, dtype=np.float32)
    else:
        mask_image = Image.fromarray(np.uint8(np.clip(mask, 0.0, 1.0) * 255), mode="L")
        mask_image = mask_image.resize(frame.size, resample=Image.Resampling.BICUBIC)
        mask_array = np.asarray(mask_image).astype(np.float32) / 255.0

    frame_array = np.asarray(frame.convert("RGB")).astype(np.float32)
    color = np.zeros_like(frame_array)
    color[~group_array] = np.array([0.0, 120.0, 255.0], dtype=np.float32)
    color[group_array] = np.array([255.0, 70.0, 0.0], dtype=np.float32)
    weight = alpha * mask_array[..., None] * (0.25 + 0.75 * heat_array[..., None])
    overlay = frame_array * (1.0 - weight) + color * weight
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def expanded_box(box: np.ndarray, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    width = x2 - x1
    height = y2 - y1
    x1 -= width * margin
    x2 += width * margin
    y1 -= height * margin
    y2 += height * margin
    return np.clip(np.array([x1, y1, x2, y2], dtype=np.float32), 0.0, 1.0)


def face_box_mask(box: np.ndarray, height: int, width: int, margin: float) -> np.ndarray:
    x1, y1, x2, y2 = expanded_box(box, margin)
    yy, xx = np.meshgrid(
        (np.arange(height, dtype=np.float32) + 0.5) / height,
        (np.arange(width, dtype=np.float32) + 0.5) / width,
        indexing="ij",
    )
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    rx = max((x2 - x1) * 0.5, 1e-6)
    ry = max((y2 - y1) * 0.5, 1e-6)
    return (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0).astype(np.float32)


def attention_head_masks(
    latent_frames: int,
    patch_h: int,
    patch_w: int,
    face_boxes: torch.Tensor | None,
    fallback_face_box: tuple[float, float, float, float],
    margin: float,
) -> np.ndarray:
    if face_boxes is None:
        boxes = np.repeat(np.array(fallback_face_box, dtype=np.float32)[None], latent_frames, axis=0)
    else:
        boxes = face_boxes.detach().cpu().float().numpy()
        if boxes.ndim == 3:
            boxes = boxes[0]
        frame_ids = np.linspace(0, boxes.shape[0] - 1, latent_frames).round().astype(int)
        boxes = boxes[frame_ids]

    return np.stack(
        [face_box_mask(box, patch_h, patch_w, margin=margin) for box in boxes],
        axis=0,
    )


def save_condition_winner_attention_video(
    frames: list[Image.Image],
    attention: torch.Tensor,
    groups: torch.Tensor,
    save_path: str | Path,
    height: int,
    width: int,
    fps: int,
    quality: int,
    alpha: float,
    head_only: bool,
    face_boxes: torch.Tensor | None,
    fallback_face_box: tuple[float, float, float, float],
    face_margin: float,
) -> None:
    patch_h = height // 16
    patch_w = width // 16
    tokens_per_latent_frame = patch_h * patch_w
    if attention.numel() % tokens_per_latent_frame != 0:
        raise ValueError(
            "Cannot reshape condition attention: "
            f"{attention.numel()} tokens is not divisible by {patch_h}x{patch_w}."
        )
    if groups.numel() != attention.numel():
        raise ValueError(
            "Condition attention groups must have the same token count as attention "
            f"({groups.numel()} vs {attention.numel()})."
        )

    latent_frames = attention.numel() // tokens_per_latent_frame
    heat = attention.float().reshape(latent_frames, patch_h, patch_w).numpy()
    heat = heat - heat.min()
    heat = heat / (heat.max() + 1e-8)
    group_grid = groups.long().reshape(latent_frames, patch_h, patch_w).numpy()
    masks = np.ones_like(heat, dtype=np.float32)
    if head_only:
        masks = attention_head_masks(
            latent_frames=latent_frames,
            patch_h=patch_h,
            patch_w=patch_w,
            face_boxes=face_boxes,
            fallback_face_box=fallback_face_box,
            margin=face_margin,
        )

    frame_to_latent = np.linspace(0, latent_frames - 1, len(frames)).round().astype(int)
    overlay_frames = [
        overlay_condition_winner_attention(
            frame,
            heat[latent_index],
            group_grid[latent_index],
            alpha=alpha,
            mask=masks[latent_index],
        )
        for frame, latent_index in zip(frames, frame_to_latent)
    ]
    save_video(overlay_frames, save_path, fps=fps, quality=quality)


def load_condition_lora_file(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "lora_state_dict" in checkpoint:
            return checkpoint["lora_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def resolve_lora_path(path: str | Path) -> Path:
    raw = project_path(path)
    candidates = []
    if str(path).lower() == "latest":
        raw = project_path("train_logs/fsdp_rank128_cpu_offload")
    if raw.is_dir():
        candidates = sorted(raw.glob("standin_condition_lora_step_*.safetensors"))
    elif raw.exists():
        return raw
    else:
        raise FileNotFoundError(f"LoRA checkpoint not found: {raw}")
    if not candidates:
        raise FileNotFoundError(f"No standin_condition_lora_step_*.safetensors found in {raw}")

    def step_id(candidate: Path) -> int:
        match = re.search(r"step_(\d+)", candidate.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step_id)


def infer_lora_rank(state: dict[str, torch.Tensor], dit: torch.nn.Module | None = None) -> int:
    for name, tensor in state.items():
        if name.endswith(".down.weight"):
            if tensor.ndim == 2:
                return int(tensor.shape[0])
            if tensor.ndim == 1 and dit is not None:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "blocks":
                    block_idx = int(parts[1])
                    dim = int(module_dim := dit.blocks[block_idx].self_attn.dim)
                    if tensor.numel() % dim == 0:
                        rank = int(tensor.numel() // dim)
                        debug(
                            f"Inferred rank={rank} from flattened {name}: "
                            f"numel={tensor.numel()} dim={module_dim}"
                        )
                        return rank
            return int(tensor.shape[0])
    raise ValueError("Could not infer LoRA rank: no *.down.weight key found.")


def load_wan_pipe(args: argparse.Namespace, dtype: torch.dtype, device: str) -> WanVideoPipeline:
    base_path = project_path(args.wan_model_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Wan model path not found: {base_path}")
    shard_count = 7 if args.use_vace else 6
    diffusion_paths = [
        str(base_path / f"diffusion_pytorch_model-0000{i}-of-0000{shard_count}.safetensors")
        for i in range(1, shard_count + 1)
    ]
    model_device = "cpu" if args.enable_vram_management else device
    model_configs = [
        ModelConfig(path=diffusion_paths, offload_device=model_device, skip_download=True),
        ModelConfig(
            path=str(base_path / "models_t5_umt5-xxl-enc-bf16.pth"),
            offload_device=model_device,
            skip_download=True,
        ),
        ModelConfig(
            path=str(base_path / "Wan2.1_VAE.pth"),
            offload_device=model_device,
            skip_download=True,
        ),
    ]
    debug(f"Loading Wan pipeline from {base_path} with dtype={dtype} device={device}")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(
            path=str(base_path / "google/umt5-xxl"),
            offload_device="cpu",
            skip_download=True,
        ),
    )
    debug_cuda("after Wan pipeline load", device)
    if args.enable_vram_management:
        debug(
            "Enabling VRAM management "
            f"(limit={args.vram_limit}, buffer={args.vram_buffer}, "
            f"persistent_dit={args.num_persistent_param_in_dit})"
        )
        pipe.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit,
            vram_limit=args.vram_limit,
            vram_buffer=args.vram_buffer,
        )
        debug_cuda("after VRAM management setup", device)
    else:
        debug("VRAM management disabled")
    return pipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Wan identity/expression condition LoRA inference")
    parser.add_argument("--wan_model_path", default="checkpoints/Wan2.1/t2v")
    parser.add_argument(
        "--lora_path",
        default=DEFAULT_LORA_PATH,
        help="LoRA file, checkpoint directory, or 'latest'.",
    )
    parser.add_argument("--identity_image", required=True)
    parser.add_argument("--expression_video", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="outputs/infer.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--expression_frames", type=int, default=33)
    parser.add_argument("--identity_height", type=int, default=None)
    parser.add_argument("--identity_width", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--quality", type=int, default=9)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_vace", action="store_true")
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_stride", type=int, default=64)
    parser.add_argument("--max_expression_tokens", type=int, default=8192)
    parser.add_argument(
        "--expression_face_box",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional normalized face box. If omitted, InsightFace detection is used.",
    )
    parser.add_argument(
        "--disable_face_detection",
        action="store_true",
        help="Use the adapter fallback face box instead of running InsightFace.",
    )
    parser.add_argument("--enable_vram_management", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None)
    parser.add_argument("--vram_limit", type=float, default=None)
    parser.add_argument("--vram_buffer", type=float, default=8.0)
    parser.add_argument(
        "--condition_attention_output",
        default=None,
        help=(
            "Optional MP4 path for a combined condition-attention overlay. "
            "Blue means the winning condition token is identity; orange means expression."
        ),
    )
    parser.add_argument("--condition_attention_block", type=int, default=30)
    parser.add_argument("--condition_attention_alpha", type=float, default=0.75)
    parser.add_argument("--condition_attention_chunk_size", type=int, default=512)
    parser.add_argument("--condition_attention_head_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition_attention_face_margin", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    debug(
        "Starting inference "
        f"height={args.height} width={args.width} frames={args.num_frames} "
        f"steps={args.num_inference_steps} cfg={args.cfg_scale}"
    )
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("--height and --width must be divisible by 16.")
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("--num_frames must follow Wan's 4n+1 rule, for example 81.")

    dtype = parse_dtype(args.dtype)
    debug(f"Resolving LoRA path from {args.lora_path}")
    lora_path = resolve_lora_path(args.lora_path)
    debug(f"Using condition LoRA: {lora_path}")

    pipe = load_wan_pipe(args, dtype=dtype, device=args.device)
    debug("Loading condition LoRA state dict")
    lora_state = load_condition_lora_file(lora_path)
    debug(f"Loaded LoRA tensors: {len(lora_state)}")
    rank = infer_lora_rank(lora_state, pipe.dit)
    debug(f"Inferred LoRA rank={rank}; initializing condition LoRA modules")
    initialized_loras = init_condition_loras(pipe.dit, rank=rank, device=args.device, dtype=dtype)
    debug(f"Initialized condition LoRA modules: {len(initialized_loras)}")
    debug("Copying LoRA weights into DiT")
    load_condition_lora_state_dict(pipe.dit, lora_state)
    debug_cuda("after LoRA load", args.device)
    debug("Building condition adapter")
    condition_builder = pipe.get_condition_builder(pipe.dit)
    condition_builder.expression_adapter.max_expression_tokens = args.max_expression_tokens
    condition_builder.requires_grad_(False)
    pipe.dit.eval()
    debug("Condition adapter ready; DiT set to eval")
    if args.condition_attention_output:
        if args.condition_attention_block < 0 or args.condition_attention_block >= len(pipe.dit.blocks):
            raise ValueError(
                f"--condition_attention_block must be between 0 and {len(pipe.dit.blocks) - 1}."
            )
        attention_module = pipe.dit.blocks[args.condition_attention_block].self_attn
        attention_module.capture_condition_winner_attention = True
        attention_module.condition_attention_chunk_size = args.condition_attention_chunk_size
        debug(f"Capturing combined condition attention from DiT block {args.condition_attention_block}")

    identity_height = args.identity_height or args.height
    identity_width = args.identity_width or args.width
    identity_path = existing_input_path(args.identity_image)
    debug(f"Loading identity image: {identity_path}")
    identity_image = load_image(identity_path, identity_height, identity_width)
    debug(f"Loaded identity image size={identity_image.size}")
    expression_frames = None
    if args.expression_video:
        expression_path = existing_input_path(args.expression_video)
        debug(
            f"Loading expression video frames: {expression_path} "
            f"target_frames={args.expression_frames}"
        )
        expression_frames = load_video_frames(
            expression_path,
            args.expression_frames,
            height=args.height,
            width=args.width,
        )
        debug(f"Loaded expression frames: {len(expression_frames)} size={expression_frames[0].size}")
    expression_face_boxes = args.expression_face_box
    if args.disable_face_detection and expression_face_boxes is None:
        expression_face_boxes = condition_builder.expression_adapter.fallback_face_box
        debug(f"Face detection disabled; using fallback box={expression_face_boxes}")
    elif expression_face_boxes is not None:
        debug(f"Using provided expression face box={expression_face_boxes}")
    else:
        debug("Expression face box not provided; pipeline will run automatic face detection")

    debug("Calling Wan pipeline")
    with torch.inference_mode():
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            tiled=args.tiled,
            tile_size=(args.tile_size, args.tile_size),
            tile_stride=(args.tile_stride, args.tile_stride),
            identity_reference_image=identity_image,
            expression_reference_frames=expression_frames,
            expression_face_boxes=expression_face_boxes,
        )
    debug_cuda("after Wan pipeline call", args.device)
    debug(f"Pipeline returned {len(video)} frames; saving to {args.output}")
    save_video(video, args.output, fps=args.fps, quality=args.quality)
    debug(f"Saved video to {args.output}")
    generated_face_boxes = None
    needs_generated_face_boxes = (
        args.condition_attention_output and args.condition_attention_head_only
    )
    if needs_generated_face_boxes:
        debug("Detecting generated face boxes for head-only attention overlay")
        generated_face_boxes = pipe.detect_expression_face_boxes(video)
        if generated_face_boxes is None:
            debug("Generated face detection failed; using fallback head box")

    if args.condition_attention_output:
        attention_module = pipe.dit.blocks[args.condition_attention_block].self_attn
        attention = getattr(attention_module, "last_condition_attention", None)
        groups = getattr(attention_module, "last_condition_attention_group", None)
        if attention is None or groups is None:
            raise RuntimeError(
                "Combined condition attention was not captured. Check that identity_image "
                "and expression_video are set and condition tokens reached the selected block."
            )
        debug(f"Saving combined condition attention overlay to {args.condition_attention_output}")
        save_condition_winner_attention_video(
            video,
            attention[0],
            groups[0],
            args.condition_attention_output,
            height=args.height,
            width=args.width,
            fps=args.fps,
            quality=args.quality,
            alpha=args.condition_attention_alpha,
            head_only=args.condition_attention_head_only,
            face_boxes=generated_face_boxes,
            fallback_face_box=condition_builder.expression_adapter.fallback_face_box,
            face_margin=args.condition_attention_face_margin,
        )
        debug(f"Saved combined condition attention overlay to {args.condition_attention_output}")


if __name__ == "__main__":
    main()
