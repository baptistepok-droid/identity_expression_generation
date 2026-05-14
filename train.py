#!/usr/bin/env python3
"""
TOKENIZERS_PARALLELISM=false accelerate launch --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no train.py --dataset_base_path /home/ens.old/Bpokrzywa/datasets/dataset/MEAD --dataset_metadata_path datasets/mead_identity_smoke.csv --wan_model_path checkpoints/Wan2.1/t2v --num_frames 81 --max_steps 500 --save_every 50 --batch_size 1 --target_effective_batch_size 4 --gradient_accumulation_steps 1 --rank 4 --dtype bf16 --gradient_checkpointing --gradient_checkpointing_offload --vram_buffer 8  | tee train_logs/run.log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import imageio
import torch
import torchvision
from accelerate import Accelerator
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_CODE_ROOT = REPO_ROOT / "expression_identity_gen"
if str(WAN_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_CODE_ROOT))

from pipelines.wan_video import ModelConfig, WanVideoPipeline 
CONDITION_LORA_NAMES = ("q_loras", "k_loras", "v_loras")


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


def is_main(accelerator) -> bool:
    return accelerator is None or accelerator.is_main_process


def unwrap_fsdp_module(module: nn.Module) -> nn.Module:
    return getattr(module, "_fsdp_wrapped_module", module)


@contextmanager
def summon_full_params_if_fsdp(module: nn.Module, writeback: bool = False):
    if hasattr(module, "_fsdp_wrapped_module"):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.summon_full_params(module, writeback=writeback, recurse=False):
            yield
    else:
        yield


def init_condition_loras(dit: nn.Module, train: bool, rank: int) -> list[str]:
    initialized = []
    for block_idx, block in enumerate(dit.blocks):
        block = unwrap_fsdp_module(block)
        attn = block.self_attn
        attn.init_lora(train=train, rank=rank)
        for name in CONDITION_LORA_NAMES:
            initialized.append(f"blocks.{block_idx}.self_attn.{name}")
    if not initialized:
        raise RuntimeError("No Wan self-attention condition LoRA modules were initialized.")
    return initialized


def condition_lora_parameters(dit: nn.Module) -> list[nn.Parameter]:
    params = []
    for block in dit.blocks:
        block = unwrap_fsdp_module(block)
        attn = block.self_attn
        for name in CONDITION_LORA_NAMES:
            if not hasattr(attn, name):
                continue
            params.extend(getattr(attn, name).parameters())
    return params


def export_condition_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    if hasattr(module, "blocks"):
        for block_idx, block in enumerate(module.blocks):
            with summon_full_params_if_fsdp(block):
                block = unwrap_fsdp_module(block)
                attn = block.self_attn
                for lora_name in CONDITION_LORA_NAMES:
                    if not hasattr(attn, lora_name):
                        continue
                    submodule = getattr(attn, lora_name)
                    prefix = f"blocks.{block_idx}.self_attn.{lora_name}"
                    state[f"{prefix}.down.weight"] = submodule.down.weight.detach().cpu()
                    state[f"{prefix}.up.weight"] = submodule.up.weight.detach().cpu()
        return state
    for name, submodule in module.named_modules():
        if name.rsplit(".", 1)[-1] in CONDITION_LORA_NAMES:
            state[f"{name}.down.weight"] = submodule.down.weight.detach().cpu()
            state[f"{name}.up.weight"] = submodule.up.weight.detach().cpu()
    return state


def load_condition_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    if hasattr(module, "blocks"):
        missing = []
        for name, value in state.items():
            parts = name.split(".")
            if len(parts) != 6 or parts[0] != "blocks" or parts[2] != "self_attn":
                missing.append(name)
                continue
            block_idx = int(parts[1])
            lora_name = parts[3]
            weight_name = parts[4]
            block = module.blocks[block_idx]
            with summon_full_params_if_fsdp(block, writeback=True):
                block = unwrap_fsdp_module(block)
                if not hasattr(block.self_attn, lora_name):
                    missing.append(name)
                    continue
                target = getattr(getattr(block.self_attn, lora_name), weight_name).weight
                target.data.copy_(value.to(device=target.device, dtype=target.dtype))
        if missing:
            raise KeyError(f"Unexpected condition LoRA keys in checkpoint: {missing[:8]}")
        return
    model_state = module.state_dict()
    missing = []
    for name, value in state.items():
        if name not in model_state:
            missing.append(name)
            continue
        model_state[name].copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))
    if missing:
        raise KeyError(f"Unexpected condition LoRA keys in checkpoint: {missing[:8]}")


def save_lora(path: Path, module: nn.Module, metadata: dict[str, str], accelerator) -> None:
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = export_condition_lora_state_dict(module)
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(state, str(path), metadata=metadata)
    else:
        torch.save({"state_dict": state, "metadata": metadata}, path)


@dataclass
class TrainSample:
    video: list[Image.Image]
    prompt: str
    identity_image: Image.Image | None = None
    expression_video: list[Image.Image] | None = None


class WanVideoTrainingDataset(Dataset):
    image_extensions = ("jpg", "jpeg", "png", "webp")
    video_extensions = ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm")

    def __init__(
        self,
        base_path: str | Path,
        metadata_path: str | Path | None,
        height: int | None,
        width: int | None,
        num_frames: int,
        expression_frames: int,
        max_pixels: int,
        repeat: int,
    ) -> None:
        self.base_path = Path(base_path)
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.expression_frames = expression_frames
        self.max_pixels = max_pixels
        self.repeat = repeat
        self.dynamic_resolution = height is None and width is None
        self.rows = self._load_metadata(metadata_path)
        if not self.rows:
            raise ValueError("Training dataset is empty.")

    def __len__(self) -> int:
        return len(self.rows) * self.repeat

    def __getitem__(self, index: int) -> TrainSample:
        row = self.rows[index % len(self.rows)]
        video_key = row.get("video") or row.get("video_path")
        if not video_key:
            raise ValueError("Each metadata row must contain video or video_path.")
        return TrainSample(
            video=self.load_media(self.resolve(video_key), self.num_frames),
            prompt=row.get("prompt", ""),
            identity_image=self.load_image(self.resolve(row["identity_image"])) if row.get("identity_image") else None,
            expression_video=(
                self.load_media(self.resolve(row["expression_video"]), self.expression_frames)
                if row.get("expression_video")
                else None
            ),
        )

    def resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.base_path / path

    def _load_metadata(self, metadata_path: str | Path | None) -> list[dict]:
        if metadata_path is None:
            return self.generate_metadata()
        path = Path(metadata_path)
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text())
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def generate_metadata(self) -> list[dict]:
        rows = []
        files = {path.name for path in self.base_path.iterdir() if path.is_file()}
        for file_name in sorted(files):
            suffix = file_name.rsplit(".", 1)[-1].lower()
            if suffix not in self.image_extensions and suffix not in self.video_extensions:
                continue
            stem = file_name[: -len(suffix) - 1]
            prompt_file = f"{stem}.txt"
            if prompt_file not in files:
                continue
            rows.append(
                {
                    "video": file_name,
                    "prompt": (self.base_path / prompt_file).read_text(encoding="utf-8").strip(),
                }
            )
        return rows

    def target_size(self, image: Image.Image) -> tuple[int, int]:
        if not self.dynamic_resolution:
            return int(self.height), int(self.width)
        width, height = image.size
        if width * height > self.max_pixels:
            scale = math.sqrt(width * height / self.max_pixels)
            width = int(width / scale)
            height = int(height / scale)
        height = max(16, height // 16 * 16)
        width = max(16, width // 16 * 16)
        return height, width

    def crop_and_resize(self, image: Image.Image, target_height: int, target_width: int) -> Image.Image:
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        resized = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return torchvision.transforms.functional.center_crop(resized, (target_height, target_width))

    def load_image(self, path: Path) -> Image.Image:
        image = Image.open(path).convert("RGB")
        height, width = self.target_size(image)
        return self.crop_and_resize(image, height, width)

    def load_media(self, path: Path, num_frames: int) -> list[Image.Image]:
        suffix = path.suffix.lower().lstrip(".")
        if suffix in self.image_extensions:
            return [self.load_image(path)]
        reader = imageio.get_reader(path)
        frame_count = int(reader.count_frames())
        if frame_count <= 0:
            reader.close()
            raise ValueError(f"No frames found in {path}")
        clip_frames = min(num_frames, frame_count)
        start = random.randint(0, frame_count - clip_frames) if frame_count > clip_frames else 0
        ids = list(range(start, start + clip_frames))
        frames = []
        target_size = None
        for frame_id in ids:
            frame = Image.fromarray(reader.get_data(frame_id)).convert("RGB")
            if target_size is None:
                target_size = self.target_size(frame)
            frames.append(self.crop_and_resize(frame, *target_size))
        reader.close()
        if len(frames) < num_frames:
            frames.extend([frames[-1]] * (num_frames - len(frames)))
        return frames


def collate_one(samples: list[TrainSample]) -> TrainSample | None:
    samples = [sample for sample in samples if sample is not None]
    return samples[0] if samples else None


def collate_samples(samples: list[TrainSample]) -> TrainSample | list[TrainSample] | None:
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return None
    return samples[0] if len(samples) == 1 else samples


def load_wan_pipe(args: argparse.Namespace, dtype: torch.dtype, device: torch.device | str) -> WanVideoPipeline:
    base_path = Path(args.wan_model_path)
    shard_count = 7 if args.use_vace else 6
    diffusion_paths = [
        str(base_path / f"diffusion_pytorch_model-0000{i}-of-0000{shard_count}.safetensors")
        for i in range(1, shard_count + 1)
    ]
    model_configs = [
        ModelConfig(path=diffusion_paths, offload_device="cpu", skip_download=True),
        ModelConfig(path=str(base_path / "models_t5_umt5-xxl-enc-bf16.pth"), offload_device="cpu", skip_download=True),
        ModelConfig(path=str(base_path / "Wan2.1_VAE.pth"), offload_device="cpu", skip_download=True),
    ]
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
    pipe.scheduler.set_timesteps(args.num_train_timesteps, training=True)
    if args.enable_vram_management and not args.fsdp_shard_dit:
        pipe.enable_vram_management(
            num_persistent_param_in_dit=args.num_persistent_param_in_dit,
            vram_limit=args.vram_limit,
            vram_buffer=args.vram_buffer,
        )
    return pipe


class WanConditionLoRATrainingModule(nn.Module):
    def __init__(self, pipe: WanVideoPipeline, args: argparse.Namespace) -> None:
        super().__init__()
        self.pipe = pipe
        self.args = args
        self.pipe.dit.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.condition_loras = init_condition_loras(self.pipe.dit, train=True, rank=args.rank)
        self.condition_lora_param_count = sum(param.numel() for param in condition_lora_parameters(self.pipe.dit))
        self.pipe.dit.train()
        self.pipe.text_encoder.eval()
        self.pipe.vae.eval()

    def forward(self, sample: TrainSample) -> torch.Tensor:
        if sample is None:
            return torch.zeros((), device=self.pipe.device, dtype=torch.float32)
        samples = sample if isinstance(sample, list) else [sample]
        losses = []
        for item in samples:
            inputs = self.forward_preprocess(item)
            if not self.args.fsdp_shard_dit:
                self.pipe.load_models_to_device(["dit"])
            losses.append(
                self.pipe.training_loss(
                    **inputs,
                    min_timestep_boundary=self.args.min_timestep_boundary,
                    max_timestep_boundary=self.args.max_timestep_boundary,
                )
            )
        return torch.stack(losses).mean()

    def forward_preprocess(self, sample: TrainSample) -> dict:
        height, width = sample.video[0].height, sample.video[0].width
        with torch.no_grad():
            if self.args.fsdp_shard_dit:
                self.pipe.text_encoder.to(self.pipe.device)
                self.pipe.vae.to(self.pipe.device)
            else:
                self.pipe.load_models_to_device(["text_encoder", "vae"])
            context = self.pipe.prompter.encode_prompt(
                sample.prompt,
                positive=True,
                device=self.pipe.device,
            )
            video = self.pipe.preprocess_video(sample.video)
            input_latents = self.pipe.vae.encode(
                video,
                device=self.pipe.device,
                tiled=self.args.tiled,
                tile_size=(self.args.tile_size, self.args.tile_size),
                tile_stride=(self.args.tile_stride, self.args.tile_stride),
            ).to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            noise = torch.randn_like(input_latents)

            identity_latents = None
            if sample.identity_image is not None:
                identity_latents = self.pipe.encode_ip_image(sample.identity_image)

            expression_latents = None
            expression_face_boxes = None
            if sample.expression_video is not None:
                expression_latents = self.pipe.encode_condition_frames(sample.expression_video, height, width)
                if self.args.detect_expression_boxes:
                    expression_face_boxes = self.pipe.detect_expression_face_boxes(sample.expression_video)
            if self.args.fsdp_shard_dit:
                self.pipe.text_encoder.cpu()
                self.pipe.vae.cpu()
                torch.cuda.empty_cache()

        condition_builder = None
        if identity_latents is not None or expression_latents is not None:
            condition_builder = self.pipe.get_condition_builder(self.pipe.dit)
            condition_builder.requires_grad_(False)

        else:
            raise ValueError(
                "Stand-In condition LoRA training requires identity_image or expression_video "
                "in each metadata row; otherwise q_loras/k_loras/v_loras are not used."
            )

        return {
            "dit": self.pipe.dit,
            "latents": noise,
            "input_latents": input_latents,
            "noise": noise,
            "context": context,
            "condition_builder": condition_builder,
            "identity_latents": identity_latents,
            "expression_latents": expression_latents,
            "expression_face_boxes": expression_face_boxes,
            "clip_feature": None,
            "y": None,
            "reference_latents": None,
            "control_camera_latents_input": None,
            "use_gradient_checkpointing": self.args.gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.args.gradient_checkpointing_offload,
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return condition_lora_parameters(self.pipe.dit)


def save_training_state(
    accelerator: Accelerator,
    model: WanConditionLoRATrainingModule,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(model)
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lora_state = export_condition_lora_state_dict(unwrapped.pipe.dit)
    torch.save(
        {
            "step": step,
            "lora_state_dict": lora_state,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(args),
        },
        checkpoint_dir / "training_state.pt",
    )
    save_lora(
        output_dir / f"standin_condition_lora_step_{step:06d}{args.output_suffix}",
        unwrapped.pipe.dit,
        {
            "base": "WanVideo",
            "targets": ",".join(CONDITION_LORA_NAMES),
            "rank": str(args.rank),
            "standin_condition_lora": "true",
        },
        accelerator,
    )


def load_training_state(
    path: str | None,
    model: WanConditionLoRATrainingModule,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    accelerator: Accelerator,
) -> int:
    if not path:
        return 0
    checkpoint = torch.load(path, map_location="cpu")
    unwrapped = accelerator.unwrap_model(model)
    load_condition_lora_state_dict(unwrapped.pipe.dit, checkpoint["lora_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    return int(checkpoint.get("step", 0))


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def move_parameters_to_device(params: list[nn.Parameter], device: torch.device, dtype: torch.dtype) -> None:
    for param in params:
        param.data = param.data.to(device=device, dtype=dtype)
        if param.grad is not None:
            param.grad = param.grad.to(device=device, dtype=dtype)


def move_attention_norms_to_device(pipe: WanVideoPipeline, device: torch.device, dtype: torch.dtype) -> None:
    for dit in (getattr(pipe, "dit", None), getattr(pipe, "dit2", None)):
        if dit is None:
            continue
        for block in getattr(dit, "blocks", []):
            block = unwrap_fsdp_module(block)
            attn = getattr(block, "self_attn", None)
            if attn is None:
                continue
            for name in ("norm_q", "norm_k"):
                module = getattr(attn, name, None)
                if module is not None:
                    module.to(device=device, dtype=dtype)


def enable_fsdp_for_dit(
    pipe: WanVideoPipeline,
    accelerator: Accelerator,
    dtype: torch.dtype,
    cpu_offload: bool,
    wrap_policy: str,
) -> None:
    if not accelerator.distributed_type.name.upper().endswith("MULTI_GPU") and accelerator.num_processes <= 1:
        raise RuntimeError("--fsdp_shard_dit requires a multi-GPU distributed launch.")
    try:
        from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    except ImportError as exc:
        raise RuntimeError("This PyTorch build does not provide FSDP.") from exc

    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    offload = CPUOffload(offload_params=True) if cpu_offload else None

    def wrap(module: nn.Module) -> FSDP:
        return FSDP(
            module,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            cpu_offload=offload,
            device_id=accelerator.device,
            use_orig_params=True,
            limit_all_gathers=True,
        )

    if wrap_policy == "block":
        for name in ("patch_embedding", "text_embedding", "time_embedding", "time_projection", "head"):
            module = getattr(pipe.dit, name, None)
            if module is not None:
                module.to(device=accelerator.device, dtype=dtype)
        for index, block in enumerate(pipe.dit.blocks):
            pipe.dit.blocks[index] = wrap(block)
        return

    if wrap_policy != "linear":
        raise ValueError(f"Unsupported FSDP wrap policy: {wrap_policy}")

    def wrap_linear_children(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, FSDP):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, wrap(child))
            else:
                wrap_linear_children(child)

    def move_unwrapped_tensors(module: nn.Module) -> None:
        if isinstance(module, FSDP):
            return
        for param in module.parameters(recurse=False):
            param.data = param.data.to(device=accelerator.device, dtype=dtype)
            if param.grad is not None:
                param.grad = param.grad.to(device=accelerator.device, dtype=dtype)
        for buffer_name, buffer in module.named_buffers(recurse=False):
            setattr(module, buffer_name, buffer.to(device=accelerator.device))
        for child in module.children():
            move_unwrapped_tensors(child)

    wrap_linear_children(pipe.dit)
    move_unwrapped_tensors(pipe.dit)


def sync_gradients(params: list[nn.Parameter], accelerator: Accelerator) -> None:
    if accelerator.num_processes <= 1:
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(accelerator.num_processes)


def broadcast_parameters(params: list[nn.Parameter], accelerator: Accelerator) -> None:
    if accelerator.num_processes <= 1:
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    for param in params:
        dist.broadcast(param.data, src=0)


def print_cuda_memory(tag: str, accelerator: Accelerator) -> None:
    if not torch.cuda.is_available() or not accelerator.is_local_main_process:
        return
    device = accelerator.device
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    print(
        f"[cuda-mem][rank {accelerator.process_index}] {tag}: "
        f"allocated={allocated / 1024**3:.2f}GB "
        f"reserved={reserved / 1024**3:.2f}GB "
        f"max_allocated={max_allocated / 1024**3:.2f}GB "
        f"free={free / 1024**3:.2f}GB "
        f"total={total / 1024**3:.2f}GB",
        flush=True,
    )


def cuda_memory_profile_enabled() -> bool:
    return os.environ.get("WAN_MEMORY_PROFILE", "").lower() in {"1", "true", "yes", "on"}


def start_cuda_memory_history(accelerator: Accelerator) -> None:
    if not cuda_memory_profile_enabled() or not torch.cuda.is_available():
        return
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            stacks="all",
            max_entries=100000,
        )
        if accelerator.is_local_main_process:
            print("[cuda-mem] recording CUDA memory history", flush=True)
    except TypeError:
        torch.cuda.memory._record_memory_history(enabled=True)
    except Exception as exc:
        if accelerator.is_local_main_process:
            print(f"[cuda-mem] memory history unavailable: {exc}", flush=True)


def dump_cuda_oom_debug(output_dir: Path, accelerator: Accelerator, tag: str) -> None:
    if not torch.cuda.is_available():
        return
    rank = accelerator.process_index
    snapshot_dir = output_dir / "memory_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{tag}_rank{rank}.pickle"
    summary_path = snapshot_dir / f"{tag}_rank{rank}.txt"
    try:
        torch.cuda.synchronize(accelerator.device)
    except Exception:
        pass
    try:
        summary_path.write_text(torch.cuda.memory_summary(accelerator.device, abbreviated=False))
        print(f"[cuda-mem][rank {rank}] wrote memory summary: {summary_path}", flush=True)
    except Exception as exc:
        print(f"[cuda-mem][rank {rank}] failed to write memory summary: {exc}", flush=True)
    if cuda_memory_profile_enabled():
        try:
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            print(f"[cuda-mem][rank {rank}] wrote memory snapshot: {snapshot_path}", flush=True)
        except Exception as exc:
            print(f"[cuda-mem][rank {rank}] failed to write memory snapshot: {exc}", flush=True)


def train(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grad_accum = args.gradient_accumulation_steps
    if grad_accum <= 0:
        grad_accum = max(1, math.ceil(args.target_effective_batch_size / (world_size * args.batch_size)))
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed + accelerator.process_index)
    random.seed(args.seed + accelerator.process_index)
    dtype = parse_dtype(args.dtype)
    output_dir = Path(args.output_dir)

    dataset = WanVideoTrainingDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path or args.manifest,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        expression_frames=args.expression_frames,
        max_pixels=args.max_pixels,
        repeat=args.dataset_repeat,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_samples,
        pin_memory=args.pin_memory,
    )

    pipe = load_wan_pipe(args, dtype=dtype, device=accelerator.device)
    model = WanConditionLoRATrainingModule(pipe, args)
    if args.fsdp_shard_dit:
        enable_fsdp_for_dit(
            pipe,
            accelerator,
            dtype=dtype,
            cpu_offload=args.fsdp_cpu_offload,
            wrap_policy=args.fsdp_wrap_policy,
        )
    print_cuda_memory("after model load", accelerator)
    params = model.trainable_parameters()
    if not params:
        raise RuntimeError("No trainable Stand-In condition LoRA parameters found.")
    if not args.fsdp_shard_dit or args.fsdp_wrap_policy == "linear":
        move_parameters_to_device(params, accelerator.device, dtype)
        move_attention_norms_to_device(pipe, accelerator.device, dtype)
        broadcast_parameters(params, accelerator)
    print_cuda_memory("after LoRA/support move to GPU", accelerator)
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )
    print_cuda_memory("after optimizer init", accelerator)
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps, args.max_steps)

    if accelerator.is_main_process:
        print(f"Initialized {len(model.condition_loras)} Stand-In condition LoRA modules.")
        print("Example targets:", ", ".join(model.condition_loras[:6]))
        print(f"Trainable condition LoRA parameters: {model.condition_lora_param_count:,}")
        print(f"World size: {accelerator.num_processes}")
        print(f"Gradient accumulation: {grad_accum}")
        print(f"Batch size per process: {args.batch_size}")
        print(f"Effective batch size: {accelerator.num_processes * args.batch_size * grad_accum}")

    dataloader = accelerator.prepare(dataloader)
    print_cuda_memory("after dataloader prepare", accelerator)
    global_step = load_training_state(args.resume_from_checkpoint, model, optimizer, lr_scheduler, accelerator)
    if not args.fsdp_shard_dit or args.fsdp_wrap_policy == "linear":
        broadcast_parameters(params, accelerator)
    print_cuda_memory("after checkpoint load", accelerator)
    start_cuda_memory_history(accelerator)

    progress = tqdm(
        total=args.max_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="training",
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
    )
    accumulation_step = 0
    while global_step < args.max_steps:
        for sample in dataloader:
            profile_memory = global_step == 0 and accelerator.is_local_main_process
            if accumulation_step == 0:
                optimizer.zero_grad(set_to_none=True)
            if profile_memory:
                torch.cuda.reset_peak_memory_stats(accelerator.device)
                print_cuda_memory("step0 before forward", accelerator)
            try:
                loss = model(sample)
            except torch.OutOfMemoryError:
                dump_cuda_oom_debug(output_dir, accelerator, "oom_forward")
                raise
            loss_for_log = loss.detach()
            if profile_memory:
                print_cuda_memory("step0 after forward", accelerator)
                torch.cuda.empty_cache()
                print_cuda_memory("step0 after forward empty_cache", accelerator)
            accelerator.backward(loss / grad_accum)
            if profile_memory:
                print_cuda_memory("step0 after backward", accelerator)
            accumulation_step += 1
            if accumulation_step >= grad_accum:
                if (
                    not args.fsdp_shard_dit
                    or (args.fsdp_wrap_policy == "linear" and not args.fsdp_cpu_offload)
                ):
                    sync_gradients(params, accelerator)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                accumulation_step = 0
                if profile_memory:
                    print_cuda_memory("step0 after optimizer step", accelerator)

                global_step += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{accelerator.gather(loss_for_log).mean().item():.5f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )
                progress.refresh()
                if global_step % args.save_every == 0 or global_step == args.max_steps:
                    save_training_state(
                        accelerator,
                        model,
                        optimizer,
                        lr_scheduler,
                        output_dir,
                        global_step,
                        args,
                    )
                if global_step >= args.max_steps:
                    break
    progress.close()
    accelerator.wait_for_everyone()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stand-In-style WanVideo condition LoRA training.")
    parser.add_argument("--dataset_base_path", default=".", help="Base directory for relative metadata paths.")
    parser.add_argument("--dataset_metadata_path", default=None, help="CSV, JSON, or JSONL with video,prompt[,identity_image,expression_video].")
    parser.add_argument("--manifest", default=None, help="Alias for --dataset_metadata_path.")
    parser.add_argument("--wan_model_path", required=True, help="Directory containing Wan2.1 shards, T5, VAE and tokenizer.")
    parser.add_argument("--output_dir", default="train_logs")
    parser.add_argument("--output_suffix", default=".safetensors", choices=(".safetensors", ".pt"))
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--mixed_precision", default="bf16", choices=("no", "fp16", "bf16"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max_pixels", type=int, default=1280 * 720)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--expression_frames", type=int, default=65)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per process/GPU.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=128.0, help="Deprecated; condition LoRA follows Stand-In and does not use alpha scaling.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Deprecated; condition LoRA follows Stand-In and does not use dropout.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--target_effective_batch_size", type=int, default=48)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=0, help="0 means auto: ceil(target_effective_batch_size / (world_size * batch_size)).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0)
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tile_size", type=int, default=128)
    parser.add_argument("--tile_stride", type=int, default=64)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--use_vace", action="store_true")
    parser.add_argument("--detect_expression_boxes", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing_offload", action="store_true")
    parser.add_argument("--fsdp_shard_dit", action="store_true", help="Experimentally shard Wan DiT blocks with PyTorch FSDP.")
    parser.add_argument("--fsdp_wrap_policy", default="linear", choices=("linear", "block"), help="FSDP wrapping granularity for the Wan DiT.")
    parser.add_argument("--fsdp_cpu_offload", action="store_true", help="Use FSDP CPU parameter offload. Usually slower; prefer leaving this off.")
    parser.add_argument("--enable_vram_management", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_persistent_param_in_dit", type=int, default=None)
    parser.add_argument("--vram_limit", type=float, default=None)
    parser.add_argument("--vram_buffer", type=float, default=8.0)
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint-*/training_state.pt.")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
