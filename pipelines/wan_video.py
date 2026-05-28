import torch, types
import time
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import imageio
import os
from typing import List, Tuple
import PIL
from utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from models import DualConditionBuilder, ModelManager, load_state_dict
from models.rope_utils import condition_freqs_from_geometry
from models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from models.attention import RMSNorm as AttentionRMSNorm
from models.wan_video_text_encoder import (
    WanTextEncoder,
    T5RelativeEmbedding,
    T5LayerNorm,
)
from models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from models.wan_video_image_encoder import WanImageEncoder
from models.wan_video_motion_controller import WanMotionControllerModel
from schedulers.flow_match import FlowMatchScheduler
from prompters import WanPrompter
from face_detection import InsightFaceBoxDetector
from vram_management import (
    enable_vram_management,
    AutoWrappedModule,
    AutoWrappedLinear,
    WanAutoCastLayerNorm,
)
from lora import GeneralLoRALoader


_WAN_DEBUG_START = time.perf_counter()


def wan_debug(message):
    elapsed = time.perf_counter() - _WAN_DEBUG_START
    print(f"[wan-debug +{elapsed:8.1f}s] {message}", flush=True)


def load_video_as_list(video_path: str) -> Tuple[List[Image.Image], int, int, int]:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    reader = imageio.get_reader(video_path)

    meta_data = reader.get_meta_data()
    original_width = meta_data['size'][0]
    original_height = meta_data['size'][1]
    
    new_width = (original_width // 16) * 16
    new_height = (original_height // 16) * 16
    
    left = (original_width - new_width) // 2
    top = (original_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    crop_box = (left, top, right, bottom)

    original_frame_count = reader.count_frames()
    new_frame_count = original_frame_count - ((original_frame_count - 1) % 4)

    frames = []
    for i in range(new_frame_count):
        try:
            frame_data = reader.get_data(i)
            pil_image = Image.fromarray(frame_data)
            cropped_image = pil_image.crop(crop_box)
            frames.append(cropped_image)
        except IndexError:
            print(f"Warning: Actual number of frames is less than expected. Stopping at frame {i}.")
            new_frame_count = len(frames)
            break

    reader.close()

    return frames, new_width, new_height, new_frame_count

class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.condition_builder: DualConditionBuilder = None
        self.condition_builder_2: DualConditionBuilder = None
        self.face_box_detector = None
        self.face_box_detection_failed = False
        self.in_iteration_models = ("dit", "motion_controller")
        self.in_iteration_models_2 = ("dit2", "motion_controller")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video

    def get_condition_builder(self, dit):
        if dit is None:
            return None
        attr_name = "condition_builder_2" if dit is self.dit2 else "condition_builder"
        builder = getattr(self, attr_name)
        if builder is None:
            builder = DualConditionBuilder(dit=dit).to(
                device=self.device,
                dtype=self.torch_dtype,
            )
            setattr(self, attr_name, builder)
        return builder

    def encode_identity_image(self, image, height=None, width=None):
        self.load_models_to_device(["vae"])
        if height is not None and width is not None:
            image = image.resize((width, height))
        image = (
            torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        )  # [3, H, W]
        image = (
            image.unsqueeze(1).unsqueeze(0).to(dtype=self.torch_dtype)
        )  # [B, 3, 1, H, W]
        image = image * 2 - 1
        image_latents = self.vae.encode(image, device=self.device, tiled=False)
        return image_latents

    def encode_condition_frames(self, frames, height, width):
        self.load_models_to_device(["vae"])
        if isinstance(frames, Image.Image):
            frames = [frames]
        frames = [frame.resize((width, height)) for frame in frames]
        video = self.preprocess_video(frames)
        return self.vae.encode(video, device=self.device, tiled=False)

    def detect_expression_face_boxes(self, frames):
        if isinstance(frames, Image.Image):
            frames = [frames]
        if self.face_box_detection_failed:
            wan_debug("Skipping expression face detection because it failed earlier")
            return None
        if self.face_box_detector is None:
            try:
                wan_debug("Initializing InsightFace detector for expression boxes")
                self.face_box_detector = InsightFaceBoxDetector()
                wan_debug("InsightFace detector ready")
            except ImportError as exc:
                print(f"Warning: automatic face detection disabled: {exc}")
                self.face_box_detection_failed = True
                return None
        try:
            wan_debug(f"Detecting expression face boxes on {len(frames)} frames")
            boxes = self.face_box_detector.detect_frames(frames)
            wan_debug(f"Detected expression face boxes with shape={tuple(boxes.shape)}")
            return boxes
        except ValueError as exc:
            print(f"Warning: automatic face detection failed: {exc}")
            return None

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def training_loss(self, **inputs):
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps
        )
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(
            dtype=self.torch_dtype, device=self.device
        )

        inputs["latents"] = self.scheduler.add_noise(
            inputs["input_latents"], inputs["noise"], timestep
        )
        training_target = self.scheduler.training_target(
            inputs["input_latents"], inputs["noise"], timestep
        )

        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    AttentionRMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    AttentionRMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from distributed.xdit_context_parallel import (
            usp_attn_forward,
            usp_dit_forward,
        )

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"
        ),
        redirect_common_files: bool = True,
        use_usp=False,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern]
                ):
                    print(
                        f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection."
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ]

        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            pipe.initialize_usp()

        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
            )

        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model(
            "wan_video_motion_controller"
        )

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[
            Literal[
                "Left",
                "Right",
                "Up",
                "Down",
                "LeftUp",
                "LeftDown",
                "RightUp",
                "RightDown",
            ]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        # Experimental identity/expression condition branch
        identity_reference_image=None,
        expression_reference_frames=None,
        expression_face_boxes=None,
    ):
        identity_latents = None
        expression_latents = None
        if identity_reference_image is not None:
            wan_debug("Encoding identity reference image")
            identity_latents = self.encode_identity_image(
                identity_reference_image,
                height=height,
                width=width,
            )
            wan_debug(f"Encoded identity latents shape={tuple(identity_latents.shape)}")
        if expression_reference_frames is not None:
            if expression_face_boxes is None:
                wan_debug("No expression face boxes supplied; starting automatic detection")
                expression_face_boxes = self.detect_expression_face_boxes(
                    expression_reference_frames
                )
            else:
                wan_debug("Using supplied expression face boxes")
            wan_debug(f"Encoding expression reference frames count={len(expression_reference_frames)}")
            expression_latents = self.encode_condition_frames(
                expression_reference_frames,
                height=height,
                width=width,
            )
            wan_debug(f"Encoded expression latents shape={tuple(expression_latents.shape)}")
        if expression_face_boxes is not None and not isinstance(
            expression_face_boxes, torch.Tensor
        ):
            expression_face_boxes = torch.tensor(expression_face_boxes)
            wan_debug(f"Converted expression face boxes to tensor shape={tuple(expression_face_boxes.shape)}")
        # Scheduler
        wan_debug(f"Setting scheduler timesteps num_inference_steps={num_inference_steps}")
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "condition_builder": None,
            "identity_latents": identity_latents,
            "expression_latents": expression_latents,
            "expression_face_boxes": expression_face_boxes,
        }
        for unit in self.units:
            wan_debug(f"Running pipeline unit {unit.__class__.__name__}")
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )
            wan_debug(f"Finished pipeline unit {unit.__class__.__name__}")
        # Denoise
        wan_debug(f"Loading iteration models to device: {self.in_iteration_models}")
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        wan_debug(f"Starting denoise loop with {len(self.scheduler.timesteps)} timesteps")
        for progress_id, timestep in enumerate(
            progress_bar_cmd(self.scheduler.timesteps)
        ):
            # Switch DiT if necessary
            if (
                timestep.item()
                < switch_DiT_boundary * self.scheduler.num_train_timesteps
                and self.dit2 is not None
                and not models["dit"] is self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2

            # Timestep
            timestep = timestep.unsqueeze(0).to(
                dtype=self.torch_dtype, device=self.device
            )

            # Inference
            if identity_latents is not None or expression_latents is not None:
                inputs_shared["condition_builder"] = self.get_condition_builder(
                    models["dit"]
                )
            noise_pred_posi = self.model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep
            )
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    condition_inputs = {
                        "condition_builder": inputs_shared.get("condition_builder"),
                        "identity_latents": inputs_shared.get("identity_latents"),
                        "expression_latents": inputs_shared.get("expression_latents"),
                        "expression_face_boxes": inputs_shared.get("expression_face_boxes"),
                    }
                    inputs_shared["condition_builder"] = None
                    inputs_shared["identity_latents"] = None
                    inputs_shared["expression_latents"] = None
                    inputs_shared["expression_face_boxes"] = None
                    try:
                        noise_pred_nega = self.model_fn(
                            **models, **inputs_shared, **inputs_nega, timestep=timestep
                        )
                    finally:
                        inputs_shared.update(condition_inputs)
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared[
                    "first_frame_latents"
                ]

        # Decode
        wan_debug("Denoise loop done; loading VAE for decode")
        self.load_models_to_device(["vae"])
        wan_debug("Decoding latents to video")
        video = self.vae.decode(
            inputs_shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video = self.vae_output_to_video(video)
        wan_debug(f"Decoded video frames={len(video)}")
        self.load_models_to_device([])

        return video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(
            height, width, num_frames
        )
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "seed",
                "rand_device",
            )
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        seed,
        rand_device,
    ):
        length = (num_frames - 1) // 4 + 1
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(
                input_latents, noise, timestep=pipe.scheduler.timesteps[0]
            )
            return {"latents": latents}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(
            prompt, positive=positive, device=pipe.device
        )
        return {"context": prompt_emb}


class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("image_encoder", "vae"),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, pipe.image_encoder.encode_image([end_image])], dim=1
                )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context, "y": y}


class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",),
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if (
            input_image is None
            or pipe.image_encoder is None
            or not pipe.dit.require_clip_embedding
        ):
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat(
                    [clip_context, pipe.image_encoder.encode_image([end_image])], dim=1
                )
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "end_image",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        end_image,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(
            pipe.device
        )
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 2, height, width).to(image.device),
                    end_image.transpose(0, 1),
                ],
                dim=1,
            )
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )

        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        y = pipe.vae.encode(
            [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"y": y}


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """

    def __init__(self):
        super().__init__(
            input_params=(
                "input_image",
                "latents",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_image,
        latents,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
    ):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(
            0, 1
        )
        z = pipe.vae.encode(
            [image],
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        latents[:, :, 0:1] = z
        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": z,
        }


class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "control_video",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "clip_feature",
                "y",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        control_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        clip_feature,
        y,
    ):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(
            control_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if clip_feature is None or y is None:
            clip_feature = torch.zeros(
                (1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device
            )
            y = torch.zeros(
                (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        else:
            y = y[:, -16:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "camera_control_direction",
                "camera_control_speed",
                "camera_control_origin",
                "latents",
                "input_image",
            ),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        camera_control_direction,
        camera_control_speed,
        camera_control_origin,
        latents,
        input_image,
    ):
        if camera_control_direction is None:
            return {}
        camera_control_plucker_embedding = (
            pipe.dit.control_adapter.process_camera_coordinates(
                camera_control_direction,
                num_frames,
                height,
                width,
                camera_control_speed,
                camera_control_origin,
            )
        )

        control_camera_video = (
            camera_control_plucker_embedding[:num_frames]
            .permute([3, 0, 1, 2])
            .unsqueeze(0)
        )
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(
                    control_camera_video[:, :, 0:1], repeats=4, dim=2
                ),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )
        control_camera_latents_input = control_camera_latents.to(
            device=pipe.device, dtype=pipe.torch_dtype
        )

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        pipe.load_models_to_device(self.onload_model_names)
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
            input_params_nega={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        num_inference_steps,
        tea_cache_l1_thresh,
        tea_cache_model_id,
    ):
        if tea_cache_l1_thresh is None:
            return {}
        return {
            "tea_cache": TeaCache(
                num_inference_steps,
                rel_l1_thresh=tea_cache_l1_thresh,
                model_id=tea_cache_model_id,
            )
        }


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat(
                    (tensor_shared, tensor_shared), dim=0
                )
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + 1) / border_width, dims=(0,)
            )
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device,
        computation_dtype,
        model_kwargs,
        tensor_names,
        batch_size=None,
    ):
        tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if model_kwargs.get(tensor_name) is not None
        ]
        tensor_dict = {
            tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names
        }
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = (
            tensor_dict[tensor_names[0]].device,
            tensor_dict[tensor_names[0]].dtype,
        )
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if (
                t - sliding_window_stride >= 0
                and t - sliding_window_stride + sliding_window_size >= T
            ):
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update(
                {
                    tensor_name: tensor_dict[tensor_name][:, :, t:t_:, :].to(
                        device=computation_device, dtype=computation_dtype
                    )
                    for tensor_name in tensor_names
                }
            )
            model_output = model_fn(**model_kwargs).to(
                device=data_device, dtype=data_dtype
            )
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,),
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents=None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    condition_builder: Optional[DualConditionBuilder] = None,
    identity_latents: Optional[torch.Tensor] = None,
    expression_latents: Optional[torch.Tensor] = None,
    expression_face_boxes: Optional[torch.Tensor] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            condition_builder=condition_builder,
            identity_latents=identity_latents,
            expression_latents=expression_latents,
            expression_face_boxes=expression_face_boxes,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )
    x_ip = None
    t_mod_ip = None
    condition_token_counts = None
    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=latents.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0)
        )
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    offset = 1
    freqs = (
        torch.cat(
            [
                dit.freqs[0][offset : f + offset].view(f, 1, 1, -1).expand(f, h, w, -1),
                dit.freqs[1][offset : h + offset].view(1, h, 1, -1).expand(f, h, w, -1),
                dit.freqs[2][offset : w + offset].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .to(x.device)
    )

    ############################################################################################
    if condition_builder is not None and (
        identity_latents is not None or expression_latents is not None
    ):
        condition = condition_builder(
            identity_latents=identity_latents,
            expression_latents=expression_latents,
            expression_face_boxes=expression_face_boxes,
        )
        if condition is not None:
            condition_tokens = condition.tokens.to(dtype=x.dtype, device=x.device)
            condition_token_counts = (
                condition.identity_token_count,
                condition.expression_token_count,
            )
            condition_time = torch.zeros_like(timestep)
            condition_time_emb = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, condition_time)
            )
            t_mod_ip = dit.time_projection(condition_time_emb).unflatten(1, (6, dit.dim))
            condition_freq = condition_freqs_from_geometry(
                dit=dit,
                main_grid=(f, h, w),
                device=x.device,
                identity_grid=condition.identity_grid,
                expression_grid=condition.expression_grid,
                expression_token_indices=condition.expression_token_indices,
            )
            x_ip = condition_tokens
            freqs = torch.cat([freqs, condition_freq], dim=0)
    if dit.training and not getattr(dit, "_printed_condition_token_debug", False):
        should_print = True
        try:
            import torch.distributed as dist

            should_print = not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            should_print = True
        if should_print:
            identity_tokens = 0
            expression_tokens = 0
            if condition_token_counts is not None:
                identity_tokens, expression_tokens = condition_token_counts
            condition_total = identity_tokens + expression_tokens
            print(
                "[token-debug] "
                f"main_video_tokens={x.shape[1]} "
                f"latent_grid=({f},{h},{w}) "
                f"text_tokens={context.shape[1]} "
                f"identity_tokens={identity_tokens} "
                f"expression_tokens={expression_tokens} "
                f"condition_total={condition_total} "
                f"condition_tokens_total={0 if x_ip is None else x_ip.shape[1]} "
                f"freqs_total={freqs.shape[0]}",
                flush=True,
            )
        dit._printed_condition_token_debug = True
    ############################################################################################
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[
                get_sequence_parallel_rank()
            ]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        train_start = getattr(dit, "condition_lora_train_start", 0)
        for block_idx, block in enumerate(dit.blocks):
            skip_grad = dit.training and train_start > 0 and block_idx < train_start
            if skip_grad:
                with torch.no_grad():
                    x, x_ip = block(
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_ip=x_ip,
                        t_mod_ip=t_mod_ip,
                        condition_token_counts=condition_token_counts,
                    )
                continue
            if dit.training and train_start > 0 and block_idx == train_start:
                x = x.detach()
                if x_ip is not None:
                    x_ip = x_ip.detach()
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x, x_ip = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        x_ip,
                        t_mod_ip,
                        condition_token_counts,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x, x_ip = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    x_ip,
                    t_mod_ip,
                    condition_token_counts,
                    use_reentrant=False,
                )
            else:
                x, x_ip = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    x_ip=x_ip,
                    t_mod_ip=t_mod_ip,
                    condition_token_counts=condition_token_counts,
                )
        if tea_cache is not None:
            tea_cache.store(x)

    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1] :]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x
