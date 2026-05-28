#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# OOD example: identity and expression come from the same subject W033.
# This tests expression/pose generalization without mixing identities.
IDENTITY_IMAGE="${IDENTITY_IMAGE:-/home/ens.old/Bpokrzywa/datasets/MEAD_neutral_frames/M009/front/neutral_frames.png}"
EXPRESSION_VIDEO="${EXPRESSION_VIDEO:-/home/ens.old/Bpokrzywa/datasets/MEAD_full/MEAD/M027/video/front/fear/level_3/002.mp4}"
LORA_PATH="${LORA_PATH:-train_logs/fsdp_rank128_cpu_offload/standin_condition_lora_step_003000.safetensors}"
OUTPUT="${OUTPUT:-outputs/ood_identity_M009_expression_M027_fear_step3000_2.mp4}"


CONDITION_ATTENTION_OUTPUT="${CONDITION_ATTENTION_OUTPUT:-outputs/ood_identity_M009_expression_M027_fear_step3000_condition_attention.mp4}"
ATTENTION_BLOCK="${ATTENTION_BLOCK:-30}"
ATTENTION_ALPHA="${ATTENTION_ALPHA:-0.75}"
ATTENTION_CHUNK_SIZE="${ATTENTION_CHUNK_SIZE:-512}"
ATTENTION_FACE_MARGIN="${ATTENTION_FACE_MARGIN:-0.25}"
SEED="${SEED:-42}"

python infer.py \
  --wan_model_path checkpoints/Wan2.1/t2v \
  --lora_path "$LORA_PATH" \
  --identity_image "$IDENTITY_IMAGE" \
  --expression_video "$EXPRESSION_VIDEO" \
  --prompt "A realistic close-up talking-head video of a young man speaking directly to the camera, facing the camera head-on. The camera is static and centered. He shows a fearful facial expression." \
  --output "$OUTPUT" \
  --seed "$SEED" \
  --height 384 \
  --width 640 \
  --num_frames 81 \
  --num_inference_steps 30 \
  --cfg_scale 5.0 \
  --condition_attention_output "$CONDITION_ATTENTION_OUTPUT" \
  --condition_attention_block "$ATTENTION_BLOCK" \
  --condition_attention_alpha "$ATTENTION_ALPHA" \
  --condition_attention_chunk_size "$ATTENTION_CHUNK_SIZE" \
  --condition_attention_head_only \
  --condition_attention_face_margin "$ATTENTION_FACE_MARGIN" \
  --expression_frames 65 \
  --dtype bf16 \
  --device cuda \
  --enable_vram_management \
  --tiled
