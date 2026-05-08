# Expression Identity Generation

Architecture project for controlling visual identity and expressive behavior
separately on top of a Wan/DiT video backbone. The active pipeline is
`pipelines/wan_video.py`, with the experimental condition branch wired directly
into the Wan forward pass.

## Idea

The diffusion backbone can remain frozen while two groups of condition tokens
are added:

- identity: a reference image is encoded with the backbone VAE;
- expression: expressive reference frames are encoded with the VAE, then the
  tokens corresponding to the face region are selected.

The final tokens are concatenated:

```text
condition_tokens = [identity_tokens ; expression_tokens]
```

Expression flow:

```text
expression_video
  -> VAE encoder
  -> expression_latents
  -> DiT patchify
  -> expression_vae_tokens
  -> ExpressionAdapter / FaceTokenSelector
  -> expression_tokens
```

They are then injected into the backbone DiT through restricted self-attention:
video tokens can attend to the condition tokens, while the condition branch
remains stable. Identity and expression groups also use distinct RoPE mappings
through `models/rope_utils.py`, so they do not occupy the exact same conditional
position space.

## Active Pipeline

The main execution path is:

```text
pipelines/wan_video.py
  -> WanVideoPipeline
  -> model_fn_wan_video
  -> DualConditionBuilder
  -> condition_freqs
  -> DiT blocks / restricted self-attention
```

## Project Layout

```text
expression_identity_gen/
  configs/
    emotion_identity_wan21.yaml
  models/
    condition_builder.py
    expression_adapter.py 
    diffusion_forward.py
    rope_utils.py
    time_embedding.py
    wan_video_dit.py
    wan_video_vae.py
  pipelines/
    wan_video.py
  trainers/
    trainer.py
    losses.py
```

