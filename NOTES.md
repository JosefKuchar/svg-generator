## OmniSVG
### Install
```sh
git clone https://github.com/OmniSVG/OmniSVG.git
cd OmniSVG
uv init --python=3.10
uv sync
uv pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
echo "accelerate" >> requirements.txt
uv pip install -r requirements.txt
uv run huggingface-cli download OmniSVG/OmniSVG1.1_8B --local-dir ./pretrained_models/OmniSVG1.1_8B
uv run huggingface-cli download OmniSVG/OmniSVG1.1_4B --local-dir ./pretrained_models/OmniSVG1.1_4B
uv run huggingface-cli download OmniSVG/OmniSVG --local-dir ./pretrained_models/OmniSVG-3B
```

### Run
```sh
# 8b model
uv run inference.py --task image-to-svg --input ./input --output ./output_image_8b --save-all-candidates
# 4b model
uv run inference.py --task image-to-svg --input ./input --output ./output_image_4b --save-all-candidates --model-size 4B
```

## Star Vector
```sh
git clone git@github.com:JosefKuchar/star-vector.git
cd star-vector
uv sync
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl
CUDA_VISIBLE_DEVICES=10 uv run batch.py reference 8b
CUDA_VISIBLE_DEVICES=9 uv run batch.py reference 1b --model-name starvector/starvector-1b-im2svg
# TODO
```

## AI toolkit
### Install
```sh
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
uv init --python=3.12
uv sync
uv pip uv pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

### Run
```sh
cd ui
npm run build_and_start
```
Forwarding
```sh
ssh -L 8675:localhost:8675 akeso
```

## LORA
### Dataset
Generate dataset
```sh
uv run python scripts/export_svg_svgrepo_dataset.py --caption-index 1 --caption-prefix "SVG illustration with white background. " --num-samples 8000
```

Config
```
---
job: "extension"
config:
  name: "svg-lora-v2"
  process:
    - type: "diffusion_trainer"
      training_folder: "/var/tmp/xkuchar/ai-toolkit/output"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda"
      trigger_word: null
      performance_log_every: 10
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        conv: 16
        conv_alpha: 16
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "bf16"
        save_every: 500
        max_step_saves_to_keep: 10
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "/var/tmp/xkuchar/ai-toolkit/datasets/svg-svgrepo"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution:
            - 1024
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          flip_x: false
          flip_y: false
          num_repeats: 1
      train:
        batch_size: 2
        bypass_guidance_embedding: false
        steps: 5000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw"
        timestep_type: "weighted"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.0001
        unload_text_encoder: false
        cache_text_embeddings: false
        lr: 0.0001
        ema_config:
          use_ema: false
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
      logging:
        log_every: 1
        use_ui_logger: true
      model:
        name_or_path: "Tongyi-MAI/Z-Image"
        quantize: false
        qtype: "qfloat8"
        quantize_te: false
        qtype_te: "qfloat8"
        arch: "zimage"
        low_vram: false
        model_kwargs: {}
        layer_offloading: false
        layer_offloading_text_encoder_percent: 1
        layer_offloading_transformer_percent: 1
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        samples:
          - prompt: "SVG illustration with white background. Create a minimalist stethoscope design with sleek, modern lines. The earpieces are dark gray, while the tubing is a contrasting light gray. The chest piece is circular with a black center, emphasizing simplicity and elegance. The stethoscope should appear floating, highlighting its clean, professional look, ideal for medical professionals. The image should convey precision and trustworthiness in healthcare."
          - prompt: "SVG illustration with white background. Create a minimalist icon of a dark gray disc with a central hole and two small rectangular highlights. Adjacent to the disc, include a bold, dark gray downward arrow, suggesting a download action. The disc and arrow should be cleanly separated, emphasizing the concept of downloading a disc. Ensure the design is simple, modern, and easily recognizable."
          - prompt: "SVG illustration with white background. Create an image featuring a classic puzzle game scene with geometric blocks in mid-fall. A square block hovers above a larger, intricate structure composed of interlocking shapes, forming an incomplete line. The design is minimalist, with bold black outlines and empty spaces, capturing the essence of strategic gameplay and the anticipation of the next move. The composition highlights the challenge and elegance of block arrangement."
          - prompt: "SVG illustration with white background. Create a simple, black-and-white illustration of a kayaker in action. The figure should be depicted in a streamlined kayak, paddling vigorously with a double-bladed oar. Waves should be shown beneath the kayak to suggest movement through water. The kayaker's posture should convey effort and focus, emphasizing the dynamic nature of the sport. The overall design should be clean and minimalistic, highlighting the essence of kayaking."
          - prompt: "SVG illustration with white background. Create a minimalist black footprint silhouette. The heel is rounded, curving smoothly into the base. Four distinct circles represent the toes, with the big toe slightly separated. The arch is clearly defined, showing a natural foot shape. The design is simple and clean, emphasizing the outline without any additional details or textures."
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 50
        num_frames: 1
        fps: 1
meta:
  name: "[name]"
  version: "1.0"
```

```
---
job: "extension"
config:
  name: "svg-lora-v3"
  process:
    - type: "diffusion_trainer"
      training_folder: "/var/tmp/xkuchar/ai-toolkit/output"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda"
      trigger_word: null
      performance_log_every: 10
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        conv: 16
        conv_alpha: 16
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "bf16"
        save_every: 500
        max_step_saves_to_keep: 10
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "/var/tmp/xkuchar/ai-toolkit/datasets/svg-svgrepo-prefixed"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution:
            - 1024
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          flip_x: false
          flip_y: false
          num_repeats: 1
      train:
        batch_size: 2
        bypass_guidance_embedding: false
        steps: 5000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "prodigyopt"
        timestep_type: "weighted"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.01
          d_coef: 1
        unload_text_encoder: false
        cache_text_embeddings: false
        lr: 1
        ema_config:
          use_ema: false
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
      logging:
        log_every: 1
        use_ui_logger: true
      model:
        name_or_path: "Tongyi-MAI/Z-Image"
        quantize: false
        qtype: "qfloat8"
        quantize_te: false
        qtype_te: "qfloat8"
        arch: "zimage"
        low_vram: false
        model_kwargs: {}
        layer_offloading: false
        layer_offloading_text_encoder_percent: 1
        layer_offloading_transformer_percent: 1
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        samples:
          - prompt: "SVG illustration with white background. Create a minimalist stethoscope design with sleek, modern lines. The earpieces are dark gray, while the tubing is a contrasting light gray. The chest piece is circular with a black center, emphasizing simplicity and elegance. The stethoscope should appear floating, highlighting its clean, professional look, ideal for medical professionals. The image should convey precision and trustworthiness in healthcare."
          - prompt: "SVG illustration with white background. Create a minimalist icon of a dark gray disc with a central hole and two small rectangular highlights. Adjacent to the disc, include a bold, dark gray downward arrow, suggesting a download action. The disc and arrow should be cleanly separated, emphasizing the concept of downloading a disc. Ensure the design is simple, modern, and easily recognizable."
          - prompt: "SVG illustration with white background. Create an image featuring a classic puzzle game scene with geometric blocks in mid-fall. A square block hovers above a larger, intricate structure composed of interlocking shapes, forming an incomplete line. The design is minimalist, with bold black outlines and empty spaces, capturing the essence of strategic gameplay and the anticipation of the next move. The composition highlights the challenge and elegance of block arrangement."
          - prompt: "SVG illustration with white background. Create a simple, black-and-white illustration of a kayaker in action. The figure should be depicted in a streamlined kayak, paddling vigorously with a double-bladed oar. Waves should be shown beneath the kayak to suggest movement through water. The kayaker's posture should convey effort and focus, emphasizing the dynamic nature of the sport. The overall design should be clean and minimalistic, highlighting the essence of kayaking."
          - prompt: "SVG illustration with white background. Create a minimalist black footprint silhouette. The heel is rounded, curving smoothly into the base. Four distinct circles represent the toes, with the big toe slightly separated. The arch is clearly defined, showing a natural foot shape. The design is simple and clean, emphasizing the outline without any additional details or textures."
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 50
        num_frames: 1
        fps: 1
meta:
  name: "[name]"
  version: "1.0"
```


best
```
---
job: "extension"
config:
  name: "svg-lora-v4-16"
  process:
    - type: "diffusion_trainer"
      training_folder: "/var/tmp/xkuchar/ai-toolkit/output"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda"
      trigger_word: null
      performance_log_every: 10
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        conv: 16
        conv_alpha: 16
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "bf16"
        save_every: 100
        max_step_saves_to_keep: 10
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "/var/tmp/xkuchar/ai-toolkit/datasets/svg-svgrepo-prefixed"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution:
            - 1024
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          flip_x: false
          flip_y: false
          num_repeats: 1
      train:
        batch_size: 2
        bypass_guidance_embedding: false
        steps: 1200
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "prodigyopt"
        timestep_type: "weighted"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.01
          d_coef: 1
        unload_text_encoder: false
        cache_text_embeddings: false
        lr: 1
        ema_config:
          use_ema: false
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
      logging:
        log_every: 1
        use_ui_logger: true
      model:
        name_or_path: "Tongyi-MAI/Z-Image"
        quantize: false
        qtype: "qfloat8"
        quantize_te: false
        qtype_te: "qfloat8"
        arch: "zimage"
        low_vram: false
        model_kwargs: {}
        layer_offloading: false
        layer_offloading_text_encoder_percent: 1
        layer_offloading_transformer_percent: 1
      sample:
        sampler: "flowmatch"
        sample_every: 100
        width: 1024
        height: 1024
        samples:
          - prompt: "SVG illustration with white background. Create a minimalist stethoscope design with sleek, modern lines. The earpieces are dark gray, while the tubing is a contrasting light gray. The chest piece is circular with a black center, emphasizing simplicity and elegance. The stethoscope should appear floating, highlighting its clean, professional look, ideal for medical professionals. The image should convey precision and trustworthiness in healthcare."
          - prompt: "SVG illustration with white background. Create a minimalist icon of a dark gray disc with a central hole and two small rectangular highlights. Adjacent to the disc, include a bold, dark gray downward arrow, suggesting a download action. The disc and arrow should be cleanly separated, emphasizing the concept of downloading a disc. Ensure the design is simple, modern, and easily recognizable."
          - prompt: "SVG illustration with white background. Create an image featuring a classic puzzle game scene with geometric blocks in mid-fall. A square block hovers above a larger, intricate structure composed of interlocking shapes, forming an incomplete line. The design is minimalist, with bold black outlines and empty spaces, capturing the essence of strategic gameplay and the anticipation of the next move. The composition highlights the challenge and elegance of block arrangement."
          - prompt: "SVG illustration with white background. Create a simple, black-and-white illustration of a kayaker in action. The figure should be depicted in a streamlined kayak, paddling vigorously with a double-bladed oar. Waves should be shown beneath the kayak to suggest movement through water. The kayaker's posture should convey effort and focus, emphasizing the dynamic nature of the sport. The overall design should be clean and minimalistic, highlighting the essence of kayaking."
          - prompt: "SVG illustration with white background. Create a minimalist black footprint silhouette. The heel is rounded, curving smoothly into the base. Four distinct circles represent the toes, with the big toe slightly separated. The arch is clearly defined, showing a natural foot shape. The design is simple and clean, emphasizing the outline without any additional details or textures."
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 50
        num_frames: 1
        fps: 1
meta:
  name: "[name]"
  version: "1.0"
```


```
---
job: "extension"
config:
  name: "svg-lora-v6-64"
  process:
    - type: "diffusion_trainer"
      training_folder: "/var/tmp/xkuchar/ai-toolkit/output"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda"
      trigger_word: null
      performance_log_every: 10
      network:
        type: "lora"
        linear: 64
        linear_alpha: 64
        conv: 16
        conv_alpha: 16
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "bf16"
        save_every: 200
        max_step_saves_to_keep: 10
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "/var/tmp/xkuchar/ai-toolkit/datasets/svg-svgrepo-8000"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution:
            - 1024
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          flip_x: false
          flip_y: false
          num_repeats: 1
      train:
        batch_size: 2
        bypass_guidance_embedding: false
        steps: 4000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "prodigyopt"
        timestep_type: "weighted"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.01
          d_coef: 1
        unload_text_encoder: false
        cache_text_embeddings: false
        lr: 1
        ema_config:
          use_ema: false
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
      logging:
        log_every: 1
        use_ui_logger: true
      model:
        name_or_path: "Tongyi-MAI/Z-Image"
        quantize: false
        qtype: "qfloat8"
        quantize_te: false
        qtype_te: "qfloat8"
        arch: "zimage"
        low_vram: false
        model_kwargs: {}
        layer_offloading: false
        layer_offloading_text_encoder_percent: 1
        layer_offloading_transformer_percent: 1
      sample:
        sampler: "flowmatch"
        sample_every: 200
        width: 1024
        height: 1024
        samples:
          - prompt: "SVG illustration with white background. Create a simple, minimalist stethoscope design with black tubing, gray earpieces, and a chest piece, embodying medical professionalism and clarity."
          - prompt: "SVG illustration with white background. Create a sleek, dark gray disc with a white central hole and two small rectangles, accompanied by a bold black downward arrow, symbolizing download."
          - prompt: "SVG illustration with white background. Create a minimalist design featuring black Tetris blocks stacked unevenly, with a small square piece precariously balanced above, hinting at imminent collapse."
          - prompt: "SVG illustration with white background. A black silhouette of a person paddling a kayak, with simple waves and a paddle mid-motion, capturing the essence of water sports and recreation."
          - prompt: "SVG illustration with white background. Create a bold, black silhouette of a single human footprint with four toe circles, emphasizing simplicity and clarity."
          - prompt: "SVG illustration with white background. Create a minimalist chandelier design with two dark blue glass shades hanging from a central bar, emphasizing clean lines and modern simplicity."
          - prompt: "SVG illustration with white background. Create a minimalist blue memory card icon with a light blue label and dark blue notches, emphasizing clean lines and simplicity."
          - prompt: "SVG illustration with white background. Create a bold, red lipstick kiss mark with intricate texture, featuring overlapping lips in a circular pattern, showcasing a vibrant gradient effect."
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 50
        num_frames: 1
        fps: 1
meta:
  name: "[name]"
  version: "1.0"
```

```
---
job: "extension"
config:
  name: "svg-lora-ablation-4"
  process:
    - type: "diffusion_trainer"
      training_folder: "/var/tmp/xkuchar/ai-toolkit/output"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda"
      trigger_word: null
      performance_log_every: 10
      network:
        type: "lora"
        linear: 4
        linear_alpha: 4
        conv: 16
        conv_alpha: 16
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "bf16"
        save_every: 500
        max_step_saves_to_keep: 10
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "/var/tmp/xkuchar/ai-toolkit/datasets/svg-svgrepo-8000"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution:
            - 1024
          controls: []
          shrink_video_to_frames: true
          num_frames: 1
          flip_x: false
          flip_y: false
          num_repeats: 1
      train:
        batch_size: 2
        bypass_guidance_embedding: false
        steps: 4000
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw"
        timestep_type: "weighted"
        content_or_style: "balanced"
        optimizer_params:
          weight_decay: 0.0001
        unload_text_encoder: false
        cache_text_embeddings: false
        lr: 0.0001
        ema_config:
          use_ema: false
          ema_decay: 0.99
        skip_first_sample: false
        force_first_sample: false
        disable_sampling: false
        dtype: "bf16"
        diff_output_preservation: false
        diff_output_preservation_multiplier: 1
        diff_output_preservation_class: "person"
        switch_boundary_every: 1
        loss_type: "mse"
      logging:
        log_every: 1
        use_ui_logger: true
      model:
        name_or_path: "Tongyi-MAI/Z-Image"
        quantize: false
        qtype: "qfloat8"
        quantize_te: false
        qtype_te: "qfloat8"
        arch: "zimage"
        low_vram: false
        model_kwargs: {}
        layer_offloading: false
        layer_offloading_text_encoder_percent: 1
        layer_offloading_transformer_percent: 1
      sample:
        sampler: "flowmatch"
        sample_every: 500
        width: 1024
        height: 1024
        samples:
          - prompt: "SVG illustration with white background. Create a simple, minimalist stethoscope design with black tubing, gray earpieces, and a chest piece, embodying medical professionalism and clarity."
          - prompt: "SVG illustration with white background. Create a sleek, dark gray disc with a white central hole and two small rectangles, accompanied by a bold black downward arrow, symbolizing download."
          - prompt: "SVG illustration with white background. Create a minimalist design featuring black Tetris blocks stacked unevenly, with a small square piece precariously balanced above, hinting at imminent collapse."
          - prompt: "SVG illustration with white background. A black silhouette of a person paddling a kayak, with simple waves and a paddle mid-motion, capturing the essence of water sports and recreation."
          - prompt: "SVG illustration with white background. Create a bold, black silhouette of a single human footprint with four toe circles, emphasizing simplicity and clarity."
          - prompt: "SVG illustration with white background. Create a minimalist chandelier design with two dark blue glass shades hanging from a central bar, emphasizing clean lines and modern simplicity."
          - prompt: "SVG illustration with white background. Create a minimalist blue memory card icon with a light blue label and dark blue notches, emphasizing clean lines and simplicity."
          - prompt: "SVG illustration with white background. Create a bold, red lipstick kiss mark with intricate texture, featuring overlapping lips in a circular pattern, showcasing a vibrant gradient effect."
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 50
        num_frames: 1
        fps: 1
meta:
  name: "[name]"
  version: "1.0"
```

### EVal
```sh
uv run z-image-dataset.py --batch-size 8 --caption-index 1 --prompt-prefix "SVG illustration with white background. " --output-dir ./z-image-renders/caption-1-prefixed
# Base
uv run z-image-dataset.py --batch-size 8 --caption-index 1 --output-dir ./z-image-renders/base --model-id "Tongyi-MAI/Z-Image" --num-inference-steps 50 --guidance-scale 4.0
# Base prefixed
uv run z-image-dataset.py --batch-size 8 --caption-index 1 --output-dir ./z-image-renders/base-prefixed --model-id "Tongyi-MAI/Z-Image" --num-inference-steps 50 --guidance-scale 4.0 --prompt-prefix "SVG illustration with white background. "
```
```sh
uv run python render_svg_svgrepo_valid_raster.py  --output-dir ./raster/reference
```

```sh
# Metrics
# Base
uv run python scripts/benchmark_image_folders.py ./raster/reference/ z-image-renders/base
clip_similarity: 0.818210
dino_similarity: 0.509159
vectorization_mse: 266.565137

# Base prefixed
uv run python scripts/benchmark_image_folders.py ./raster/reference/ z-image-renders/base-prefixed
clip_similarity: 0.819865
dino_similarity: 0.545802
vectorization_mse: 230.160058

# Turbo
uv run python scripts/benchmark_image_folders.py ./raster/reference/ z-image-renders/turbo
clip_similarity: 0.826786
dino_similarity: 0.509892
vectorization_mse: 227.691742

# Turbo prefixed
uv run python scripts/benchmark_image_folders.py ./raster/reference/ z-image-renders/turbo-prefixed
clip_similarity: 0.871237
dino_similarity: 0.583856
vectorization_mse: 142.711678

# Turbo prefixed with LORA
clip_similarity: 0.879104
dino_similarity: 0.600208
vectorization_mse: 143.174617

```

TODOS:


```
xkuchar@akeso:/var/tmp/xkuchar/projects/svg-generator$ uv run python scripts/evaluate_vectorization.py outputs/svg-svgrepo-valid-svgs/ model_outputs/0485_reference/
Evaluating pairs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [12:58<00:00,  1.30it/s]
pairs: 1010
mse: 7444.155031
mae: 32.557324
psnr: 10.951674
ssim: 0.644719
mask_iou: 0.637241
boundary_f1_1px: 0.256705
boundary_f1_2px: 0.323872
boundary_f1_4px: 0.406090
chamfer_px: 14.447945
hausdorff_px: 96.423200
gen_svg_bytes: 11443.058416
gen_elements: 5.301980
gen_paths: 4.301980
gen_path_commands: 112.230693
render_time_ms: 51.473508
gen_render_errors: 0

xkuchar@akeso:/var/tmp/xkuchar/projects/svg-generator$ uv run python scripts/evaluate_vectorization.py outputs/svg-svgrepo-valid-svgs/ model_outputs/0130_reference/
Evaluating pairs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [12:59<00:00,  1.30it/s]
pairs: 1010
mse: 7656.750692
mae: 33.399880
psnr: 10.841842
ssim: 0.638186
mask_iou: 0.632600
boundary_f1_1px: 0.257388
boundary_f1_2px: 0.324550
boundary_f1_4px: 0.406181
chamfer_px: 15.071722
hausdorff_px: 101.201834
gen_svg_bytes: 10453.233663
gen_elements: 5.267327
gen_paths: 4.267327
gen_path_commands: 103.140594
render_time_ms: 50.874528
gen_render_errors: 0

xkuchar@akeso:/var/tmp/xkuchar/projects/svg-generator$ uv run python scripts/evaluate_vectorization.py outputs/svg-svgrepo-valid-svgs/ model_outputs/0627_reference/
Evaluating pairs:   1%|██▎                                                                                                                                                     | 15/1010 [00:11<12:57,  1.28it/s]Evaluating pairs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [12:53<00:00,  1.31it/s]
pairs: 1010
mse: 7107.516058
mae: 31.468682
psnr: 11.165508
ssim: 0.653349
mask_iou: 0.643546
boundary_f1_1px: 0.255342
boundary_f1_2px: 0.323967
boundary_f1_4px: 0.409332
chamfer_px: 16.036777
hausdorff_px: 103.240024
gen_svg_bytes: 10329.023762
gen_elements: 5.266337
gen_paths: 4.266337
gen_path_commands: 102.545545
render_time_ms: 53.227388
gen_render_errors: 0

```


# Proposed
```
images: 1010
valid: 1010
invalid: 0
valid_rate: 1.000000
mse: 6004.362248
mae: 27.312405
psnr: 12.147383
ssim: 0.681470
mask_iou: 0.650743
boundary_f1_1px: 0.312101
boundary_f1_2px: 0.371474
boundary_f1_4px: 0.443863
chamfer_px: 15.619802
hausdorff_px: 102.792629
svg_bytes: 8005.755446
svg_elements: 4.672277
svg_paths: 3.672277
svg_path_commands: 79.528713
render_time_ms: 27.702087
```

VTRACER: 2 minuty
OUR: 6 minut
Starvector 8b: 21:14:37
Starbector 1b: 8:57:37


```
Add and update numbers for the quantitative end2end, you will figure out the variants from the commands xkuchar@akeso:/var/tmp/xkuchar/projects/svg-generator$ uv run python scripts/evaluate_raster_vectorization.py z-
  image-renders/base_prefixed_lora/ star-vector-raster/base-1b/
  Evaluating raster/vector pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [06:17<00:00,
  2.67it/s]
  images: 1010
  valid: 495
  invalid: 515
  valid_rate: 0.490099
  mse: 5057.472308
  mae: 22.354983
  psnr: 13.249188
  ssim: 0.728711
  mask_iou: 0.690275
  boundary_f1_1px: 0.335543
  boundary_f1_2px: 0.416901
  boundary_f1_4px: 0.529655
  chamfer_px: 13.091332
  hausdorff_px: 88.630620
  svg_bytes: 2102.436364
  svg_elements: 11.492929
  svg_paths: 2.872727
  svg_path_commands: 40.135354
  render_time_ms: 30.876259
  missing_svg: 0
  render_errors: 515 xkuchar@akeso:/var/tmp/xkuchar/projects/svg-generator$ uv run python scripts/evaluate_raster_vectorization.py z-image-renders/base_prefixed_lora/ star-vector-raster/base-8b/
  Evaluating raster/vector pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [06:22<00:00,
  2.64it/s]
  images: 1010
  valid: 510
  invalid: 500
  valid_rate: 0.504950
  mse: 6592.780984
  mae: 27.765789
  psnr: 12.078963
  ssim: 0.621840
  mask_iou: 0.596102
  boundary_f1_1px: 0.361363
  boundary_f1_2px: 0.426550
  boundary_f1_4px: 0.505963
  chamfer_px: 18.502480
  hausdorff_px: 120.811723
  svg_bytes: 1270.678431
  svg_elements: 7.100000
  svg_paths: 2.313725
  svg_path_commands: 44.360784
  render_time_ms: 25.694973
  missing_svg: 0
  render_errors: 500 xkuchar@akeso:/var/tmp/xkuchar/projects/svg-generator$ uv run python scripts/evaluate_raster_vectorization.py z-image-renders/base_prefixed_lora/ ../../OmniSVG/rendered2_4b
  Evaluating raster/vector pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [12:49<00:00,
  valid: 1002
  invalid: 8
  mae: 38.546811
  psnr: 12.066503
  ssim: 0.550203
  mask_iou: 0.548975
  boundary_f1_1px: 0.411653
  boundary_f1_2px: 0.526917
  boundary_f1_4px: 0.639779
  chamfer_px: 22.148928
  hausdorff_px: 167.652673
  svg_bytes: 6696.195609
  svg_elements: 6.461078
  Evaluating raster/vector pairs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1010/1010 [12:31<00:00,
  1.34it/s]
  images: 1010
  valid: 996
  invalid: 14
  valid_rate: 0.986139
  mse: 9923.598002
  mae: 42.472481
  psnr: 11.583518
  ssim: 0.525772
  mask_iou: 0.532133
  boundary_f1_1px: 0.398455
  chamfer_px: 24.054087
  hausdorff_px: 172.407555
  svg_bytes: 6803.540161
  svg_elements: 10.887550
  svg_paths: 9.887550
  svg_path_commands: 250.280120
  render_time_ms: 24.824815
  missing_svg: 14
  render_errors: 0
  ```
