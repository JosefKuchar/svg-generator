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
uv run export_svg_svgrepo_dataset.py --caption-index 1 --caption-prefix "SVG illustration with white background. " --num-samples 8000
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
uv run benchmark_image_folders.py ./raster/reference/ z-image-renders/base
clip_similarity: 0.818210
dino_similarity: 0.509159
vectorization_mse: 266.565137

# Base prefixed
uv run benchmark_image_folders.py ./raster/reference/ z-image-renders/base-prefixed
clip_similarity: 0.819865
dino_similarity: 0.545802
vectorization_mse: 230.160058

# Turbo
uv run benchmark_image_folders.py ./raster/reference/ z-image-renders/turbo
clip_similarity: 0.826786
dino_similarity: 0.509892
vectorization_mse: 227.691742

# Turbo prefixed
uv run benchmark_image_folders.py ./raster/reference/ z-image-renders/turbo-prefixed
clip_similarity: 0.871237
dino_similarity: 0.583856
vectorization_mse: 142.711678

# Turbo prefixed with LORA - provisional
clip_similarity: 0.879104
dino_similarity: 0.600208
vectorization_mse: 143.174617

```

TODOS:
