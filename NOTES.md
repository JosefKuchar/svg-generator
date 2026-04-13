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
uv run inference.py --help
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
