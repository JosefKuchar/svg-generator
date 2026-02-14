# Flow matching vectorizer
This project is a raster to bezier curve vectorizer. The model is flow matching based - conditioned on raster images (using DINOv3 image features) and trained to predict the control points of bezier curves (and other attributes).

## Tech stack
- PyTorch
- PyTorch Lightning

## Project structure
- `model.py` - main model implementation
- `representation.py` - bezier curve representation and utilities
- `dataset.py` - dataset loading and preprocessing
- `main.py` - main training loop
- `infer.py` - inference script for testing trained models
- `raster.py` - utilities for raster image processing
- `parsing.py` - utilities for creating dataset from SVG files
