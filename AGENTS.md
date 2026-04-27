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

## Thesis text
This project also includes a thesis text.
The text is written in Typst, a modern typesetting system. The source files for the text are located in the `text/` directory.
When writing, use academic style and ensure that the text is well-structured and clear.
Use `typst compile text/main.typ` to check that the text compiles correctly.

When adding citations, check the internet first to ensure the bibtext entry is correct. Don't make up the citations from memory.
