# Flow matching vectorizer
This project is a raster to bezier curve vectorizer. The model is flow matching based - conditioned on raster images (using DINOv3 image features) and trained to predict the control points of bezier curves (and other attributes).

## Tech stack
- PyTorch
- PyTorch Lightning

## Project structure
- `model.py` - main model implementation
- `representation.py` - bezier curve representation and utilities
- `dataset.py` - dataset loading and preprocessing
- `train.py` - main training loop
- `infer.py` - inference script for testing trained models
- `raster.py` - utilities for raster image processing
- `parsing.py` - utilities for creating dataset from SVG files

## Thesis text
This project also includes a thesis text.
The text is written in Typst, a modern typesetting system. The source files for the text are located in the `text/` directory.
When writing, use academic style and ensure that the text is well-structured and clear.
Prefer motivation before implementation: first explain why a design choice is needed, then describe how it is implemented.
Avoid overly long paragraphs; split them when they contain multiple ideas or become hard to scan.
Use `typst compile text/main.typ` to check that the text compiles correctly.

When adding citations, check the internet first to ensure the bibtext entry is correct. Don't make up the citations from memory.

### Official requirements
This thesis aims to develop a two-stage pipeline that enables the creation of neural networks capable of generating Scalable Vector Graphics (SVGs) from text prompts. The first stage leverages a pretrained bitmap image generation model, while the second stage requires development of a custom model for vectorization, i.e., converting the bitmap outputs into vector graphics.

Towards this goal, the thesis will create:

An implementation of a training pipeline for adapting a bitmap generation model,
A design and implementation of a training pipeline for creating a vectorization model utilizing synthetic data,
An experimental comparison of flow-matching and autoregressive model architecture for vectorization.
The resulting end-to-end pipeline for generating SVGs will be demonstrated on unseen text prompts. All implementation code will be published in IS under the MIT license.

### Keywords
It should be 5 - 10 keywords. By a keyword, we also mean a term expressed by multiple words (a collocation). Keywords should capture the issue addressed in the work, rather than being a list of the technologies used.

### Structure
The first chapter of the thesis is an introduction that serves to place the addressed issue into a broader context. The introductory chapter must clearly formulate the objectives of the work; furthermore, it may outline the structure of the thesis by briefly describing the content of the individual chapters. This is followed by an analysis of the problem and a description of the solution. The final chapter contains an evaluation of the achieved results, with special emphasis on the author's own contribution and an assessment of the addressed issue from a broader perspective. In conclusion, it is also advisable to indicate possible directions for future research or development.
