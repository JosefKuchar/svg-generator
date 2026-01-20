import typer
import torch
import torch.nn as nn
import math
import lightning as pl
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
from torch.utils.data import IterableDataset, DataLoader, Dataset
from datasets import load_dataset
import imageio
import os
from parsing import save_bezier_shapes_to_svg, BezierShape
from raster import render_svg
import glob

app = typer.Typer()


@app.command()
def app():
    torch.set_float32_matmul_precision("medium")

    # module = FlowMatchingTransformer(
    #     input_dim=11, cond_dim=2, hidden_size=256, num_layers=6, num_heads=8
    # )

    # trainer = pl.Trainer(max_epochs=500, limit_train_batches=300, accelerator="auto")
    # trainer.fit(
    #     module,
    #     datamodule=DataModule(),
    # )

    # Load lightning checkpoint

    # Find .ckpt in folder
    ckpt_files = glob.glob("./lightning_logs/version_30/checkpoints/*.ckpt")
    ckpt_files.sort(key=os.path.getmtime)
    latest_ckpt = ckpt_files[-1]

    module = FlowMatchingTransformer.load_from_checkpoint(latest_ckpt)

    # Test
    # cond = torch.zeros(1, 2).to(module.device)
    cond = torch.tensor([[0.0, 0.0]]).to(module.device)
    x = module.sample(cond, shape=(1, 128, 11)).to("cpu")
    shapes = tensor_to_bezier_shapes(x, 256, 256)
    print("before")
    print(shapes[0].curves)
    shapes = close_shapes(shapes, epsilon=10.0)
    print("after")
    print(shapes[0].curves)
    svg = save_bezier_shapes_to_svg(shapes, 256, 256)
    with open("blob.svg", "w") as f:
        f.write(svg)
    image = render_svg(svg)
    image.save("blob.png")


if __name__ == "__main__":
    app()
