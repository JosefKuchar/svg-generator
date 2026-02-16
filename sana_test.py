from sana_transformer import SanaTransformer2DModel

# Load weights from the original model
model = SanaTransformer2DModel.from_pretrained(
    "Efficient-Large-Model/Sana_600M_512px_diffusers", subfolder="transformer"
)
