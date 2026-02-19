import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Efficient-Large-Model/Sana_600M_512px_diffusers"
PROMPTS = [
    "A clean black-and-white line drawing of a bird",
    "A minimal logo of a mountain made of one continuous stroke",
    "A simple geometric icon of a tree",
]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_text_encoder(model_id: str = MODEL_ID, device: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer.padding_side = "right"
    model = AutoModel.from_pretrained(model_id, subfolder="text_encoder")
    resolved_device = device or get_device()
    model = model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


@torch.no_grad()
def encode_prompts(prompts, tokenizer, model, device, max_length: int = 300):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    attention_mask = inputs["attention_mask"]
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state, attention_mask


if __name__ == "__main__":
    tokenizer, model, device = load_text_encoder()
    embeddings, attention_mask = encode_prompts(PROMPTS, tokenizer, model, device)

    print(f"device={device}")
    print(f"attention_mask_shape={tuple(attention_mask.shape)}")
    print(f"attention_tokens_per_prompt={attention_mask.sum(dim=1).tolist()}")
    print(f"last_hidden_state_shape={tuple(embeddings.shape)}")
