import os

import torch

from model import MiniGPT


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load("model_checkpoint.pt", map_location=device)
chars = checkpoint["chars"]
vocab_size = checkpoint["vocab_size"]

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def decode(indices):
    return "".join([itos[i] for i in indices])


model = MiniGPT(vocab_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

start = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = decode(model.generate(start, max_new_tokens=300)[0].tolist())

os.makedirs("samples", exist_ok=True)
with open("samples/generated_sample.txt", "w", encoding="utf-8") as file:
    file.write(generated)

print(generated)
