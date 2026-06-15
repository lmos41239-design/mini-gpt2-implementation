import os

import torch

import config
from model import MiniGPT


torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/alice_wonderland_sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(indices):
    return "".join([itos[i] for i in indices])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - config.block_size, (config.batch_size,))
    x = torch.stack([source[i : i + config.block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + config.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


model = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

os.makedirs("results", exist_ok=True)
result_lines = []

for step in range(config.max_iters + 1):
    if step % config.eval_interval == 0:
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        line = f"step {step} train loss: {loss.item():.3f}"
        print(line)
        result_lines.append(line)

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "chars": chars,
        "vocab_size": vocab_size,
    },
    "model_checkpoint.pt",
)

with open("results/training_result.txt", "w", encoding="utf-8") as file:
    file.write("Dataset: Alice in Wonderland sample\n")
    file.write("This dataset is different from tiny Shakespeare.\n\n")
    file.write("\n".join(result_lines))
    file.write("\n")
