import os
import torch
import re

def save_checkpoint(state, is_best, ckpt_dir, max_keep=3):
    os.makedirs(ckpt_dir, exist_ok=True)
    epoch = state["epoch"]
    ckpt_name = f"epoch_{epoch:03d}.pth"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save(state, ckpt_path)
    # ----- Save best -----
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pth")
        torch.save(state, best_path)
    # ----- Remove old checkpoints -----
    ckpts = []
    for f in os.listdir(ckpt_dir):
        if re.match(r"epoch_\d+\.pth", f):
            epoch_id = int(re.findall(r"\d+", f)[0])
            ckpts.append((epoch_id, f))
    ckpts.sort(key=lambda x: x[0])  # sort by epoch
    while len(ckpts) > max_keep:
        _, old_ckpt = ckpts.pop(0)
        os.remove(os.path.join(ckpt_dir, old_ckpt))

def load_checkpoint(ckpt_dir, model, optimizer=None, scheduler=None, ema=None):
    if not os.path.exists(ckpt_dir):
        print("Checkpoint directory not found")
        return 0, float("inf")
    ckpts = []
    for f in os.listdir(ckpt_dir):
        if f.startswith("epoch_") and f.endswith(".pth"):
            epoch_id = int(f.split("_")[1].split(".")[0])
            ckpts.append((epoch_id, f))
    if len(ckpts) == 0:
        print("ℹ️ No checkpoint found, training from scratch")
        return 0, float("inf")
    ckpts.sort(key=lambda x: x[0])
    latest_epoch, latest_ckpt = ckpts[-1]
    ckpt_path = os.path.join(ckpt_dir, latest_ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if ema is not None and "ema_state" in ckpt:
        ema.load_state_dict(ckpt["ema_state"], device=next(model.parameters()).device)
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"✅ Resumed from {latest_ckpt}")
    return ckpt["epoch"] + 1, ckpt["best_loss"]

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
        self.stop = False

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class EMA:
    """Exponential Moving Average for model weights.
    Usage:
      ema = EMA(decay=0.9999)
      ema.register(model)
      ema.update(model)  # after each optimizer.step()
      ema.store(model); ema.copy_to(model)  # to use EMA weights
      ema.restore(model)  # to go back
    """
    def __init__(self, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone().detach()

    def update(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone().detach()

    def store(self, model: torch.nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone().detach()

    def copy_to(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                p.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])

    def state_dict(self):
        return {k: v.clone().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict, device=None):
        self.shadow = {}
        if not isinstance(state_dict, dict):
            return
        for k, v in state_dict.items():
            if device is not None:
                self.shadow[k] = v.clone().to(device)
            else:
                self.shadow[k] = v.clone()

