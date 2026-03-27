import torch
from model.architecture import create_model
from pretrained_train import train
from data.brats_dataset import train_loader, val_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(
    latent_dim=128,
    num_timesteps=1000,
    num_modalities=4,
    num_domains=3,
)

train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=300,
    lr=1e-5,
    checkpoint_dir="/home/nvsinh1/brats_segmentation/synthesis-model/model",
    eval_every=5,
    val_infer_steps=1000,
)
