import torch
from model.architecture import create_model
from train import train
from data.brats_dataset import train_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model()
train(
    model=model,
    train_loader=train_loader,
    device=device,
    num_epochs=300,
    lr=1e-4,
    checkpoint_dir="/home/nvsinh1/brats_segmentation/synthesis-model/model",
)