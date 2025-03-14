import torch
from utilsinference import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import configinference
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
import os

torch.backends.cudnn.benchmark = True

def inference_fn(gen, disc, bce, D_real_losses, D_fake_losses, G_losses, l1_loss, loader):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):  
        x = x.to(configinference.DEVICE)
        y = y.to(configinference.DEVICE)

        with torch.no_grad():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        D_real_losses.append(D_real_loss.item())
        D_fake_losses.append(D_fake_loss.item())

        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * configinference.L1_LAMBDA
            G_loss = G_fake_loss + L1

        G_losses.append(G_loss.item())

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    df = pd.DataFrame({
        "D_real_loss": D_real_losses,
        "D_fake_loss": D_fake_losses,
        "G_loss": G_losses,
    })
    output_dir = "/home/dt681254/jupyterlab/CGAN"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "lossestest.csv"), index=False)
    print(f"Saved losses to {os.path.join(output_dir, 'lossestest.csv')}")

def main():
    disc = Discriminator(in_channels=3).to(configinference.DEVICE)
    gen = Generator(in_channels=3, features=64).to(configinference.DEVICE)
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if configinference.LOAD_MODEL:
        load_checkpoint(configinference.CHECKPOINT_GEN, gen, None, configinference.LEARNING_RATE)

    gen.eval()

    val_dataset = MapDataset(root_dir=configinference.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    start_epoch = 0
    
    for epoch in range(start_epoch, configinference.NUM_EPOCHS):
        D_real_losses = []
        D_fake_losses = []
        G_losses = []

        inference_fn(gen, disc, BCE, D_real_losses, D_fake_losses, G_losses, L1_LOSS, val_loader)
    
        save_some_examples(gen, val_loader, epoch, folder="/home/dt681254/jupyterlab/CGAN/inference")

if __name__ == "__main__":
    main()
