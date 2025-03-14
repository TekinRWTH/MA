import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
import os

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, 
    D_real_losses, D_fake_losses, G_losses, epoch  
):
  
    loop = tqdm(loader, leave=True, desc=f"Epoch {epoch+1}") 

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        D_real_losses.append(D_real_loss.item())
        D_fake_losses.append(D_fake_loss.item())
        

     
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        
        G_losses.append(G_loss.item())

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()


    D_real_losses_total = []
    D_fake_losses_total = []
    G_losses_total = []

  
    start_epoch = 0

   
    if config.LOAD_MODEL:
        start_epoch = load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

       
        if os.path.exists("/home/dt681254/jupyterlab/CGAN/losses.csv"):
            df = pd.read_csv("/home/dt681254/jupyterlab/CGAN/losses.csv")
            D_real_losses_total = df["D_real_loss"].tolist()
            D_fake_losses_total = df["D_fake_loss"].tolist()
            G_losses_total = df["G_loss"].tolist()
            print("Loaded existing losses from CSV.")

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    for epoch in range(start_epoch, config.NUM_EPOCHS):
        D_real_losses = []
        D_fake_losses = []
        G_losses = []

        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
            D_real_losses, D_fake_losses, G_losses, epoch  
        )

    
        D_real_losses_total.extend(D_real_losses)
        D_fake_losses_total.extend(D_fake_losses)
        G_losses_total.extend(G_losses)

        
        if config.SAVE_MODEL and (epoch % 10 == 0 or epoch == 681):
            print(f"Saving checkpoint at epoch {epoch}...")
            save_checkpoint(model=gen, optimizer=opt_gen, epoch=epoch, filename=config.CHECKPOINT_GEN)
            save_checkpoint(model=disc, optimizer=opt_disc, epoch=epoch, filename=config.CHECKPOINT_DISC)

        
            loss_data = {
                "D_real_loss": D_real_losses_total,
                "D_fake_loss": D_fake_losses_total,
                "G_loss": G_losses_total
            }
            df = pd.DataFrame(loss_data)
            df.to_csv("/home/dt681254/jupyterlab/CGAN/losses.csv", index=False)

    
        save_some_examples(gen, val_loader, epoch, folder="/home/dt681254/jupyterlab/CGAN/evaluation")

if __name__ == "__main__":
    main()
