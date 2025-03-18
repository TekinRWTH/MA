#Hier wird noch zusätzlich GPU-Auslastung gemessen#

import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not installed. GPU usage stats will not be recorded.")

# For CPU usage
import psutil

# ----------------------------
# Replace these with your own:
# ----------------------------
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader

torch.backends.cudnn.benchmark = True

def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
    epoch, gpu_handles
):
    loop = tqdm(loader, leave=True, desc=f"Epoch {epoch+1}")

    # Track GPU stats
    gpu_usage_stats = {f"gpu_{i}_usage": [] for i in range(len(gpu_handles))}

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # ---------------------
        #  Train Discriminator
        # ---------------------
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

        # -----------------
        #  Train Generator
        # -----------------
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Sample GPU stats every 10 batches
        if idx % 10 == 0 and gpu_handles:
            for i, handle in enumerate(gpu_handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage_stats[f"gpu_{i}_usage"].append(util.gpu)

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )

    # Calculate averages
    stats = {}
    for metric in gpu_usage_stats:
        stats[f"{metric}_percent"] = np.mean(gpu_usage_stats[metric]) if gpu_usage_stats[metric] else 0
        
    return stats

def main():
    # Initialize NVML
    gpu_handles = []
    if PYNVML_AVAILABLE:
        pynvml.nvmlInit()
        num_gpus = pynvml.nvmlDeviceGetCount()
        gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_gpus)]

    # Model setup (ONLY CHANGE FROM ORIGINAL)
    disc = Discriminator(in_channels=3)
    gen = Generator(in_channels=3, features=64)
    
    # Add DataParallel wrapper
    if torch.cuda.device_count() >= 1:
        disc = nn.DataParallel(disc)
        gen = nn.DataParallel(gen)
    
    disc = disc.to(config.DEVICE)
    gen = gen.to(config.DEVICE)

    # Rest remains identical to original code
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen.module, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc.module, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )

    val_dataset = MapDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    epoch_stats = []
    total_start = time.time()

    for epoch in range(config.NUM_EPOCHS):
        psutil.cpu_percent(interval=None)  # Reset CPU counter
        
        epoch_start = time.time()
        gpu_stats = train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE,
            g_scaler, d_scaler, epoch, gpu_handles
        )
        epoch_end = time.time()

        stats = {
            "epoch": epoch + 1,
            "epoch_time_secs": epoch_end - epoch_start,
            "cpu_usage_percent": psutil.cpu_percent(interval=None),
            **gpu_stats
        }
        epoch_stats.append(stats)

        if config.SAVE_MODEL and (epoch % 10 == 0 or epoch == (config.NUM_EPOCHS - 1)):
            save_checkpoint(gen.module, opt_gen, epoch, config.CHECKPOINT_GEN)
            save_checkpoint(disc.module, opt_disc, epoch, config.CHECKPOINT_DISC)
            pd.DataFrame(epoch_stats).to_csv(config.STATS_PATH, index=False)

    total_end = time.time()
    pd.DataFrame(epoch_stats).to_csv(config.STATS_PATH, index=False)
    
    if PYNVML_AVAILABLE:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
