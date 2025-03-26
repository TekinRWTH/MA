import torch
from utilsinference import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import configinference  # Keep original config without BATCH_SIZE
from dataset import MapDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import pandas as pd
import os
import time

# Convert configinference.DEVICE string to a torch.device object
DEVICE = torch.device(configinference.DEVICE)

# Set CUDA backend only if using a CUDA device
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

def inference_fn(gen, disc, bce, D_real_losses, D_fake_losses, G_losses, l1_loss, loader):
    loop = tqdm(loader, leave=True)
    inference_times = []
    fps_values = []
    gpu_memory = []  # Track GPU memory per batch

    for idx, (x, y) in enumerate(loop):  
        x, y = x.to(DEVICE), y.to(DEVICE)
        current_batch_size = x.size(0)  # Get actual batch size

        # Reset CUDA memory stats
        if DEVICE.type == 'cuda':
            for gpu_id in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(gpu_id)

        # Timing
        start_time = time.time()
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()

        with torch.no_grad():
            y_fake = gen(x)

        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        elapsed_time = time.time() - start_time

        inference_times.append(elapsed_time)
        fps_values.append(current_batch_size / elapsed_time if elapsed_time > 0 else 0)  # Changed FPS calculation

        # GPU memory tracking
        batch_gpu_memory = []
        if DEVICE.type == 'cuda':
            for gpu_id in range(torch.cuda.device_count()):
                mem = torch.cuda.max_memory_allocated(gpu_id)
                batch_gpu_memory.append(mem)
        gpu_memory.append(batch_gpu_memory)

        # Discriminator evaluation
        with torch.no_grad():
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        D_real_losses.append(D_real_loss.item())
        D_fake_losses.append(D_fake_loss.item())

        # Generator evaluation
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
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

    # Prepare data dictionary
    data_dict = {
        "D_real_loss": D_real_losses,
        "D_fake_loss": D_fake_losses,
        "G_loss": G_losses,
        "inference_time": inference_times,
        "fps": fps_values
    }

    if DEVICE.type == 'cuda' and len(gpu_memory) > 0:
        num_gpus = len(gpu_memory[0])
        for gpu_id in range(num_gpus):
            data_dict[f"gpu_{gpu_id}_memory"] = [batch[gpu_id] for batch in gpu_memory]

    df = pd.DataFrame(data_dict)
    output_dir = "/home/dt681254/jupyterlab/CGAN"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "losses_and_timing.csv"), index=False)
    print("Metrics saved.")

def main():
    # Initialize models
    disc = Discriminator(in_channels=3).to(DEVICE)
    gen = Generator(in_channels=3, features=64).to(DEVICE)
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if configinference.LOAD_MODEL:
        load_checkpoint(configinference.CHECKPOINT_GEN, gen, None, configinference.LEARNING_RATE)
        load_checkpoint(configinference.CHECKPOINT_DISC, disc, None, configinference.LEARNING_RATE)

    gen.eval()
    disc.eval()

    val_dataset = MapDataset(root_dir=configinference.VAL_DIR)
    
    # Hardcoded batch sizes here
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,  # Change this to 8 or 16 directly
        shuffle=False
    )
    
    for epoch in range(configinference.NUM_EPOCHS):
        D_real_losses, D_fake_losses, G_losses = [], [], []
        inference_fn(gen, disc, BCE, D_real_losses, D_fake_losses, G_losses, L1_LOSS, val_loader)
        
        save_some_examples(gen, val_loader, epoch, folder="/rwthfs/rz/cluster/home/dt681254/jupyterlab/CGAN/inference")

if __name__ == "__main__":
    main()
