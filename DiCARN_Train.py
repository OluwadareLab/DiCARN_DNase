import sys
import time
import multiprocessing
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import torch.nn as nn
# from Models.HiCARN_1 import Generator
from Models.DC_orig_FrEndDL import Generator
from Models.HiCARN_1_Loss import GeneratorLoss
from Utils.SSIM import ssim
from math import log10
from Arg_Parser import root_dir

print("Training With MSE\n")
# Additions for timing and memory measurement
start_time = time.time()

# Check if CUDA is available, and reset peak memory statistics
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

cs = np.column_stack
#
#
#
root_dir = ''
#
#
#
def adjust_learning_rate(epoch):
    lr = 0.0003 * (0.1 ** (epoch // 30))
    return lr

# data_dir: directory storing processed data
data_dir = os.path.join(root_dir, 'Data/R16_down/K562/data')
dnase_dir = os.path.join(root_dir, 'Data/DNase/target/K562/data')  # Directory for DNase data
ckpt_dir = 'checkpoints/R16'
# out_dir: directory storing checkpoint files
out_dir = os.path.join(ckpt_dir, 'DC_DNase_K562_orig_FrEndDL')
os.makedirs(out_dir, exist_ok=True)

datestr = time.strftime('%m_%d_%H_%M')
visdom_str = time.strftime('%m%d')

resos = '10kb40kb'
chunk = 40
stride = 40
bound = 201
pool = 'nonpool'
name = 'DC_DNase_target_K562_orig_FrEndDL'

num_epochs = 100
# num_epochs = 1
batch_size = 64
# batch_size = 128

# whether using GPU for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("CUDA available? ", torch.cuda.is_available())
print("Device being used: ", device)

# prepare training dataset
train_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_train.npz')
train = np.load(train_file)

train_data = torch.tensor(train['data'], dtype=torch.float)
train_target = torch.tensor(train['target'], dtype=torch.float)
train_inds = torch.tensor(train['inds'], dtype=torch.long)

# Load DNase data
dnase_train_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}DNase_train.npz')
dnase_train = np.load(dnase_train_file)

dnase_train_data = torch.tensor(dnase_train['data'], dtype=torch.float)
dnase_train_target = torch.tensor(dnase_train['target'], dtype=torch.float)
dnase_train_inds = torch.tensor(dnase_train['inds'], dtype=torch.long)

# Concatenate DNase data with HiC data
combined_train_data = torch.cat((train_data, dnase_train_data), dim=0)
combined_train_target = torch.cat((train_target, dnase_train_target), dim=0)
combined_train_inds = torch.cat((train_inds, dnase_train_inds), dim=0)

train_set = TensorDataset(combined_train_data, combined_train_target, combined_train_inds)

# prepare valid dataset
valid_file = os.path.join(data_dir, f'hicarn_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_valid.npz')
valid = np.load(valid_file)

valid_data = torch.tensor(valid['data'], dtype=torch.float)
valid_target = torch.tensor(valid['target'], dtype=torch.float)
valid_inds = torch.tensor(valid['inds'], dtype=torch.long)

valid_set = TensorDataset(valid_data, valid_target, valid_inds)

# DataLoader for batched training
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)

# load network
netG = Generator(num_channels=64).to(device)
# netG = Generator().to(device)

# loss function
# criterionG = GeneratorLoss().to(device)
criterionG = nn.MSELoss()

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=0.0003)

ssim_scores = []
psnr_scores = []
mse_scores = []
mae_scores = []

best_ssim = 0
for epoch in range(1, num_epochs + 1):
    run_result = {'nsamples': 0, 'g_loss': 0, 'g_score': 0}

    alr = adjust_learning_rate(epoch)
    optimizerG = optim.Adam(netG.parameters(), lr=alr)

    for p in netG.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()

    netG.train()
    train_bar = tqdm(train_loader)
    for data, target, _ in train_bar:
        batch_size = data.size(0)
        run_result['nsamples'] += batch_size

        real_img = target.to(device)
        z = data.to(device)
        fake_img = netG(z)

        ######### Train generator #########
        netG.zero_grad()
        g_loss = criterionG(fake_img, real_img)
        g_loss.backward()
        optimizerG.step()

        run_result['g_loss'] += g_loss.item() * batch_size

        train_bar.set_description(
            desc=f"[{epoch}/{num_epochs}] Loss_G: {run_result['g_loss'] / run_result['nsamples']:.4f}")
    train_gloss = run_result['g_loss'] / run_result['nsamples']
    train_gscore = run_result['g_score'] / run_result['nsamples']

    valid_result = {'g_loss': 0,
                    'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
    netG.eval()

    batch_ssims = []
    batch_mses = []
    batch_psnrs = []
    batch_maes = []

    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for val_lr, val_hr, inds in valid_bar:
            batch_size = val_lr.size(0)
            valid_result['nsamples'] += batch_size
            lr = val_lr.to(device)
            hr = val_hr.to(device)
            sr = netG(lr)

            sr_out = sr
            hr_out = hr

            # print("Original LR size: ", sr.size())
            # print("Original HR size: ", hr.size())
            # print("Derived SR: ", sr.size())
            # exit()
            g_loss = criterionG(sr, hr)

            valid_result['g_loss'] += g_loss.item() * batch_size

            batch_mse = ((sr - hr) ** 2).mean()
            batch_mae = (abs(sr - hr)).mean()
            valid_result['mse'] += batch_mse * batch_size
            batch_ssim = ssim(sr[:, 0:1, :, :], hr[:, 0:1, :, :])  # Ensure single channel for SSIM
            valid_result['ssims'] += batch_ssim * batch_size
            valid_result['psnr'] = 10 * log10(1 / (valid_result['mse'] / valid_result['nsamples']))
            valid_result['ssim'] = valid_result['ssims'] / valid_result['nsamples']
            valid_bar.set_description(
                desc=f"[Predicting in Test set] PSNR: {valid_result['psnr']:.4f} dB SSIM: {valid_result['ssim']:.4f}")

            batch_ssims.append(valid_result['ssim'])
            batch_psnrs.append(valid_result['psnr'])
            batch_mses.append(batch_mse)
            batch_maes.append(batch_mae)

    ssim_scores.append((sum(batch_ssims) / len(batch_ssims)))
    psnr_scores.append((sum(batch_psnrs) / len(batch_psnrs)))
    mse_scores.append((sum(batch_mses) / len(batch_mses)))
    mae_scores.append((sum(batch_maes) / len(batch_maes)))

    valid_gloss = valid_result['g_loss'] / valid_result['nsamples']
    now_ssim = valid_result['ssim'].item()

    if now_ssim > best_ssim:
        best_ssim = now_ssim
        print(f'Now, Best ssim is {best_ssim:.6f}')
        best_ckpt_file = f'{datestr}_bestg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'
        torch.save(netG.state_dict(), os.path.join(out_dir, best_ckpt_file))
final_ckpt_g = f'{datestr}_finalg_{resos}_c{chunk}_s{stride}_b{bound}_{pool}_{name}.pytorch'


subdir = os.path.join('score_tracker', 'R16')
if not os.path.exists(subdir):
    os.makedirs(subdir)

# Function to ensure all elements are converted to NumPy arrays from either floats or CUDA tensors
def to_numpy(array):
    if isinstance(array[0], torch.Tensor):
        return np.array([item.cpu().numpy() for item in array])
    else:
        return np.array(array)

# Convert the scores to numpy arrays
ssim_scores_np = to_numpy(ssim_scores)
psnr_scores_np = to_numpy(psnr_scores)
mse_scores_np = to_numpy(mse_scores)
mae_scores_np = to_numpy(mae_scores)

# Save the scores to the specified directory
np.savetxt(os.path.join(subdir, f'valid_ssim_scores_{name}.txt'), X=ssim_scores_np, delimiter=',')
np.savetxt(os.path.join(subdir, f'valid_psnr_scores_{name}.txt'), X=psnr_scores_np, delimiter=',')
np.savetxt(os.path.join(subdir, f'valid_mse_scores_{name}.txt'), X=mse_scores_np, delimiter=',')
np.savetxt(os.path.join(subdir, f'valid_mae_scores_{name}.txt'), X=mae_scores_np, delimiter=',')

torch.save(netG.state_dict(), os.path.join(out_dir, final_ckpt_g))
print("\n Checkpoint is saved to: ", out_dir)

end_time = time.time()
total_time = end_time - start_time

# Convert total_time from seconds to hours, minutes, and seconds for printing
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print("\nTotal training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes),seconds))

# Recording and printing peak memory usage
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert bytes to megabytes
    torch.cuda.synchronize()
    print(f"\n Peak GPU memory usage: {peak_memory:.2f} MB")
