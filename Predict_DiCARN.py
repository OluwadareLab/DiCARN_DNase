import sys
import os
import time
import multiprocessing
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from math import log10
import torch

from Models.DiCARN_model import Generator

from Utils.SSIM import ssim
from Utils.GenomeDISCO import compute_reproducibility
from Utils.io import spreadM, together
from Arg_Parser import *


def dataloader(data, batch_size=64):
	inputs = torch.tensor(data['data'], dtype=torch.float)
	target = torch.tensor(data['target'], dtype=torch.float)
	inds = torch.tensor(data['inds'], dtype=torch.long)
	dataset = TensorDataset(inputs, target, inds)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	return loader
	
def get_chr_nums(data):
	inds = torch.tensor(data['inds'], dtype=torch.long)
	chr_nums = sorted(list(np.unique(inds[:, 0])))
	return chr_nums


def data_info(data):
	indices = data['inds']
	compacts = data['compacts'][()]
	sizes = data['sizes'][()]
	return indices, compacts, sizes


get_digit = lambda x: int(''.join(list(filter(str.isdigit, x))))


def filename_parser(filename):
	info_str = filename.split('.')[0].split('_')[2:-1]
	chunk = get_digit(info_str[0])
	stride = get_digit(info_str[1])
	bound = get_digit(info_str[2])
	scale = 1 if info_str[3] == 'nonpool' else get_digit(info_str[3])
	return chunk, stride, bound, scale


def hicarn_predictor(model, hicarn_loader, ckpt_file, device, data_file):
	# For DiCARN
	deepmodel = Generator(num_channels=64).to(device)

	# For HiCSR & DFHiC
	# deepmodel = Generator().to(device)

	# deepmodel = model.Generator(num_channels=64).to(device)
	if not os.path.isfile(ckpt_file):
		ckpt_file = f'save/{ckpt_file}'
	deepmodel.load_state_dict(torch.load(ckpt_file, map_location=device))
	print(f'Loading checkpoint file from "{ckpt_file}"')

	result_data = []
	result_inds = []
		
	chr_nums = get_chr_nums(data_file)
	print("Chromosomes: ", chr_nums)
	
	results_dict = dict()
	test_metrics = dict()
	for chr in chr_nums:
		test_metrics[f'{chr}'] = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'nsamples': 0}
		results_dict[f'{chr}'] = [[], [], [], []]  # Make respective lists for ssim, psnr, mse, and repro   
	
	deepmodel.eval()
	with torch.no_grad():
		for batch in tqdm(hicarn_loader, desc='DiCARN Model Testing...: '):
			lr, hr, inds = batch
			batch_size = lr.size(0)
			ind = f'{(inds[0][0]).item()}'
			test_metrics[ind]['nsamples'] += batch_size
			lr = lr.to(device)
			hr = hr.to(device)
			out = deepmodel(lr)
			
			batch_mse = ((out - hr) ** 2).mean()
			test_metrics[ind]['mse'] += batch_mse * batch_size
			batch_ssim = ssim(out[:, 0:1, :, :], hr[:, 0:1, :, :])  # Ensure single channel for SSIM
			test_metrics[ind]['ssims'] += batch_ssim * batch_size
			test_metrics[ind]['psnr'] = 10 * log10(1 / (test_metrics[ind]['mse'] / test_metrics[ind]['nsamples']))
			test_metrics[ind]['ssim'] = test_metrics[ind]['ssims'] / test_metrics[ind]['nsamples']

			((results_dict[ind])[0]).append((test_metrics[ind]['ssim']).item())
			((results_dict[ind])[1]).append(batch_mse.item())
			((results_dict[ind])[2]).append(test_metrics[ind]['psnr'])
			
			for i, j in zip(hr, out):
				out1 = torch.squeeze(j, dim=0)
				hr1 = torch.squeeze(i, dim=0)
				out2 = out1.cpu().detach().numpy()
				hr2 = hr1.cpu().detach().numpy()
				genomeDISCO = compute_reproducibility(out2, hr2, transition=True)
				((results_dict[ind])[3]).append(genomeDISCO)
				
			result_data.append(out.cpu().numpy())
			result_inds.append(inds.numpy())
	result_data = np.concatenate(result_data, axis=0)
	result_inds = np.concatenate(result_inds, axis=0)
	
	mean_ssims = []
	mean_mses = []
	mean_psnrs = []
	mean_gds = []
		
	for key, value in results_dict.items():
		value[0] = round(sum(value[0])/len(value[0]), 4)
		value[1] = round(sum(value[1])/len(value[1]), 4)
		value[2] = round(sum(value[2])/len(value[2]), 4)
		value[3] = round(sum(value[3])/len(value[3]), 4)

		mean_ssims.append(value[0])
		mean_mses.append(value[1])
		mean_psnrs.append(value[2])
		mean_gds.append(value[3])
		
		print("\n")
		print("Chr", key, "SSIM: ", value[0])
		print("Chr", key, "MSE: ", value[1])
		print("Chr", key, "PSNR: ", value[2])
		print("Chr", key, "GenomeDISCO: ", value[3])

	print("\n")
	print("___________________________________________")
	print("Means across chromosomes")
	print("SSIM: ", round(sum(mean_ssims) / len(mean_ssims), 4))
	print("MSE: ", round(sum(mean_mses) / len(mean_mses), 4))
	print("PSNR: ", round(sum(mean_psnrs) / len(mean_psnrs), 4))
	print("GenomeDISCO: ", round(sum(mean_gds) / len(mean_gds), 4))
	print("___________________________________________")
	print("\n")
	
	hicarn_hics = together(result_data, result_inds, tag='Reconstructing: ')
	return hicarn_hics        	
	

def save_data(carn, compact, size, file):
	hicarn = spreadM(carn, compact, size, convert_int=False, verbose=True)
	np.savez_compressed(file, hicarn=hicarn, compact=compact)
	print('Saving file:', file)


if __name__ == '__main__':
	args = data_predict_parser().parse_args(sys.argv[1:])
	cell_line = args.cell_line
	low_res = args.low_res
	ckpt_file = args.checkpoint
	cuda = args.cuda
	model = args.model
	HiCARN_file = args.file_name
	print('WARNING: Predict process requires large memory, thus ensure that your machine has ~150G memory.')
	if multiprocessing.cpu_count() > 23:
		pool_num = 23
	else:
		exit()

	in_dir = os.path.join(root_dir, 'data')
	out_dir = os.path.join(root_dir, 'predict', cell_line)
	mkdir(out_dir)

	files = [f for f in os.listdir(in_dir) if f.find(low_res) >= 0]

	chunk, stride, bound, scale = filename_parser(HiCARN_file)

	device = torch.device(
		f'cuda:{cuda}' if (torch.cuda.is_available() and cuda > -1 and cuda < torch.cuda.device_count()) else 'cpu')
	print(f'Using device: {device}')


	start = time.time()
	print(f'Loading data[GM12878]: {HiCARN_file}')
	hc_data = os.path.join(in_dir, HiCARN_file)
	hicarn_data = np.load(os.path.join(in_dir, HiCARN_file), allow_pickle=True)
	hicarn_loader = dataloader(hicarn_data)

	print("Input Test Data: ", hc_data)
	
	print("Checkpoint in use: ", ckpt_file)
	# exit()


	indices, compacts, sizes = data_info(hicarn_data)

	# if model == "HiCSR":
	# 	model = HiCARN_1

	# if model == "DiCARN":
	# 	model = DiCARN

	# if model == "DeepHiC":
	# 	model = DeepHiC

	hicarn_hics = hicarn_predictor(model, hicarn_loader, ckpt_file, device, hicarn_data)


	def save_data_n(key):
		file = os.path.join(out_dir, f'predict_chr{key}_{low_res}.npz')
		save_data(hicarn_hics[key], compacts[key], sizes[key], file)



	pool = multiprocessing.Pool(processes=pool_num)
	print(f'Start a multiprocess pool with process_num = {pool_num} for saving predicted data')
	for key in compacts.keys():
		pool.apply_async(save_data_n, (key,))
	pool.close()
	pool.join()
	print(f'All data saved. Running cost is {(time.time() - start) / 60:.1f} min.')
