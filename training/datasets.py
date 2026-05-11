#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


def split_indices(num_samples, val_fraction, seed):
	if num_samples < 2:
		return np.arange(num_samples, dtype=np.int64), np.arange(0, dtype=np.int64)

	val_count = max(1, int(round(num_samples * val_fraction)))
	val_count = min(val_count, num_samples - 1)

	rng = np.random.default_rng(seed)
	indices = np.arange(num_samples, dtype=np.int64)
	rng.shuffle(indices)
	return np.sort(indices[val_count:]), np.sort(indices[:val_count])


def compute_image_stats(array, indices):
	subset = np.asarray(array[:, :, :, indices], dtype=np.float32)
	mean = subset.mean(axis=(0, 1, 3))
	std = subset.std(axis=(0, 1, 3))
	std = np.where(std < 1.0e-8, 1.0, std)
	return mean.astype(np.float32), std.astype(np.float32)


def compute_vector_stats(array, indices):
	subset = np.asarray(array[:, indices], dtype=np.float32)
	mean = subset.mean(axis=1)
	std = subset.std(axis=1)
	std = np.where(std < 1.0e-8, 1.0, std)
	return mean.astype(np.float32), std.astype(np.float32)


def normalize_image(array, mean, std):
	return (array - mean[None, None, :, None]) / std[None, None, :, None]


def normalize_vector(array, mean, std):
	return (array - mean[:, None]) / std[:, None]


class ForwardSurrogateDataset(Dataset):
	def __init__(self, x_array, indices, y_fields=None, y_global=None):
		self.x_array = np.asarray(x_array, dtype=np.float32)
		self.indices = np.asarray(indices, dtype=np.int64)
		self.y_fields = None if y_fields is None else np.asarray(y_fields, dtype=np.float32)
		self.y_global = None if y_global is None else np.asarray(y_global, dtype=np.float32)

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, item):
		sample_index = int(self.indices[item])
		x_item = torch.from_numpy(np.transpose(self.x_array[:, :, :, sample_index], (2, 0, 1)))
		batch = {"x": x_item, "index": sample_index}
		if self.y_fields is not None:
			batch["y_fields"] = torch.from_numpy(np.transpose(self.y_fields[:, :, :, sample_index], (2, 0, 1)))
		if self.y_global is not None:
			batch["y_global"] = torch.from_numpy(self.y_global[:, sample_index])
		return batch
