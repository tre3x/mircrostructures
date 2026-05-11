#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.datasets import ForwardSurrogateDataset, normalize_image, normalize_vector
from training.models import ForwardSurrogateNet


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate a trained forward surrogate checkpoint")
	parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt or last_model.pt")
	parser.add_argument("--x", type=str, required=True, help="Path to X tensor (.npy)")
	parser.add_argument("--y_fields", type=str, default="", help="Optional Y field tensor (.npy)")
	parser.add_argument("--y_global", type=str, default="", help="Optional Y global tensor (.npy)")
	parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Which split to evaluate")
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
	parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, or auto")
	parser.add_argument("--output", type=str, default="evaluation_metrics.json", help="Output metrics JSON")
	return parser.parse_args()


def _resolve_device(device_arg):
	if device_arg != "auto":
		return torch.device(device_arg)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def _reduce_field_metrics(pred, target):
	diff = pred - target
	return {
		"mse": float(torch.mean(diff ** 2).item()),
		"mae": float(torch.mean(torch.abs(diff)).item()),
	}


def _reduce_global_metrics(pred, target):
	diff = pred - target
	return {
		"mse": float(torch.mean(diff ** 2).item()),
		"mae": float(torch.mean(torch.abs(diff)).item()),
	}


def main():
	args = parse_args()
	checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	config = checkpoint["config"]
	stats = checkpoint["stats"]
	meta = checkpoint["meta"]

	device = _resolve_device(args.device)
	x_array = np.asarray(np.load(args.x), dtype=np.float32)
	x_norm = normalize_image(x_array, np.asarray(stats["x_mean"]), np.asarray(stats["x_std"]))

	y_fields = np.asarray(np.load(args.y_fields), dtype=np.float32) if args.y_fields else None
	y_global = np.asarray(np.load(args.y_global), dtype=np.float32) if args.y_global else None
	if y_fields is not None and stats["y_field_mean"] is not None:
		y_fields = normalize_image(y_fields, np.asarray(stats["y_field_mean"]), np.asarray(stats["y_field_std"]))
	if y_global is not None and stats["y_global_mean"] is not None:
		y_global = normalize_vector(y_global, np.asarray(stats["y_global_mean"]), np.asarray(stats["y_global_std"]))

	if args.split == "train":
		indices = np.asarray(meta["train_indices"], dtype=np.int64)
	elif args.split == "val":
		indices = np.asarray(meta["val_indices"], dtype=np.int64)
	else:
		indices = np.arange(x_array.shape[3], dtype=np.int64)

	dataset = ForwardSurrogateDataset(x_norm, indices, y_fields=y_fields, y_global=y_global)
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

	model = ForwardSurrogateNet(
		in_channels=config["in_channels"],
		out_field_channels=config["out_field_channels"],
		out_global_channels=config["out_global_channels"],
		base_channels=config["base_channels"],
	).to(device)
	model.load_state_dict(checkpoint["model_state"])
	model.eval()

	field_metrics = []
	global_metrics = []

	with torch.no_grad():
		for batch in tqdm(loader, desc="Evaluating", unit="batch"):
			x_batch = batch["x"].to(device)
			outputs = model(x_batch)
			if "y_fields" in batch and "fields" in outputs:
				field_metrics.append(_reduce_field_metrics(outputs["fields"], batch["y_fields"].to(device)))
			if "y_global" in batch and "global" in outputs:
				global_metrics.append(_reduce_global_metrics(outputs["global"], batch["y_global"].to(device)))

	def _average(metrics_list):
		if not metrics_list:
			return None
		return {
			"mse": float(np.mean([item["mse"] for item in metrics_list])),
			"mae": float(np.mean([item["mae"] for item in metrics_list])),
		}

	result = {
		"checkpoint": os.path.abspath(args.checkpoint),
		"split": args.split,
		"num_samples": int(len(indices)),
		"field_metrics": _average(field_metrics),
		"global_metrics": _average(global_metrics),
	}

	with open(args.output, "w") as fobj:
		json.dump(result, fobj, indent=2)

	print("Saved metrics:", args.output)
	print(result)


if __name__ == "__main__":
	main()
