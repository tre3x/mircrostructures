#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.datasets import (
	ForwardSurrogateDataset,
	compute_image_stats,
	compute_vector_stats,
	normalize_image,
	normalize_vector,
	split_indices,
)
from training.models import ForwardSurrogateNet


def parse_args():
	parser = argparse.ArgumentParser(description="Train a forward surrogate on microstructure-response tensors")
	parser.add_argument("--x", type=str, required=True, help="Path to X tensor (.npy), shape (H, W, C, N)")
	parser.add_argument("--x_meta", type=str, default="", help="Optional X metadata JSON")
	parser.add_argument("--y_fields", type=str, default="", help="Optional Y field tensor (.npy), shape (H, W, C, N)")
	parser.add_argument("--y_global", type=str, default="", help="Optional Y global tensor (.npy), shape (C, N)")
	parser.add_argument("--y_meta", type=str, default="", help="Optional Y metadata JSON")
	parser.add_argument("--task", type=str, default="auto", choices=["auto", "fields", "global", "both"], help="Training task")
	parser.add_argument("--out_dir", type=str, default="training_runs/forward_surrogate", help="Output directory")
	parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
	parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
	parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate")
	parser.add_argument("--weight_decay", type=float, default=1.0e-5, help="AdamW weight decay")
	parser.add_argument("--val_fraction", type=float, default=0.2, help="Validation split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--base_channels", type=int, default=32, help="Base feature width for the U-Net")
	parser.add_argument("--field_loss_weight", type=float, default=1.0, help="Weight for field loss")
	parser.add_argument("--global_loss_weight", type=float, default=1.0, help="Weight for global-response loss")
	parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
	parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps, or auto")
	return parser.parse_args()


def _load_json(path):
	with open(path, "r") as fobj:
		return json.load(fobj)


def _resolve_device(device_arg):
	if device_arg != "auto":
		return torch.device(device_arg)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def _set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _infer_task(args):
	has_fields = bool(args.y_fields)
	has_global = bool(args.y_global)
	if args.task != "auto":
		return args.task
	if has_fields and has_global:
		return "both"
	if has_fields:
		return "fields"
	if has_global:
		return "global"
	raise ValueError("Provide at least one of --y_fields or --y_global.")


def _load_channel_names(meta_path, task):
	if not meta_path:
		return None, None
	meta = _load_json(meta_path)
	field_channels = meta.get("field_channels") if task in ("fields", "both") else None
	global_channels = meta.get("global_channels") if task in ("global", "both") else None
	return field_channels, global_channels


def _compute_loss(outputs, batch, criterion, task, field_loss_weight, global_loss_weight):
	total_loss = 0.0
	loss_parts = {}
	if task in ("fields", "both"):
		field_loss = criterion(outputs["fields"], batch["y_fields"])
		total_loss = total_loss + field_loss_weight * field_loss
		loss_parts["fields"] = field_loss.item()
	if task in ("global", "both"):
		global_loss = criterion(outputs["global"], batch["y_global"])
		total_loss = total_loss + global_loss_weight * global_loss
		loss_parts["global"] = global_loss.item()
	loss_parts["total"] = float(total_loss.item())
	return total_loss, loss_parts


def _run_epoch(model, loader, optimizer, criterion, task, device, field_loss_weight, global_loss_weight, training):
	model.train(training)
	running = {"total": 0.0, "fields": 0.0, "global": 0.0, "count": 0}

	iterator = tqdm(loader, leave=False)
	for batch in iterator:
		for key, value in batch.items():
			if key != "index":
				batch[key] = value.to(device)

		with torch.set_grad_enabled(training):
			outputs = model(batch["x"])
			loss, loss_parts = _compute_loss(outputs, batch, criterion, task, field_loss_weight, global_loss_weight)
			if training:
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()

		running["total"] += loss_parts["total"]
		running["fields"] += loss_parts.get("fields", 0.0)
		running["global"] += loss_parts.get("global", 0.0)
		running["count"] += 1
		iterator.set_description(("train" if training else "val") + " loss={:.4e}".format(loss_parts["total"]))

	if running["count"] == 0:
		return {"total": 0.0, "fields": 0.0, "global": 0.0}

	return {
		"total": running["total"] / running["count"],
		"fields": running["fields"] / running["count"],
		"global": running["global"] / running["count"],
	}


def main():
	args = parse_args()
	task = _infer_task(args)
	os.makedirs(args.out_dir, exist_ok=True)
	_set_seed(args.seed)
	device = _resolve_device(args.device)

	x_array = np.asarray(np.load(args.x), dtype=np.float32)
	y_fields = np.asarray(np.load(args.y_fields), dtype=np.float32) if args.y_fields else None
	y_global = np.asarray(np.load(args.y_global), dtype=np.float32) if args.y_global else None

	if x_array.ndim != 4:
		raise ValueError("X tensor must have shape (H, W, C, N)")

	num_samples = x_array.shape[3]
	if y_fields is not None and y_fields.shape[3] != num_samples:
		raise ValueError("X and Y_fields sample counts do not match.")
	if y_global is not None and y_global.shape[1] != num_samples:
		raise ValueError("X and Y_global sample counts do not match.")

	train_indices, val_indices = split_indices(num_samples, args.val_fraction, args.seed)
	x_mean, x_std = compute_image_stats(x_array, train_indices)
	x_norm = normalize_image(x_array, x_mean, x_std)

	y_field_mean = y_field_std = None
	if y_fields is not None:
		y_field_mean, y_field_std = compute_image_stats(y_fields, train_indices)
		y_fields = normalize_image(y_fields, y_field_mean, y_field_std)

	y_global_mean = y_global_std = None
	if y_global is not None:
		y_global_mean, y_global_std = compute_vector_stats(y_global, train_indices)
		y_global = normalize_vector(y_global, y_global_mean, y_global_std)

	train_dataset = ForwardSurrogateDataset(x_norm, train_indices, y_fields=y_fields, y_global=y_global)
	val_dataset = ForwardSurrogateDataset(x_norm, val_indices, y_fields=y_fields, y_global=y_global)
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	field_channel_names, global_channel_names = _load_channel_names(args.y_meta, task)
	model = ForwardSurrogateNet(
		in_channels=x_array.shape[2],
		out_field_channels=0 if y_fields is None else y_fields.shape[2],
		out_global_channels=0 if y_global is None else y_global.shape[0],
		base_channels=args.base_channels,
	).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	criterion = nn.MSELoss()

	history = []
	best_val_loss = float("inf")

	for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", unit="epoch"):
		print("Epoch {}/{}".format(epoch, args.epochs))
		train_metrics = _run_epoch(
			model,
			train_loader,
			optimizer,
			criterion,
			task,
			device,
			args.field_loss_weight,
			args.global_loss_weight,
			training=True,
		)
		val_metrics = _run_epoch(
			model,
			val_loader,
			optimizer,
			criterion,
			task,
			device,
			args.field_loss_weight,
			args.global_loss_weight,
			training=False,
		)

		record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
		history.append(record)
		print("  train:", train_metrics)
		print("  val  :", val_metrics)

		checkpoint = {
			"model_state": model.state_dict(),
			"config": {
				"task": task,
				"in_channels": x_array.shape[2],
				"out_field_channels": 0 if y_fields is None else y_fields.shape[2],
				"out_global_channels": 0 if y_global is None else y_global.shape[0],
				"base_channels": args.base_channels,
			},
			"stats": {
				"x_mean": x_mean.tolist(),
				"x_std": x_std.tolist(),
				"y_field_mean": None if y_field_mean is None else y_field_mean.tolist(),
				"y_field_std": None if y_field_std is None else y_field_std.tolist(),
				"y_global_mean": None if y_global_mean is None else y_global_mean.tolist(),
				"y_global_std": None if y_global_std is None else y_global_std.tolist(),
			},
			"meta": {
				"x_path": os.path.abspath(args.x),
				"y_fields_path": os.path.abspath(args.y_fields) if args.y_fields else None,
				"y_global_path": os.path.abspath(args.y_global) if args.y_global else None,
				"x_meta_path": os.path.abspath(args.x_meta) if args.x_meta else None,
				"y_meta_path": os.path.abspath(args.y_meta) if args.y_meta else None,
				"field_channel_names": field_channel_names,
				"global_channel_names": global_channel_names,
				"train_indices": train_indices.tolist(),
				"val_indices": val_indices.tolist(),
			},
			"epoch": epoch,
			"history": history,
		}
		torch.save(checkpoint, os.path.join(args.out_dir, "last_model.pt"))

		if val_metrics["total"] < best_val_loss:
			best_val_loss = val_metrics["total"]
			torch.save(checkpoint, os.path.join(args.out_dir, "best_model.pt"))

	with open(os.path.join(args.out_dir, "training_history.json"), "w") as fobj:
		json.dump(history, fobj, indent=2)

	run_summary = {
		"task": task,
		"device": str(device),
		"num_samples": int(num_samples),
		"train_count": int(len(train_indices)),
		"val_count": int(len(val_indices)),
		"best_val_loss": float(best_val_loss),
		"x_path": os.path.abspath(args.x),
		"y_fields_path": os.path.abspath(args.y_fields) if args.y_fields else None,
		"y_global_path": os.path.abspath(args.y_global) if args.y_global else None,
		"x_meta_path": os.path.abspath(args.x_meta) if args.x_meta else None,
		"y_meta_path": os.path.abspath(args.y_meta) if args.y_meta else None,
	}
	with open(os.path.join(args.out_dir, "run_summary.json"), "w") as fobj:
		json.dump(run_summary, fobj, indent=2)

	print("Saved training artifacts to:", args.out_dir)
	print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
	main()
