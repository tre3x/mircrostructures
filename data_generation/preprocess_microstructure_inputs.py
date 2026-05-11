#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build image-based X tensors from microstructure geometry.

This script supports both the legacy Abaqus ordering convention and the newer
FEniCSx batch outputs:

1. `--input <batch_summary.json or batch_folder>`:
   reads seed ordering from FEniCSx-generated sample folders
2. `--meta_y <legacy_y_meta.json>`:
   preserves the old Abaqus/ODB ordering flow
3. `--seeds 1,2,3`:
   explicit seed list

Output tensor shape:
    (img_size, img_size, n_channel, n_images)
"""

import argparse
import json
import os
import re

import numpy as np
from tqdm import tqdm

try:
	from data_generation.fiber_geometry import fibers_from_metadata, generate_fibers, rasterize_fibers
except ModuleNotFoundError:
	from fiber_geometry import fibers_from_metadata, generate_fibers, rasterize_fibers


def parse_args():
	parser = argparse.ArgumentParser(description="Generate microstructure X tensor for forward learning")
	parser.add_argument(
		"--input",
		type=str,
		default="",
		help="FEniCSx batch folder, batch_summary.json, or sample folder",
	)
	parser.add_argument(
		"--meta_y",
		type=str,
		default="",
		help="Legacy Y metadata JSON for ordering compatibility",
	)
	parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds. Example: 1,2,3")
	parser.add_argument("--L", type=float, default=150.0, help="RVE side length")
	parser.add_argument("--N_fibers", type=int, default=20, help="Number of fibers")
	parser.add_argument("--Vf", type=float, default=0.4, help="Fiber volume fraction")
	parser.add_argument(
		"--min_spacing_factor",
		type=float,
		default=1.05,
		help="Minimum center spacing factor (>1 means no touching)",
	)
	parser.add_argument(
		"--max_attempt_factor",
		type=float,
		default=5000.0,
		help="Max attempts factor for random placement",
	)
	parser.add_argument("--img_size", type=int, default=128, help="Output image size")
	parser.add_argument(
		"--channels",
		type=int,
		default=1,
		help="1: fiber mask only, 2: [matrix, fiber] one-hot",
	)
	parser.add_argument(
		"--flipud",
		type=int,
		default=1,
		help="Flip vertically for image convention",
	)
	parser.add_argument("--output", type=str, default="X_microstructure.npy", help="Output tensor file (.npy)")
	parser.add_argument("--meta_out", type=str, default="X_microstructure_meta.json", help="Output metadata JSON")
	return parser.parse_args()


def extract_seeds_from_legacy_meta(meta_path):
	if not os.path.exists(meta_path):
		raise FileNotFoundError("meta_y not found: {}".format(meta_path))

	with open(meta_path, "r") as fobj:
		meta = json.load(fobj)

	files_used = meta.get("files_used", [])
	seeds = []
	for path in files_used:
		base = os.path.basename(path)
		match = re.search(r"job_s(\d+)\.odb$", base)
		if not match:
			raise ValueError("Could not parse seed from file name: {}".format(base))
		seeds.append(int(match.group(1)))

	if len(seeds) == 0:
		raise ValueError("No files_used found in {}".format(meta_path))

	return seeds, files_used


def extract_seeds_from_fenicsx_input(input_path):
	input_path = os.path.abspath(input_path)

	if os.path.isfile(input_path) and os.path.basename(input_path) == "batch_summary.json":
		with open(input_path, "r") as fobj:
			summary = json.load(fobj)
		samples = summary.get("samples_written", [])
	elif os.path.isdir(input_path) and os.path.exists(os.path.join(input_path, "sample_data.npz")):
		samples = [input_path]
	elif os.path.isdir(input_path):
		samples = [
			os.path.join(input_path, name)
			for name in sorted(os.listdir(input_path))
			if name.startswith("job_s") and os.path.isdir(os.path.join(input_path, name))
		]
	else:
		raise ValueError("Could not resolve FEniCSx input from {}".format(input_path))

	seeds = []
	for sample in samples:
		match = re.search(r"job_s(\d+)$", os.path.basename(sample))
		if not match:
			raise ValueError("Could not parse seed from sample path: {}".format(sample))
		seeds.append(int(match.group(1)))

	return seeds, [os.path.abspath(sample) for sample in samples]


def resolve_seed_source(args):
	if args.seeds.strip():
		seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
		return [{"seed": seed, "source": "seed_{}".format(seed), "sample_dir": None} for seed in seeds], "explicit_seeds"
	if args.input.strip():
		seeds, source_files = extract_seeds_from_fenicsx_input(args.input)
		return [
			{"seed": seed, "source": sample_dir, "sample_dir": sample_dir}
			for seed, sample_dir in zip(seeds, source_files)
		], "fenicsx_input"
	if args.meta_y.strip():
		seeds, source_files = extract_seeds_from_legacy_meta(args.meta_y)
		return [
			{"seed": seed, "source": source_file, "sample_dir": None}
			for seed, source_file in zip(seeds, source_files)
		], "legacy_meta_y"
	raise ValueError("Provide one of --seeds, --input, or --meta_y.")


def resolve_fibers(entry, args):
	if entry["sample_dir"] is not None:
		metadata_path = os.path.join(entry["sample_dir"], "metadata.json")
		if os.path.exists(metadata_path):
			with open(metadata_path, "r") as fobj:
				return fibers_from_metadata(json.load(fobj))

	return generate_fibers(
		L_value=args.L,
		n_fibers_value=args.N_fibers,
		vf_value=args.Vf,
		seed_value=entry["seed"],
		min_spacing_factor_value=args.min_spacing_factor,
		max_attempt_factor_value=args.max_attempt_factor,
	)


def main():
	args = parse_args()

	if args.channels not in (1, 2):
		raise ValueError("--channels must be 1 or 2")

	entries, source_type = resolve_seed_source(args)

	print("Generating X for {} microstructures".format(len(entries)))
	print(
		"Parameters: L={}, N_fibers={}, Vf={}, min_spacing_factor={}".format(
			args.L, args.N_fibers, args.Vf, args.min_spacing_factor
		)
	)

	images = []
	failed = []

	for entry in tqdm(entries, desc="Building X microstructure", unit="sample"):
		try:
			fibers = resolve_fibers(entry, args)
			binary = rasterize_fibers(args.L, fibers, args.img_size, flipud=args.flipud)

			if args.channels == 1:
				image = binary[:, :, None]
			else:
				image = np.stack([1.0 - binary, binary], axis=2)

			images.append(image.astype(np.float32))

		except Exception as err:
			failed.append({"seed": entry["seed"], "error": str(err)})
			tqdm.write("  FAILED seed {}: {}".format(entry["seed"], err))

	if len(images) == 0:
		raise RuntimeError("No images generated.")

	x_tensor = np.stack(images, axis=3)
	np.save(args.output, x_tensor)

	meta_out = {
		"shape": list(x_tensor.shape),
		"img_size": int(args.img_size),
		"n_channel": int(args.channels),
		"channel_names": ["fiber"] if args.channels == 1 else ["matrix", "fiber"],
		"L": float(args.L),
		"N_fibers": int(args.N_fibers),
		"Vf": float(args.Vf),
		"min_spacing_factor": float(args.min_spacing_factor),
		"max_attempt_factor": float(args.max_attempt_factor),
		"flipud": int(args.flipud),
		"seeds_used": [entry["seed"] for entry in entries],
		"source_type": source_type,
		"source_files": [entry["source"] for entry in entries],
		"failed": failed,
	}

	with open(args.meta_out, "w") as fobj:
		json.dump(meta_out, fobj, indent=2)

	print("\nSaved X tensor: {}".format(args.output))
	print("Tensor shape:", x_tensor.shape)
	print("Saved metadata:", args.meta_out)
	print("Failed count:", len(failed))


if __name__ == "__main__":
	main()
