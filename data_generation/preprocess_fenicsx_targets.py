#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build learning-ready Y tensors from FEniCSx-generated microstructure outputs.

This script reads per-sample folders produced by `data_driven_ml_abaqus.py`
after the FEniCSx migration. It supports:

1. Field targets:
   Rasterized image-grid tensors with shape (H, W, C, N)
2. Global targets:
   Sample-level response vectors with shape (C, N)

Supported field channels:
   - stress: sigma_xx, sigma_yy, sigma_xy
   - strain: epsilon_xx, epsilon_yy, epsilon_xy
   - stress_strain: all six channels above
   - displacement: ux, uy
   - all: displacement + strain + stress

Supported global channels:
   - avg_epsilon_xx, avg_epsilon_yy, avg_epsilon_xy
   - avg_sigma_xx, avg_sigma_yy, avg_sigma_xy
"""

import argparse
import json
import os
import re

import numpy as np
from tqdm import tqdm


FIELD_CHANNEL_MAP = {
	"stress": ["sigma_xx", "sigma_yy", "sigma_xy"],
	"strain": ["epsilon_xx", "epsilon_yy", "epsilon_xy"],
	"stress_strain": [
		"epsilon_xx",
		"epsilon_yy",
		"epsilon_xy",
		"sigma_xx",
		"sigma_yy",
		"sigma_xy",
	],
	"displacement": ["ux", "uy"],
	"all": [
		"ux",
		"uy",
		"epsilon_xx",
		"epsilon_yy",
		"epsilon_xy",
		"sigma_xx",
		"sigma_yy",
		"sigma_xy",
	],
}

GLOBAL_CHANNELS = [
	"avg_epsilon_xx",
	"avg_epsilon_yy",
	"avg_epsilon_xy",
	"avg_sigma_xx",
	"avg_sigma_yy",
	"avg_sigma_xy",
]


def parse_args():
	parser = argparse.ArgumentParser(description="Generate Y tensors from FEniCSx sample outputs")
	parser.add_argument(
		"--input",
		type=str,
		required=True,
		help="Batch folder, batch_summary.json, or single sample folder",
	)
	parser.add_argument(
		"--seeds",
		type=str,
		default="",
		help="Optional comma-separated seed filter",
	)
	parser.add_argument(
		"--representation",
		type=str,
		default="both",
		choices=["field", "global", "both"],
		help="Which targets to generate",
	)
	parser.add_argument(
		"--field_set",
		type=str,
		default="stress_strain",
		choices=sorted(FIELD_CHANNEL_MAP.keys()),
		help="Field channels to rasterize when representation includes field",
	)
	parser.add_argument("--img_size", type=int, default=128, help="Output image size for field tensors")
	parser.add_argument(
		"--flipud",
		type=int,
		default=1,
		help="Flip vertically to match the X preprocessing convention",
	)
	parser.add_argument(
		"--field_output",
		type=str,
		default="Y_fenicsx_fields.npy",
		help="Output .npy file for field targets",
	)
	parser.add_argument(
		"--global_output",
		type=str,
		default="Y_fenicsx_global.npy",
		help="Output .npy file for global targets",
	)
	parser.add_argument(
		"--meta_out",
		type=str,
		default="Y_fenicsx_meta.json",
		help="Output metadata JSON",
	)
	return parser.parse_args()


def _load_json(path):
	with open(path, "r") as fobj:
		return json.load(fobj)


def _parse_seed_list(seeds_arg):
	if not seeds_arg.strip():
		return None
	return {int(item.strip()) for item in seeds_arg.split(",") if item.strip()}


def _extract_seed_from_sample_dir(path):
	match = re.search(r"job_s(\d+)$", os.path.basename(path))
	if match is None:
		raise ValueError("Could not parse seed from sample directory: {}".format(path))
	return int(match.group(1))


def resolve_sample_dirs(input_path, allowed_seeds=None):
	input_path = os.path.abspath(input_path)

	if os.path.isfile(input_path) and os.path.basename(input_path) == "batch_summary.json":
		summary = _load_json(input_path)
		sample_dirs = [os.path.abspath(path) for path in summary.get("samples_written", [])]
	elif os.path.isdir(input_path) and os.path.exists(os.path.join(input_path, "sample_data.npz")):
		sample_dirs = [input_path]
	elif os.path.isdir(input_path):
		sample_dirs = []
		for name in sorted(os.listdir(input_path)):
			full_path = os.path.join(input_path, name)
			if os.path.isdir(full_path) and name.startswith("job_s") and os.path.exists(
				os.path.join(full_path, "sample_data.npz")
			):
				sample_dirs.append(os.path.abspath(full_path))
	else:
		raise ValueError("Could not resolve samples from --input={}".format(input_path))

	if allowed_seeds is not None:
		sample_dirs = [path for path in sample_dirs if _extract_seed_from_sample_dir(path) in allowed_seeds]

	return sample_dirs


def _grid_coordinates(L_value, img_size):
	x_coords = np.linspace(0.0, L_value, img_size)
	y_coords = np.linspace(0.0, L_value, img_size)
	return x_coords, y_coords


def _pixel_index_from_coordinates(coords, L_value, img_size):
	scale = float(img_size) / float(L_value)
	indices = np.floor(np.asarray(coords, dtype=np.float64) * scale).astype(np.int64)
	return np.clip(indices, 0, int(img_size) - 1)


def _triangle_bbox_indices(triangle_points, L_value, img_size):
	min_x = max(0.0, np.min(triangle_points[:, 0]))
	max_x = min(L_value, np.max(triangle_points[:, 0]))
	min_y = max(0.0, np.min(triangle_points[:, 1]))
	max_y = min(L_value, np.max(triangle_points[:, 1]))

	scale = float(img_size - 1) / float(L_value)
	i0 = max(0, int(np.floor(min_x * scale)) - 1)
	i1 = min(img_size - 1, int(np.ceil(max_x * scale)) + 1)
	j0 = max(0, int(np.floor(min_y * scale)) - 1)
	j1 = min(img_size - 1, int(np.ceil(max_y * scale)) + 1)
	return i0, i1, j0, j1


def rasterize_fields(points, cells, field_specs, img_size, L_value, flipud):
	x_coords, y_coords = _grid_coordinates(L_value, img_size)
	image = np.zeros((img_size, img_size, len(field_specs)), dtype=np.float32)

	for cell_index, cell_nodes in enumerate(cells):
		triangle_points = points[cell_nodes]
		x1, y1 = triangle_points[0]
		x2, y2 = triangle_points[1]
		x3, y3 = triangle_points[2]

		denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
		if abs(denom) < 1.0e-14:
			continue

		i0, i1, j0, j1 = _triangle_bbox_indices(triangle_points, L_value, img_size)
		x_patch = x_coords[i0 : i1 + 1]
		y_patch = y_coords[j0 : j1 + 1]
		x_grid, y_grid = np.meshgrid(x_patch, y_patch)

		w1 = ((y2 - y3) * (x_grid - x3) + (x3 - x2) * (y_grid - y3)) / denom
		w2 = ((y3 - y1) * (x_grid - x3) + (x1 - x3) * (y_grid - y3)) / denom
		w3 = 1.0 - w1 - w2
		inside = (w1 >= -1.0e-9) & (w2 >= -1.0e-9) & (w3 >= -1.0e-9)
		if not np.any(inside):
			continue

		for channel_index, spec in enumerate(field_specs):
			channel_view = image[j0 : j1 + 1, i0 : i1 + 1, channel_index]
			if spec["kind"] == "cell":
				channel_view[inside] = spec["values"][cell_index]
			else:
				node_values = spec["values"][cell_nodes]
				channel_view[inside] = (
					w1[inside] * node_values[0] + w2[inside] * node_values[1] + w3[inside] * node_values[2]
				)
			image[j0 : j1 + 1, i0 : i1 + 1, channel_index] = channel_view

	if int(flipud) == 1:
		image = np.flipud(image)

	return image


def rasterize_cell_fields_from_centroids(cell_centroids, cell_areas, field_specs, img_size, L_value, flipud):
	if len(field_specs) == 0:
		return np.zeros((img_size, img_size, 0), dtype=np.float32)

	x_idx = _pixel_index_from_coordinates(cell_centroids[:, 0], L_value, img_size)
	y_idx = _pixel_index_from_coordinates(cell_centroids[:, 1], L_value, img_size)
	pixel_ids = y_idx * int(img_size) + x_idx
	num_pixels = int(img_size) * int(img_size)

	area_weights = np.asarray(cell_areas, dtype=np.float64)
	pixel_area = np.bincount(pixel_ids, weights=area_weights, minlength=num_pixels)
	image = np.zeros((num_pixels, len(field_specs)), dtype=np.float32)

	nonzero_mask = pixel_area > 0.0
	for channel_index, spec in enumerate(field_specs):
		if spec["kind"] != "cell":
			raise ValueError("rasterize_cell_fields_from_centroids only supports cellwise fields")
		weighted_values = area_weights * np.asarray(spec["values"], dtype=np.float64)
		pixel_weighted_sum = np.bincount(pixel_ids, weights=weighted_values, minlength=num_pixels)
		channel = np.zeros(num_pixels, dtype=np.float32)
		channel[nonzero_mask] = (pixel_weighted_sum[nonzero_mask] / pixel_area[nonzero_mask]).astype(np.float32)
		image[:, channel_index] = channel

	image = image.reshape((int(img_size), int(img_size), len(field_specs)))
	if int(flipud) == 1:
		image = np.flipud(image)
	return image


def load_sample(sample_dir):
	metadata_path = os.path.join(sample_dir, "metadata.json")
	data_path = os.path.join(sample_dir, "sample_data.npz")

	if not os.path.exists(metadata_path):
		raise FileNotFoundError("metadata.json not found in {}".format(sample_dir))
	if not os.path.exists(data_path):
		raise FileNotFoundError("sample_data.npz not found in {}".format(sample_dir))

	return _load_json(metadata_path), np.load(data_path)


def build_field_specs(data, field_names):
	field_specs = []
	for field_name in field_names:
		if field_name in ("ux", "uy"):
			field_specs.append({"name": field_name, "kind": "nodal", "values": np.asarray(data[field_name])})
		else:
			field_specs.append({"name": field_name, "kind": "cell", "values": np.asarray(data[field_name])})
	return field_specs


def main():
	args = parse_args()
	allowed_seeds = _parse_seed_list(args.seeds)
	sample_dirs = resolve_sample_dirs(args.input, allowed_seeds=allowed_seeds)

	if len(sample_dirs) == 0:
		raise ValueError("No FEniCSx samples found for preprocessing.")

	print("Preparing Y from {} sample(s)".format(len(sample_dirs)))

	field_names = FIELD_CHANNEL_MAP[args.field_set]
	field_images = []
	global_vectors = []
	metadata_records = []
	failed = []

	for sample_dir in tqdm(sample_dirs, desc="Building Y targets", unit="sample"):
		try:
			metadata, data = load_sample(sample_dir)

			if args.representation in ("field", "both"):
				field_specs = build_field_specs(data, field_names)
				if all(spec["kind"] == "cell" for spec in field_specs):
					field_image = rasterize_cell_fields_from_centroids(
						cell_centroids=np.asarray(data["cell_centroids"], dtype=np.float64),
						cell_areas=np.asarray(data["cell_areas"], dtype=np.float64),
						field_specs=field_specs,
						img_size=args.img_size,
						L_value=float(metadata["L"]),
						flipud=args.flipud,
					)
				else:
					field_image = rasterize_fields(
						points=np.asarray(data["points"], dtype=np.float64),
						cells=np.asarray(data["cells"], dtype=np.int32),
						field_specs=field_specs,
						img_size=args.img_size,
						L_value=float(metadata["L"]),
						flipud=args.flipud,
					)
				field_images.append(field_image.astype(np.float32))

			if args.representation in ("global", "both"):
				global_vector = np.asarray([metadata[channel] for channel in GLOBAL_CHANNELS], dtype=np.float32)
				global_vectors.append(global_vector)

			metadata_records.append(
				{
					"seed": int(metadata["seed"]),
					"sample_dir": os.path.abspath(sample_dir),
					"npz_path": os.path.abspath(os.path.join(sample_dir, "sample_data.npz")),
				}
			)

		except Exception as err:
			failed.append({"sample_dir": sample_dir, "error": str(err)})
			tqdm.write("  FAILED {}: {}".format(sample_dir, err))

	if args.representation in ("field", "both"):
		if len(field_images) == 0:
			raise RuntimeError("No field targets were generated.")
		y_fields = np.stack(field_images, axis=3)
		np.save(args.field_output, y_fields)
		print("Saved field tensor:", args.field_output, "| shape =", y_fields.shape)
	else:
		y_fields = None

	if args.representation in ("global", "both"):
		if len(global_vectors) == 0:
			raise RuntimeError("No global targets were generated.")
		y_global = np.stack(global_vectors, axis=1)
		np.save(args.global_output, y_global)
		print("Saved global tensor:", args.global_output, "| shape =", y_global.shape)
	else:
		y_global = None

	meta_out = {
		"input": os.path.abspath(args.input),
		"representation": args.representation,
		"img_size": int(args.img_size),
		"flipud": int(args.flipud),
		"field_set": args.field_set,
		"field_channels": field_names if args.representation in ("field", "both") else [],
		"global_channels": GLOBAL_CHANNELS if args.representation in ("global", "both") else [],
		"field_output": os.path.abspath(args.field_output) if y_fields is not None else None,
		"global_output": os.path.abspath(args.global_output) if y_global is not None else None,
		"field_shape": list(y_fields.shape) if y_fields is not None else None,
		"global_shape": list(y_global.shape) if y_global is not None else None,
		"samples_used": metadata_records,
		"failed": failed,
	}

	with open(args.meta_out, "w") as fobj:
		json.dump(meta_out, fobj, indent=2)

	print("Saved metadata:", args.meta_out)
	print("Failed count:", len(failed))


if __name__ == "__main__":
	main()
