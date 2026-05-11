#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build material-field X tensors from microstructure geometry.

This script is the FEniCSx-aware counterpart to the older Abaqus preprocessing
flow. It rasterizes each sample to an image grid and emits either:

- binary channels: [fiber] or [matrix, fiber]
- material channels: [E_field, nu_field]
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
	parser = argparse.ArgumentParser(description="Generate material-field X tensor for forward learning")
	parser.add_argument("--input", type=str, default="", help="FEniCSx batch folder, batch_summary.json, or sample folder")
	parser.add_argument("--meta_y", type=str, default="", help="Legacy Y metadata JSON for ordering compatibility")
	parser.add_argument("--seeds", type=str, default="", help="Comma-separated seeds. Example: 1,2,3")
	parser.add_argument("--L", type=float, default=150.0, help="RVE side length")
	parser.add_argument("--N_fibers", type=int, default=20, help="Number of fibers")
	parser.add_argument("--Vf", type=float, default=0.4, help="Fiber volume fraction")
	parser.add_argument("--min_spacing_factor", type=float, default=1.05, help="Minimum center spacing factor")
	parser.add_argument("--max_attempt_factor", type=float, default=5000.0, help="Max attempts factor for random placement")
	parser.add_argument("--img_size", type=int, default=128, help="Output image size")
	parser.add_argument(
		"--representation",
		type=str,
		default="material",
		choices=["binary", "material"],
		help="binary: mask channels, material: [E_field, nu_field] channels",
	)
	parser.add_argument(
		"--channels",
		type=int,
		default=1,
		help="Used only when --representation=binary. 1: fiber only, 2: [matrix, fiber]",
	)
	parser.add_argument(
		"--material_source",
		type=str,
		default="fenicsx",
		choices=["manual", "fenicsx", "inp", "odb"],
		help="How to source per-phase material values",
	)
	parser.add_argument("--inp_root", type=str, default="../abaqus_input_files", help="Folder containing Abaqus .inp files")
	parser.add_argument("--E_matrix", type=float, default=2500.0, help="Matrix Young's modulus")
	parser.add_argument("--E_fiber", type=float, default=70000.0, help="Fiber Young's modulus")
	parser.add_argument("--nu_matrix", type=float, default=0.35, help="Matrix Poisson ratio")
	parser.add_argument("--nu_fiber", type=float, default=0.22, help="Fiber Poisson ratio")
	parser.add_argument("--flipud", type=int, default=1, help="Flip vertically for image convention")
	parser.add_argument("--output", type=str, default="X_material_fields.npy", help="Output tensor file (.npy)")
	parser.add_argument("--meta_out", type=str, default="X_material_fields_meta.json", help="Output metadata JSON")
	return parser.parse_args()


def _extract_elastic_table_from_odb(odb_path):
	try:
		from odbAccess import openOdb
	except Exception as err:
		raise RuntimeError(
			"Could not import odbAccess. Run with Abaqus Python for --material_source odb. Error: {}".format(err)
		)

	odb = openOdb(odb_path, readOnly=True)
	try:
		materials = getattr(odb, "materials", {})
		if not materials:
			raise RuntimeError("No materials found in ODB: {}".format(odb_path))

		rows = []
		for mat_name in materials.keys():
			mat = materials[mat_name]
			elastic = getattr(mat, "elastic", None)
			table = getattr(elastic, "table", None) if elastic is not None else None
			if table is None or len(table) == 0 or len(table[0]) < 2:
				continue
			rows.append({"name": str(mat_name), "E": float(table[0][0]), "nu": float(table[0][1])})

		if len(rows) == 0:
			raise RuntimeError("No linear elastic (E, nu) table found in ODB materials: {}".format(odb_path))

		return rows
	finally:
		odb.close()


def _extract_elastic_table_from_inp(inp_path):
	if not os.path.exists(inp_path):
		raise RuntimeError("INP file not found: {}".format(inp_path))

	rows = []
	current_material = None
	expect_elastic_data = False

	with open(inp_path, "r") as fobj:
		for raw_line in fobj:
			line = raw_line.strip()
			if not line or line.startswith("**"):
				continue
			low = line.lower()
			if low.startswith("*material"):
				current_material = None
				expect_elastic_data = False
				match = re.search(r"name\s*=\s*([^,]+)", line, flags=re.IGNORECASE)
				if match:
					current_material = match.group(1).strip()
				continue
			if low.startswith("*elastic"):
				expect_elastic_data = True
				continue
			if line.startswith("*"):
				expect_elastic_data = False
				continue
			if expect_elastic_data and current_material is not None:
				parts = [item.strip() for item in line.split(",") if item.strip()]
				if len(parts) >= 2:
					rows.append({"name": current_material, "E": float(parts[0]), "nu": float(parts[1])})
				expect_elastic_data = False

	if len(rows) == 0:
		raise RuntimeError("No *Material/*Elastic (E, nu) pairs found in INP: {}".format(inp_path))
	return rows


def _pick_matrix_fiber_from_rows(rows):
	def _is_matrix(name):
		low = name.lower()
		return ("matrix" in low) or (low == "mat")

	def _is_fiber(name):
		low = name.lower()
		return ("fiber" in low) or ("fibre" in low) or ("reinf" in low)

	matrix = None
	fiber = None
	for row in rows:
		if matrix is None and _is_matrix(row["name"]):
			matrix = row
		if fiber is None and _is_fiber(row["name"]):
			fiber = row

	if matrix is not None and fiber is not None:
		return matrix["E"], matrix["nu"], fiber["E"], fiber["nu"]

	rows_sorted = sorted(rows, key=lambda item: item["E"])
	return rows_sorted[0]["E"], rows_sorted[0]["nu"], rows_sorted[-1]["E"], rows_sorted[-1]["nu"]


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

	return seeds, files_used


def extract_samples_from_fenicsx_input(input_path):
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

	result = []
	for sample in samples:
		match = re.search(r"job_s(\d+)$", os.path.basename(sample))
		if not match:
			raise ValueError("Could not parse seed from sample path: {}".format(sample))
		result.append({"seed": int(match.group(1)), "sample_dir": os.path.abspath(sample)})
	return result


def resolve_source_entries(args):
	if args.seeds.strip():
		seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
		return [{"seed": seed, "source": "seed_{}".format(seed), "sample_dir": None} for seed in seeds], "explicit_seeds"

	if args.input.strip():
		samples = extract_samples_from_fenicsx_input(args.input)
		return [{"seed": item["seed"], "source": item["sample_dir"], "sample_dir": item["sample_dir"]} for item in samples], "fenicsx_input"

	if args.meta_y.strip():
		seeds, files_used = extract_seeds_from_legacy_meta(args.meta_y)
		return [
			{"seed": seed, "source": source_file, "sample_dir": None}
			for seed, source_file in zip(seeds, files_used)
		], "legacy_meta_y"

	raise ValueError("Provide one of --seeds, --input, or --meta_y.")


def resolve_material_values(entry, args):
	if args.material_source == "manual":
		return args.E_matrix, args.nu_matrix, args.E_fiber, args.nu_fiber

	if args.material_source == "fenicsx":
		if entry["sample_dir"] is None:
			raise RuntimeError("--material_source fenicsx requires --input pointing to FEniCSx outputs.")
		metadata_path = os.path.join(entry["sample_dir"], "metadata.json")
		if not os.path.exists(metadata_path):
			raise RuntimeError("metadata.json not found for sample {}".format(entry["sample_dir"]))
		with open(metadata_path, "r") as fobj:
			meta = json.load(fobj)
		return meta["matrix_E"], meta["matrix_nu"], meta["fiber_E"], meta["fiber_nu"]

	if args.material_source == "inp":
		source_entry = entry["source"]
		inp_path = source_entry
		if source_entry.lower().endswith(".odb"):
			base = os.path.splitext(os.path.basename(source_entry))[0]
			inp_path = os.path.join(args.inp_root, base + ".inp")
		elif not source_entry.lower().endswith(".inp"):
			inp_path = os.path.join(args.inp_root, "job_s{}.inp".format(entry["seed"]))
		rows = _extract_elastic_table_from_inp(inp_path)
		return _pick_matrix_fiber_from_rows(rows)

	if args.material_source == "odb":
		rows = _extract_elastic_table_from_odb(entry["source"])
		return _pick_matrix_fiber_from_rows(rows)

	raise ValueError("Unsupported material_source: {}".format(args.material_source))


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

	if args.representation == "binary" and args.channels not in (1, 2):
		raise ValueError("--channels must be 1 or 2")

	entries, source_type = resolve_source_entries(args)
	print("Generating X for {} microstructures".format(len(entries)))

	images = []
	failed = []
	material_records = []

	for entry in tqdm(entries, desc="Building X material fields", unit="sample"):
		try:
			fibers = resolve_fibers(entry, args)
			binary = rasterize_fibers(args.L, fibers, args.img_size, flipud=args.flipud)

			if args.representation == "binary":
				if args.channels == 1:
					image = binary[:, :, None]
				else:
					image = np.stack([1.0 - binary, binary], axis=2)
			else:
				e_matrix, nu_matrix, e_fiber, nu_fiber = resolve_material_values(entry, args)
				e_field = e_matrix * (1.0 - binary) + e_fiber * binary
				nu_field = nu_matrix * (1.0 - binary) + nu_fiber * binary
				image = np.stack([e_field, nu_field], axis=2)
				material_records.append(
					{
						"seed": int(entry["seed"]),
						"source": entry["source"],
						"material_source_used": args.material_source,
						"E_matrix": float(e_matrix),
						"nu_matrix": float(nu_matrix),
						"E_fiber": float(e_fiber),
						"nu_fiber": float(nu_fiber),
					}
				)

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
		"representation": args.representation,
		"material_source": args.material_source,
		"n_channel": int(x_tensor.shape[2]),
		"channel_names": (
			["fiber"]
			if (args.representation == "binary" and args.channels == 1)
			else ["matrix", "fiber"]
			if args.representation == "binary"
			else ["E_field", "nu_field"]
		),
		"L": float(args.L),
		"N_fibers": int(args.N_fibers),
		"Vf": float(args.Vf),
		"min_spacing_factor": float(args.min_spacing_factor),
		"max_attempt_factor": float(args.max_attempt_factor),
		"E_matrix": float(args.E_matrix),
		"E_fiber": float(args.E_fiber),
		"nu_matrix": float(args.nu_matrix),
		"nu_fiber": float(args.nu_fiber),
		"flipud": int(args.flipud),
		"source_type": source_type,
		"seeds_used": [entry["seed"] for entry in entries],
		"source_files": [entry["source"] for entry in entries],
		"material_records": material_records,
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
