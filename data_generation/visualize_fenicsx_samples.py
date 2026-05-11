#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os

_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_CACHE_ROOT, "matplotlib"))
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Ellipse
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser(description="Visualize FEniCSx microstructure test results")
	parser.add_argument(
		"--input",
		type=str,
		required=True,
		help="Sample folder, batch folder, or batch_summary.json",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Optional seed filter when input is a batch folder or summary file",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="",
		help="Optional folder for PNG outputs. Defaults to each sample folder.",
	)
	parser.add_argument(
		"--show",
		type=int,
		default=0,
		help="Show figures interactively (1) or save only (0)",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=180,
		help="Output image DPI",
	)
	parser.add_argument(
		"--deformation_scale",
		type=float,
		default=1.0,
		help="Scale factor for deformed-mesh overlay",
	)
	return parser.parse_args()


def load_json(path):
	with open(path, "r") as fobj:
		return json.load(fobj)


def resolve_sample_dirs(input_path, seed_value=None):
	input_path = os.path.abspath(input_path)

	if os.path.isfile(input_path) and os.path.basename(input_path) == "batch_summary.json":
		summary = load_json(input_path)
		sample_dirs = summary.get("samples_written", [])
		if seed_value is not None:
			sample_dirs = [path for path in sample_dirs if path.endswith("job_s{}".format(seed_value))]
		return [os.path.abspath(path) for path in sample_dirs]

	if os.path.isdir(input_path):
		if os.path.exists(os.path.join(input_path, "sample_data.npz")):
			return [input_path]

		sample_dirs = []
		for name in sorted(os.listdir(input_path)):
			full_path = os.path.join(input_path, name)
			if os.path.isdir(full_path) and name.startswith("job_s") and os.path.exists(
				os.path.join(full_path, "sample_data.npz")
			):
				if seed_value is None or name == "job_s{}".format(seed_value):
					sample_dirs.append(full_path)
		return sample_dirs

	raise ValueError("Could not resolve samples from --input={}".format(input_path))


def displacement_magnitude(ux, uy):
	return np.sqrt(ux ** 2 + uy ** 2)


def plot_cell_field(ax, triangulation, values, title, cmap):
	plot = ax.tripcolor(triangulation, facecolors=values, shading="flat", cmap=cmap)
	ax.set_title(title)
	ax.set_aspect("equal")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	return plot


def plot_nodal_field(ax, triangulation, values, title, cmap):
	plot = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmap)
	ax.set_title(title)
	ax.set_aspect("equal")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	return plot


def build_summary_figure(sample_dir, output_dir, dpi, deformation_scale, show_figure):
	metadata_path = os.path.join(sample_dir, "metadata.json")
	data_path = os.path.join(sample_dir, "sample_data.npz")

	if not os.path.exists(metadata_path):
		raise FileNotFoundError("Missing metadata.json in {}".format(sample_dir))
	if not os.path.exists(data_path):
		raise FileNotFoundError("Missing sample_data.npz in {}".format(sample_dir))

	metadata = load_json(metadata_path)
	data = np.load(data_path)

	points = data["points"]
	cells = data["cells"]
	cell_tags = data["cell_tags"]
	ux = data["ux"]
	uy = data["uy"]
	disp_mag = displacement_magnitude(ux, uy)

	triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles=cells)
	deformed_points = points + deformation_scale * np.column_stack([ux, uy])
	deformed_triangulation = mtri.Triangulation(
		deformed_points[:, 0], deformed_points[:, 1], triangles=cells
	)

	fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
	axes = axes.ravel()

	phase_plot = plot_cell_field(
		axes[0],
		triangulation,
		(cell_tags == 2).astype(np.float64),
		"Phase Map",
		"viridis",
	)
	fiber_centers = metadata.get("fiber_centers", [])
	fiber_axes = metadata.get("fiber_axes")
	fiber_angles = metadata.get("fiber_angles_deg")
	if fiber_axes is not None:
		if fiber_angles is None:
			fiber_angles = [0.0] * len(fiber_axes)
		for center, axis_pair, angle_deg in zip(fiber_centers, fiber_axes, fiber_angles):
			axes[0].add_patch(
				Ellipse(
					xy=(float(center[0]), float(center[1])),
					width=2.0 * float(axis_pair[0]),
					height=2.0 * float(axis_pair[1]),
					angle=float(angle_deg),
					fill=False,
					edgecolor="white",
					linewidth=0.5,
				)
			)
	else:
		for (xc, yc) in fiber_centers:
			axes[0].plot(xc, yc, "wo", markersize=2.5, markeredgecolor="k", markeredgewidth=0.4)
	fig.colorbar(phase_plot, ax=axes[0], shrink=0.85)

	disp_plot = plot_nodal_field(axes[1], triangulation, disp_mag, "Displacement Magnitude", "plasma")
	fig.colorbar(disp_plot, ax=axes[1], shrink=0.85)

	sxx_plot = plot_cell_field(axes[2], triangulation, data["sigma_xx"], "Sigma_xx", "coolwarm")
	fig.colorbar(sxx_plot, ax=axes[2], shrink=0.85)

	syy_plot = plot_cell_field(axes[3], triangulation, data["sigma_yy"], "Sigma_yy", "coolwarm")
	fig.colorbar(syy_plot, ax=axes[3], shrink=0.85)

	sxy_plot = plot_cell_field(axes[4], triangulation, data["sigma_xy"], "Sigma_xy", "coolwarm")
	fig.colorbar(sxy_plot, ax=axes[4], shrink=0.85)

	axes[5].triplot(triangulation, color="0.75", linewidth=0.25)
	axes[5].triplot(deformed_triangulation, color="tab:red", linewidth=0.35)
	axes[5].set_title("Mesh / Deformed Mesh")
	axes[5].set_aspect("equal")
	axes[5].set_xlabel("x")
	axes[5].set_ylabel("y")

	fig.suptitle(
		"Seed {seed} | {atype} | cells={cells} | avg_sigma_xx={sxx:.3e}".format(
			seed=metadata["seed"],
			atype=metadata["analysis_type"],
			cells=metadata["mesh_num_cells"],
			sxx=metadata["avg_sigma_xx"],
		),
		fontsize=14,
	)

	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
		png_path = os.path.join(output_dir, "job_s{}_summary.png".format(metadata["seed"]))
	else:
		png_path = os.path.join(sample_dir, "summary.png")

	fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
	if show_figure:
		plt.show()
	else:
		plt.close(fig)

	return png_path


def main():
	args = parse_args()
	sample_dirs = resolve_sample_dirs(args.input, args.seed)

	if len(sample_dirs) == 0:
		raise ValueError("No samples found for visualization.")

	print("Found {} sample(s)".format(len(sample_dirs)))
	for sample_dir in tqdm(sample_dirs, desc="Rendering figures", unit="sample"):
		png_path = build_summary_figure(
			sample_dir=sample_dir,
			output_dir=args.output_dir,
			dpi=args.dpi,
			deformation_scale=args.deformation_scale,
			show_figure=bool(args.show),
		)
		tqdm.write("Saved: {}".format(png_path))


if __name__ == "__main__":
	main()
