#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Random UD composite RVE generator + FEniCSx solver.

This script replaces the original Abaqus-only workflow with an open-source
pipeline based on Gmsh + FEniCSx. For each requested seed it:

1. Samples a random fiber layout.
2. Builds a 2D tagged mesh with Gmsh.
3. Solves a linear-elastic plane stress/plane strain problem in FEniCSx.
4. Writes open outputs per sample:
   - ``solution.xdmf`` for mesh + displacement visualization
   - ``sample_data.npz`` for training-data style arrays
   - ``metadata.json`` for sample-level provenance

The key geometry/material knobs remain aligned with the original script so
downstream datasets can still be generated from the same parameter sweeps.
"""

import json
import math
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
from tqdm import tqdm


_CACHE_ROOT = os.path.join(os.getcwd(), ".cache")
os.environ.setdefault("XDG_CACHE_HOME", _CACHE_ROOT)
os.environ.setdefault("OMPI_MCA_btl", "self")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import gmsh
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio

try:
	from data_generation.fiber_geometry import (
		compute_fiber_axes,
		equivalent_radius,
		fibers_to_metadata_fields,
		generate_fibers,
		normalize_fiber_shape,
	)
except ModuleNotFoundError:
	from fiber_geometry import (
		compute_fiber_axes,
		equivalent_radius,
		fibers_to_metadata_fields,
		generate_fibers,
		normalize_fiber_shape,
	)


# -----------------------------------------------------
# USER PARAMETERS
# -----------------------------------------------------
L = 150.0
N_fibers = 20
Vf = 0.4
seed = 1
min_spacing_factor = 1.05
max_attempt_factor = 5000
fiber_shape = "circle"
fiber_aspect_ratio = 1.0
fiber_angle_deg = 0.0
random_fiber_angle = False

model_name = "Model-1"
part_name = "RVE_Part"

# Analysis controls
thickness = 1.0
analysis_type = "plane_strain"   # plane_strain | plane_stress

# Material properties
matrix_E = 3.2e9
matrix_nu = 0.35
fiber_E = 87.0e9
fiber_nu = 0.20

# Legacy step controls kept for CLI compatibility. They are not used by the
# direct linear solve in FEniCSx.
applied_strain_x = 0.001
initial_increment = 1.0
max_increment = 1.0
min_increment = 1.0e-5
max_num_inc = 100

# Mesh controls
mesh_size = 0.5
mesh_deviation_factor = 0.1
mesh_min_size_factor = 0.1

# Batch / output controls
num_microstructures = 200
start_seed = 1
submit_jobs = False
wait_for_completion = False
cpus = 1
write_input = True
input_output_dir = "fenicsx_output_files"
write_xdmf = False
write_npz = True


def _to_bool(value):
	return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_cli_args():
	args = {}
	raw = sys.argv[:]
	if "--" in raw:
		raw = raw[raw.index("--") + 1:]
	else:
		raw = raw[1:]

	for token in raw:
		if "=" in token:
			key, value = token.split("=", 1)
			args[key.strip()] = value.strip()
	return args


def _root_print(*args):
	if MPI.COMM_WORLD.rank == 0:
		print(*args)


def _progress(iterable, **kwargs):
	if MPI.COMM_WORLD.rank == 0:
		return tqdm(iterable, **kwargs)
	return iterable


def _summarize_legacy_controls():
	if submit_jobs:
		_root_print("submit=true ignored: FEniCSx solves each sample directly in-process.")
	if wait_for_completion:
		_root_print("wait=true ignored: there is no external job queue in the FEniCSx workflow.")
	if any(
		value != default
		for value, default in (
			(initial_increment, 1.0),
			(max_increment, 1.0),
			(min_increment, 1.0e-5),
			(max_num_inc, 100),
			)
		):
			_root_print(
				"Increment controls are retained for compatibility but unused by the direct linear solve."
			)


def _ensure_gmsh_ready():
	if not (hasattr(gmsh, "isInitialized") and gmsh.isInitialized()):
		gmsh.initialize()
	gmsh.option.setNumber("General.Terminal", 0)
	gmsh.clear()
	return gmsh.model


def build_fenicsx_mesh(
	model_name_value,
	fibers_value,
	L_value,
	mesh_size_value,
	mesh_deviation_factor_value,
	mesh_min_size_factor_value,
):
	model = _ensure_gmsh_ready()
	model.add(model_name_value)
	occ = model.occ

	outer = occ.addRectangle(0.0, 0.0, 0.0, L_value, L_value)
	fiber_surfaces = []
	for fiber in fibers_value:
		fiber_tag = occ.addDisk(
			float(fiber["x"]),
			float(fiber["y"]),
			0.0,
			float(fiber["a"]),
			float(fiber["b"]),
		)
		if abs(float(fiber["angle_deg"])) > 1.0e-12:
			occ.rotate(
				[(2, fiber_tag)],
				float(fiber["x"]),
				float(fiber["y"]),
				0.0,
				0.0,
				0.0,
				1.0,
				math.radians(float(fiber["angle_deg"])),
			)
		fiber_surfaces.append(fiber_tag)

	matrix_dimtags, _ = occ.cut(
		[(2, outer)],
		[(2, tag) for tag in fiber_surfaces],
		removeObject=True,
		removeTool=False,
	)
	occ.synchronize()

	matrix_surfaces = [tag for (dim, tag) in matrix_dimtags if dim == 2]
	if len(matrix_surfaces) == 0:
		raise RuntimeError("Gmsh cut produced no matrix surfaces.")

	matrix_group = model.addPhysicalGroup(2, matrix_surfaces, 1)
	model.setPhysicalName(2, matrix_group, "MATRIX")

	fiber_group = model.addPhysicalGroup(2, fiber_surfaces, 2)
	model.setPhysicalName(2, fiber_group, "FIBERS")

	min_size = max(mesh_size_value * mesh_min_size_factor_value, 1.0e-6)
	max_size = max(mesh_size_value * (1.0 + mesh_deviation_factor_value), min_size)
	point_entities = model.getEntities(0)
	if point_entities:
		model.mesh.setSize(point_entities, max_size)

	fiber_curves = []
	for fiber_tag in fiber_surfaces:
		for (dim, tag) in model.getBoundary([(2, fiber_tag)], oriented=False):
			if dim == 1:
				fiber_curves.append(tag)

	fiber_curves = sorted(set(fiber_curves))
	if fiber_curves:
		distance = model.mesh.field.add("Distance")
		model.mesh.field.setNumbers(distance, "CurvesList", fiber_curves)
		model.mesh.field.setNumber(distance, "Sampling", 100)

		threshold = model.mesh.field.add("Threshold")
		model.mesh.field.setNumber(threshold, "InField", distance)
		model.mesh.field.setNumber(threshold, "SizeMin", min_size)
		model.mesh.field.setNumber(threshold, "SizeMax", max_size)
		max_axis = max(max(float(fiber["a"]), float(fiber["b"])) for fiber in fibers_value)
		model.mesh.field.setNumber(threshold, "DistMin", max_axis)
		model.mesh.field.setNumber(threshold, "DistMax", 3.0 * max_axis)
		model.mesh.field.setAsBackgroundMesh(threshold)

	gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_size)
	gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_size)
	model.mesh.generate(2)

	mesh_data = gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
	return mesh_data.mesh, mesh_data.cell_tags


def _cell_tag_array(domain, cell_tags):
	tdim = domain.topology.dim
	num_local_cells = domain.topology.index_map(tdim).size_local
	tags = np.zeros(num_local_cells, dtype=np.int32)

	if cell_tags is None:
		return tags

	local_mask = cell_tags.indices < num_local_cells
	tags[cell_tags.indices[local_mask]] = cell_tags.values[local_mask]
	return tags


def build_material_fields(
	domain,
	cell_tags,
	matrix_E_value,
	matrix_nu_value,
	fiber_E_value,
	fiber_nu_value,
):
	tdim = domain.topology.dim
	index_map = domain.topology.index_map(tdim)
	num_cells_with_ghosts = index_map.size_local + index_map.num_ghosts

	Q = fem.functionspace(domain, ("DG", 0))
	E_field = fem.Function(Q, name="E")
	nu_field = fem.Function(Q, name="nu")

	e_values = np.full(num_cells_with_ghosts, float(matrix_E_value), dtype=np.float64)
	nu_values = np.full(num_cells_with_ghosts, float(matrix_nu_value), dtype=np.float64)

	if cell_tags is not None:
		fiber_cells = cell_tags.find(2)
		e_values[fiber_cells] = float(fiber_E_value)
		nu_values[fiber_cells] = float(fiber_nu_value)

	E_field.x.array[:] = e_values
	nu_field.x.array[:] = nu_values
	E_field.x.scatter_forward()
	nu_field.x.scatter_forward()

	return Q, E_field, nu_field


def _interpolate_cellwise_scalars(domain, named_expressions):
	Q = fem.functionspace(domain, ("DG", 0))
	interpolation_points = Q.element.interpolation_points
	functions = {}
	for name, expression in named_expressions.items():
		func = fem.Function(Q, name=name)
		func.interpolate(fem.Expression(expression, interpolation_points))
		func.x.scatter_forward()
		functions[name] = func
	return functions


def solve_microstructure(
	domain,
	cell_tags,
	L_value,
	applied_strain_x_value,
	matrix_E_value,
	matrix_nu_value,
	fiber_E_value,
	fiber_nu_value,
	analysis_type_value,
):
	if analysis_type_value not in ("plane_strain", "plane_stress"):
		raise ValueError("analysis_type must be plane_strain or plane_stress")

	_, E_field, nu_field = build_material_fields(
		domain,
		cell_tags,
		matrix_E_value,
		matrix_nu_value,
		fiber_E_value,
		fiber_nu_value,
	)

	mu_field = E_field / (2.0 * (1.0 + nu_field))
	if analysis_type_value == "plane_stress":
		lambda_field = E_field * nu_field / (1.0 - nu_field ** 2)
	else:
		lambda_field = E_field * nu_field / ((1.0 + nu_field) * (1.0 - 2.0 * nu_field))

	V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
	u = ufl.TrialFunction(V)
	v = ufl.TestFunction(V)

	def eps(w):
		return ufl.sym(ufl.grad(w))

	def sigma(w):
		return 2.0 * mu_field * eps(w) + lambda_field * ufl.tr(eps(w)) * ufl.Identity(domain.geometry.dim)

	dx = ufl.Measure("dx", domain=domain)

	Vx, _ = V.sub(0).collapse()
	Vy, _ = V.sub(1).collapse()

	left_dofs = fem.locate_dofs_geometrical(
		(V.sub(0), Vx),
		lambda x: np.isclose(x[0], 0.0),
	)[0]
	right_dofs = fem.locate_dofs_geometrical(
		(V.sub(0), Vx),
		lambda x: np.isclose(x[0], L_value),
	)[0]
	corner_dofs = fem.locate_dofs_geometrical(
		(V.sub(1), Vy),
		lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)),
	)[0]

	if len(left_dofs) == 0 or len(right_dofs) == 0 or len(corner_dofs) == 0:
		raise RuntimeError("Could not locate one or more boundary-condition dof sets.")

	bcs = [
		fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs, V.sub(0)),
		fem.dirichletbc(PETSc.ScalarType(0.0), corner_dofs, V.sub(1)),
		fem.dirichletbc(PETSc.ScalarType(applied_strain_x_value * L_value), right_dofs, V.sub(0)),
	]

	a_form = ufl.inner(sigma(u), eps(v)) * dx
	body_force = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=PETSc.ScalarType))
	l_form = ufl.dot(body_force, v) * dx

	problem = LinearProblem(
		a_form,
		l_form,
		bcs=bcs,
		petsc_options_prefix="micro_elas_",
		petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
	)

	u_sol = problem.solve()
	u_sol.name = "displacement"
	u_sol.x.scatter_forward()

	cellwise_fields = _interpolate_cellwise_scalars(
		domain,
		{
			"epsilon_xx": eps(u_sol)[0, 0],
			"epsilon_yy": eps(u_sol)[1, 1],
			"epsilon_xy": eps(u_sol)[0, 1],
			"sigma_xx": sigma(u_sol)[0, 0],
			"sigma_yy": sigma(u_sol)[1, 1],
			"sigma_xy": sigma(u_sol)[0, 1],
		},
	)

	u_x = u_sol.sub(0).collapse()
	u_y = u_sol.sub(1).collapse()

	return {
		"u": u_sol,
		"ux": u_x,
		"uy": u_y,
		"E": E_field,
		"nu": nu_field,
		**cellwise_fields,
	}


def extract_mesh_arrays(domain):
	tdim = domain.topology.dim
	domain.topology.create_connectivity(tdim, 0)
	connectivity = domain.topology.connectivity(tdim, 0)
	num_local_cells = domain.topology.index_map(tdim).size_local

	points = domain.geometry.x[:, :domain.geometry.dim].copy()
	cells = np.vstack([connectivity.links(cell_index) for cell_index in range(num_local_cells)]).astype(np.int32)

	cell_points = points[cells]
	cell_centroids = cell_points.mean(axis=1)
	cell_areas = 0.5 * np.abs(
		(cell_points[:, 1, 0] - cell_points[:, 0, 0]) * (cell_points[:, 2, 1] - cell_points[:, 0, 1])
		- (cell_points[:, 2, 0] - cell_points[:, 0, 0]) * (cell_points[:, 1, 1] - cell_points[:, 0, 1])
	)

	return points, cells, cell_centroids, cell_areas


def _tabulated_dof_coordinates(function, domain):
	coords = np.asarray(function.function_space.tabulate_dof_coordinates(), dtype=np.float64)
	if coords.ndim == 1:
		coords = coords.reshape((-1, domain.geometry.x.shape[1]))
	return coords[:, : domain.geometry.dim].copy()


def _area_average(values, cell_areas):
	return float(np.sum(values[: len(cell_areas)] * cell_areas) / np.sum(cell_areas))


def _format_timings(timings):
	return ", ".join(
		"{}={:.2f}s".format(name, float(value))
		for name, value in timings.items()
		if name != "total"
	) + ", total={:.2f}s".format(float(timings["total"]))


def write_sample_outputs(
	output_dir_value,
	seed_value,
	fibers_value,
	domain,
	cell_tags,
	solution,
	L_value,
	analysis_type_value,
	thickness_value,
	applied_strain_x_value,
	matrix_E_value,
	matrix_nu_value,
	fiber_E_value,
	fiber_nu_value,
):
	sample_dir = os.path.join(output_dir_value, "job_s{}".format(seed_value))
	os.makedirs(sample_dir, exist_ok=True)

	points, cells, cell_centroids, cell_areas = extract_mesh_arrays(domain)
	cell_phase_tags = _cell_tag_array(domain, cell_tags)

	ux_coords = _tabulated_dof_coordinates(solution["ux"], domain)
	uy_coords = _tabulated_dof_coordinates(solution["uy"], domain)

	if write_xdmf:
		xdmf_path = os.path.join(sample_dir, "solution.xdmf")
		with XDMFFile(domain.comm, xdmf_path, "w") as xdmf:
			xdmf.write_mesh(domain)
			xdmf.write_function(solution["u"])

	if write_npz:
		geometry_fields = fibers_to_metadata_fields(fibers_value, fiber_shape)
		np.savez(
			os.path.join(sample_dir, "sample_data.npz"),
			points=points,
			cells=cells,
			cell_tags=cell_phase_tags,
			cell_centroids=cell_centroids,
			cell_areas=cell_areas,
			ux_coords=ux_coords,
			uy_coords=uy_coords,
			ux=solution["ux"].x.array.copy(),
				uy=solution["uy"].x.array.copy(),
				E=solution["E"].x.array.copy(),
				nu=solution["nu"].x.array.copy(),
				epsilon_xx=solution["epsilon_xx"].x.array.copy(),
				epsilon_yy=solution["epsilon_yy"].x.array.copy(),
				epsilon_xy=solution["epsilon_xy"].x.array.copy(),
				sigma_xx=solution["sigma_xx"].x.array.copy(),
				sigma_yy=solution["sigma_yy"].x.array.copy(),
				sigma_xy=solution["sigma_xy"].x.array.copy(),
			fiber_centers=np.asarray(geometry_fields["fiber_centers"], dtype=np.float64),
			fiber_axes=np.asarray(geometry_fields["fiber_axes"], dtype=np.float64),
			fiber_angles_deg=np.asarray(geometry_fields["fiber_angles_deg"], dtype=np.float64),
			fiber_radius=np.asarray([geometry_fields["fiber_radius"]], dtype=np.float64),
		)
	avg_epsilon_xx = _area_average(solution["epsilon_xx"].x.array, cell_areas)
	avg_epsilon_yy = _area_average(solution["epsilon_yy"].x.array, cell_areas)
	avg_epsilon_xy = _area_average(solution["epsilon_xy"].x.array, cell_areas)
	avg_sigma_xx = _area_average(solution["sigma_xx"].x.array, cell_areas)
	avg_sigma_yy = _area_average(solution["sigma_yy"].x.array, cell_areas)
	avg_sigma_xy = _area_average(solution["sigma_xy"].x.array, cell_areas)

	metadata = {
		"seed": int(seed_value),
		"L": float(L_value),
		"N_fibers": int(len(fibers_value)),
		"analysis_type": analysis_type_value,
		"thickness": float(thickness_value),
		"applied_strain_x": float(applied_strain_x_value),
		"matrix_E": float(matrix_E_value),
		"matrix_nu": float(matrix_nu_value),
		"fiber_E": float(fiber_E_value),
		"fiber_nu": float(fiber_nu_value),
			"mesh_num_points": int(points.shape[0]),
			"mesh_num_cells": int(cells.shape[0]),
			"avg_epsilon_xx": avg_epsilon_xx,
			"avg_epsilon_yy": avg_epsilon_yy,
			"avg_epsilon_xy": avg_epsilon_xy,
			"avg_sigma_xx": avg_sigma_xx,
			"avg_sigma_yy": avg_sigma_yy,
			"avg_sigma_xy": avg_sigma_xy,
		"files": {
			"xdmf": "solution.xdmf" if write_xdmf else None,
			"npz": "sample_data.npz" if write_npz else None,
		},
	}
	metadata.update(fibers_to_metadata_fields(fibers_value, fiber_shape))

	with open(os.path.join(sample_dir, "metadata.json"), "w") as fobj:
		json.dump(metadata, fobj, indent=2)

	return sample_dir, metadata


def _sample_task_config(seed_i):
	return {
		"seed": int(seed_i),
		"L": float(L),
		"N_fibers": int(N_fibers),
		"Vf": float(Vf),
		"min_spacing_factor": float(min_spacing_factor),
		"max_attempt_factor": float(max_attempt_factor),
		"fiber_shape": fiber_shape,
		"fiber_aspect_ratio": float(fiber_aspect_ratio),
		"fiber_angle_deg": float(fiber_angle_deg),
		"random_fiber_angle": bool(random_fiber_angle),
		"model_name": model_name,
		"part_name": part_name,
		"analysis_type": analysis_type,
		"thickness": float(thickness),
		"matrix_E": float(matrix_E),
		"matrix_nu": float(matrix_nu),
		"fiber_E": float(fiber_E),
		"fiber_nu": float(fiber_nu),
		"applied_strain_x": float(applied_strain_x),
		"mesh_size": float(mesh_size),
		"mesh_deviation_factor": float(mesh_deviation_factor),
		"mesh_min_size_factor": float(mesh_min_size_factor),
		"write_input": bool(write_input),
		"write_xdmf": bool(write_xdmf),
		"write_npz": bool(write_npz),
		"output_dir": input_output_dir,
	}


def _generate_single_sample(task_config):
	seed_i = int(task_config["seed"])
	timings = {}
	time_start = time.perf_counter()

	stage_start = time.perf_counter()
	fibers = generate_fibers(
		task_config["L"],
		task_config["N_fibers"],
		task_config["Vf"],
		seed_i,
		task_config["min_spacing_factor"],
		task_config["max_attempt_factor"],
		task_config["fiber_shape"],
		task_config["fiber_aspect_ratio"],
		task_config["fiber_angle_deg"],
		task_config["random_fiber_angle"],
	)
	timings["placement"] = time.perf_counter() - stage_start

	stage_start = time.perf_counter()
	domain, cell_tags = build_fenicsx_mesh(
		"{}_s{}".format(task_config["model_name"], seed_i),
		fibers,
		task_config["L"],
		task_config["mesh_size"],
		task_config["mesh_deviation_factor"],
		task_config["mesh_min_size_factor"],
	)
	timings["meshing"] = time.perf_counter() - stage_start

	stage_start = time.perf_counter()
	solution = solve_microstructure(
		domain,
		cell_tags,
		task_config["L"],
		task_config["applied_strain_x"],
		task_config["matrix_E"],
		task_config["matrix_nu"],
		task_config["fiber_E"],
		task_config["fiber_nu"],
		task_config["analysis_type"],
	)
	timings["solve_and_fields"] = time.perf_counter() - stage_start

	sample_dir = None
	metadata = None
	if task_config["write_input"]:
		stage_start = time.perf_counter()
		global write_xdmf
		global write_npz
		write_xdmf_original = write_xdmf
		write_npz_original = write_npz
		write_xdmf = task_config["write_xdmf"]
		write_npz = task_config["write_npz"]
		try:
			sample_dir, metadata = write_sample_outputs(
				task_config["output_dir"],
				seed_i,
				fibers,
				domain,
				cell_tags,
				solution,
				task_config["L"],
				task_config["analysis_type"],
				task_config["thickness"],
				task_config["applied_strain_x"],
				task_config["matrix_E"],
				task_config["matrix_nu"],
				task_config["fiber_E"],
				task_config["fiber_nu"],
			)
		finally:
			write_xdmf = write_xdmf_original
			write_npz = write_npz_original
		timings["write"] = time.perf_counter() - stage_start

	timings["total"] = time.perf_counter() - time_start
	return {
		"seed": seed_i,
		"sample_dir": sample_dir if sample_dir is not None else "seed_{}".format(seed_i),
		"metadata": metadata,
		"timings": timings,
		"fiber_radius": float(equivalent_radius(fibers[0])),
		"fiber_count": int(len(fibers)),
	}


def _generate_single_sample_for_seed(seed_i):
	return _generate_single_sample(_sample_task_config(int(seed_i)))


def _parallel_sample_task(seed_i):
	try:
		return int(seed_i), {"ok": True, "result": _generate_single_sample_for_seed(int(seed_i))}
	except Exception as err:
		return int(seed_i), {"ok": False, "error": str(err)}


def _run_serial_generation(seeds):
	results = []
	failed = []
	for seed_i in _progress(seeds, desc="Generating samples", unit="sample"):
		try:
			result = _generate_single_sample(_sample_task_config(seed_i))
			results.append(result)
			_root_print(
				"Seed {} complete: fibers={} | {} | output={}".format(
					seed_i,
					result["fiber_count"],
					_format_timings(result["timings"]),
					result["sample_dir"],
				)
			)
		except Exception as err:
			failed.append((seed_i, str(err)))
			_root_print("ERROR for seed {}: {}".format(seed_i, err))
	return results, failed


def _run_parallel_generation(seeds):
	results = []
	failed = []
	ctx = get_context("fork")
	num_workers = max(1, min(int(cpus), len(seeds)))
	with ctx.Pool(processes=num_workers) as pool:
		for seed_i, payload in tqdm(
			pool.imap_unordered(_parallel_sample_task, seeds),
			total=len(seeds),
			desc="Generating samples",
			unit="sample",
		):
			if payload["ok"]:
				result = payload["result"]
				results.append(result)
				tqdm.write(
					"Seed {} complete: fibers={} | {} | output={}".format(
						result["seed"],
						result["fiber_count"],
						_format_timings(result["timings"]),
						result["sample_dir"],
					)
				)
			else:
				failed_entry = (int(seed_i), payload["error"])
				failed.append(failed_entry)
				tqdm.write("ERROR for seed {}: {}".format(failed_entry[0], failed_entry[1]))
	results.sort(key=lambda item: item["seed"])
	failed.sort(key=lambda item: item[0])
	return results, failed


def main():
	global L
	global N_fibers
	global Vf
	global seed
	global min_spacing_factor
	global max_attempt_factor
	global fiber_shape
	global fiber_aspect_ratio
	global fiber_angle_deg
	global random_fiber_angle
	global model_name
	global part_name
	global thickness
	global analysis_type
	global matrix_E
	global matrix_nu
	global fiber_E
	global fiber_nu
	global applied_strain_x
	global initial_increment
	global max_increment
	global min_increment
	global max_num_inc
	global mesh_size
	global mesh_deviation_factor
	global mesh_min_size_factor
	global num_microstructures
	global start_seed
	global submit_jobs
	global wait_for_completion
	global cpus
	global write_input
	global input_output_dir
	global write_xdmf
	global write_npz

	cli_args = _parse_cli_args()

	L = float(cli_args.get("L", L))
	N_fibers = int(cli_args.get("N_fibers", N_fibers))
	Vf = float(cli_args.get("Vf", Vf))
	min_spacing_factor = float(cli_args.get("min_spacing_factor", min_spacing_factor))
	max_attempt_factor = float(cli_args.get("max_attempt_factor", max_attempt_factor))
	fiber_shape = normalize_fiber_shape(cli_args.get("fiber_shape", fiber_shape))
	fiber_aspect_ratio = float(cli_args.get("fiber_aspect_ratio", fiber_aspect_ratio))
	fiber_angle_deg = float(cli_args.get("fiber_angle_deg", fiber_angle_deg))
	random_fiber_angle = _to_bool(cli_args.get("random_fiber_angle", random_fiber_angle))
	analysis_type = cli_args.get("analysis_type", analysis_type)

	thickness = float(cli_args.get("thickness", thickness))
	matrix_E = float(cli_args.get("matrix_E", matrix_E))
	matrix_nu = float(cli_args.get("matrix_nu", matrix_nu))
	fiber_E = float(cli_args.get("fiber_E", fiber_E))
	fiber_nu = float(cli_args.get("fiber_nu", fiber_nu))

	applied_strain_x = float(cli_args.get("applied_strain_x", applied_strain_x))
	initial_increment = float(cli_args.get("initial_increment", initial_increment))
	max_increment = float(cli_args.get("max_increment", max_increment))
	min_increment = float(cli_args.get("min_increment", min_increment))
	max_num_inc = int(cli_args.get("max_num_inc", max_num_inc))

	mesh_size = float(cli_args.get("mesh_size", mesh_size))
	mesh_deviation_factor = float(cli_args.get("mesh_deviation_factor", mesh_deviation_factor))
	mesh_min_size_factor = float(cli_args.get("mesh_min_size_factor", mesh_min_size_factor))

	model_name = cli_args.get("model_name", model_name)
	part_name = cli_args.get("part_name", part_name)

	num_microstructures = int(cli_args.get("num", num_microstructures))
	start_seed = int(cli_args.get("start_seed", start_seed))
	seed = int(cli_args.get("seed", seed))

	submit_jobs = _to_bool(cli_args.get("submit", submit_jobs))
	wait_for_completion = _to_bool(cli_args.get("wait", wait_for_completion))
	cpus = int(cli_args.get("cpus", cpus))
	write_input = _to_bool(cli_args.get("write_input", write_input))
	input_output_dir = cli_args.get("output_dir", cli_args.get("input_output_dir", input_output_dir))
	write_xdmf = _to_bool(cli_args.get("write_xdmf", write_xdmf))
	write_npz = _to_bool(cli_args.get("write_npz", write_npz))

	if "seeds" in cli_args:
		seeds = [int(seed_item.strip()) for seed_item in cli_args["seeds"].split(",") if seed_item.strip()]
	elif num_microstructures > 1:
		seeds = [start_seed + idx for idx in range(num_microstructures)]
	else:
		seeds = [seed]

	_root_print("Running for seeds:", seeds)
	_root_print("Output directory:", input_output_dir)
	_summarize_legacy_controls()
	_root_print("Worker processes:", cpus)
	_root_print("fiber_shape:", fiber_shape)
	_root_print("fiber_aspect_ratio:", fiber_aspect_ratio)
	_root_print("random_fiber_angle:", random_fiber_angle)
	_root_print("write_xdmf:", write_xdmf)

	if write_input and MPI.COMM_WORLD.rank == 0:
		os.makedirs(input_output_dir, exist_ok=True)
	if MPI.COMM_WORLD.size != 1 and cpus > 1:
		raise RuntimeError("cpus>1 uses multiprocessing and requires a non-MPI launch (COMM_WORLD.size must be 1).")

	batch_start = time.perf_counter()
	if cpus > 1:
		results, failed_seeds = _run_parallel_generation(seeds)
	else:
		results, failed_seeds = _run_serial_generation(seeds)
	batch_elapsed = time.perf_counter() - batch_start

	created_samples = [item["sample_dir"] for item in results]
	sample_metadata = [item["metadata"] for item in results if item["metadata"] is not None]
	timing_records = [item["timings"] for item in results]
	timing_summary = {}
	if timing_records:
		for key in timing_records[0].keys():
			values = np.asarray([record[key] for record in timing_records], dtype=np.float64)
			timing_summary[key] = {
				"mean_seconds": float(np.mean(values)),
				"max_seconds": float(np.max(values)),
				"min_seconds": float(np.min(values)),
			}

	if write_input and MPI.COMM_WORLD.rank == 0:
		shape_family = "2D circular fiber composite RVE" if fiber_shape == "circle" else "2D elliptical fiber composite RVE"
		semi_major, semi_minor = compute_fiber_axes(L, N_fibers, Vf, fiber_shape, fiber_aspect_ratio)
		batch_summary = {
			"generator": "fenicsx",
			"shape_family": shape_family,
			"parameters": {
				"L": float(L),
				"N_fibers": int(N_fibers),
				"Vf": float(Vf),
				"min_spacing_factor": float(min_spacing_factor),
				"max_attempt_factor": float(max_attempt_factor),
				"fiber_shape": fiber_shape,
				"fiber_aspect_ratio": float(fiber_aspect_ratio),
				"fiber_angle_deg": float(fiber_angle_deg),
				"random_fiber_angle": bool(random_fiber_angle),
				"fiber_semi_major": float(semi_major),
				"fiber_semi_minor": float(semi_minor),
				"analysis_type": analysis_type,
				"thickness": float(thickness),
				"matrix_E": float(matrix_E),
				"matrix_nu": float(matrix_nu),
				"fiber_E": float(fiber_E),
				"fiber_nu": float(fiber_nu),
				"applied_strain_x": float(applied_strain_x),
				"mesh_size": float(mesh_size),
				"mesh_deviation_factor": float(mesh_deviation_factor),
				"mesh_min_size_factor": float(mesh_min_size_factor),
			},
			"seeds_requested": seeds,
			"worker_processes": int(cpus),
			"samples_written": created_samples,
			"sample_metadata": sample_metadata,
			"failed_seeds": failed_seeds,
			"timing_summary": timing_summary,
			"wall_time_seconds": float(batch_elapsed),
		}
		with open(os.path.join(input_output_dir, "batch_summary.json"), "w") as fobj:
			json.dump(batch_summary, fobj, indent=2)

	_root_print("Microstructure batch generation complete.")
	_root_print("Created samples:", created_samples)
	_root_print("Wall time: {:.2f}s".format(batch_elapsed))
	if timing_summary:
		_root_print("Average timings:", timing_summary)
	if failed_seeds:
		_root_print("Failed seeds:", failed_seeds)


if __name__ == "__main__":
	main()
