#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random

import numpy as np


def normalize_fiber_shape(fiber_shape):
	shape = str(fiber_shape).strip().lower()
	if shape not in ("circle", "ellipse"):
		raise ValueError("fiber_shape must be 'circle' or 'ellipse'")
	return shape


def compute_fiber_axes(L_value, n_fibers_value, vf_value, fiber_shape, fiber_aspect_ratio):
	shape = normalize_fiber_shape(fiber_shape)
	area_per_fiber = float(vf_value) * float(L_value) * float(L_value) / float(n_fibers_value)

	if shape == "circle":
		radius = math.sqrt(area_per_fiber / math.pi)
		return radius, radius

	aspect_ratio = float(fiber_aspect_ratio)
	if aspect_ratio < 1.0:
		raise ValueError("fiber_aspect_ratio must be >= 1.0 for ellipse fibers")

	semi_major = math.sqrt(area_per_fiber * aspect_ratio / math.pi)
	semi_minor = math.sqrt(area_per_fiber / (math.pi * aspect_ratio))
	return semi_major, semi_minor


def equivalent_radius(fiber):
	return math.sqrt(float(fiber["a"]) * float(fiber["b"]))


def max_semi_axis(fiber):
	return max(float(fiber["a"]), float(fiber["b"]))


def _point_in_fiber(x_coord, y_coord, fiber, tol=1.0e-9):
	cos_theta = math.cos(math.radians(float(fiber["angle_deg"])))
	sin_theta = math.sin(math.radians(float(fiber["angle_deg"])))
	dx = float(x_coord) - float(fiber["x"])
	dy = float(y_coord) - float(fiber["y"])
	x_local = cos_theta * dx + sin_theta * dy
	y_local = -sin_theta * dx + cos_theta * dy
	return (x_local / float(fiber["a"])) ** 2 + (y_local / float(fiber["b"])) ** 2 <= 1.0 + tol


def _fiber_boundary_points(fiber, num_points=72):
	angles = np.linspace(0.0, 2.0 * math.pi, int(num_points), endpoint=False)
	cos_theta = math.cos(math.radians(float(fiber["angle_deg"])))
	sin_theta = math.sin(math.radians(float(fiber["angle_deg"])))
	points = []
	for angle in angles:
		x_local = float(fiber["a"]) * math.cos(angle)
		y_local = float(fiber["b"]) * math.sin(angle)
		x_coord = float(fiber["x"]) + cos_theta * x_local - sin_theta * y_local
		y_coord = float(fiber["y"]) + sin_theta * x_local + cos_theta * y_local
		points.append((x_coord, y_coord))
	return points


def _inflate_fiber(fiber, factor):
	return {
		"x": float(fiber["x"]),
		"y": float(fiber["y"]),
		"a": float(fiber["a"]) * float(factor),
		"b": float(fiber["b"]) * float(factor),
		"angle_deg": float(fiber["angle_deg"]),
	}


def fibers_overlap(fiber_a, fiber_b, spacing_factor=1.0, boundary_samples=72):
	inflated_a = _inflate_fiber(fiber_a, spacing_factor)
	inflated_b = _inflate_fiber(fiber_b, spacing_factor)

	dx = float(inflated_a["x"]) - float(inflated_b["x"])
	dy = float(inflated_a["y"]) - float(inflated_b["y"])
	center_distance = math.hypot(dx, dy)
	if center_distance > max_semi_axis(inflated_a) + max_semi_axis(inflated_b):
		return False

	if _point_in_fiber(inflated_a["x"], inflated_a["y"], inflated_b):
		return True
	if _point_in_fiber(inflated_b["x"], inflated_b["y"], inflated_a):
		return True

	for point in _fiber_boundary_points(inflated_a, num_points=boundary_samples):
		if _point_in_fiber(point[0], point[1], inflated_b):
			return True
	for point in _fiber_boundary_points(inflated_b, num_points=boundary_samples):
		if _point_in_fiber(point[0], point[1], inflated_a):
			return True
	return False


def generate_fibers(
	L_value,
	n_fibers_value,
	vf_value,
	seed_value,
	min_spacing_factor_value,
	max_attempt_factor_value,
	fiber_shape="circle",
	fiber_aspect_ratio=1.0,
	fiber_angle_deg=0.0,
	random_fiber_angle=False,
):
	shape = normalize_fiber_shape(fiber_shape)
	semi_major, semi_minor = compute_fiber_axes(
		L_value=L_value,
		n_fibers_value=n_fibers_value,
		vf_value=vf_value,
		fiber_shape=shape,
		fiber_aspect_ratio=fiber_aspect_ratio,
	)

	rng = random.Random(seed_value)
	margin = semi_major
	max_attempts = int(float(max_attempt_factor_value) * max(1, int(n_fibers_value)))
	attempts = 0
	fibers = []

	while len(fibers) < int(n_fibers_value):
		attempts += 1
		if attempts > max_attempts:
			raise RuntimeError(
				"Could not place all fibers for seed {} (placed {}/{}). "
				"Try lower Vf, fewer fibers, lower min_spacing_factor, or a smaller ellipse aspect ratio.".format(
					seed_value, len(fibers), n_fibers_value
				)
			)

		x_coord = rng.uniform(margin, float(L_value) - margin)
		y_coord = rng.uniform(margin, float(L_value) - margin)
		angle_deg = rng.uniform(0.0, 180.0) if bool(random_fiber_angle) else float(fiber_angle_deg)
		candidate = {
			"x": float(x_coord),
			"y": float(y_coord),
			"a": float(semi_major),
			"b": float(semi_minor),
			"angle_deg": float(angle_deg),
		}

		if any(fibers_overlap(candidate, existing, spacing_factor=min_spacing_factor_value) for existing in fibers):
			continue

		fibers.append(candidate)

	return fibers


def rasterize_fibers(L_value, fibers, img_size, flipud=1):
	x_coords = np.linspace(0.0, float(L_value), int(img_size))
	y_coords = np.linspace(0.0, float(L_value), int(img_size))
	x_grid, y_grid = np.meshgrid(x_coords, y_coords)

	mask = np.zeros((int(img_size), int(img_size)), dtype=np.float32)
	for fiber in fibers:
		theta = math.radians(float(fiber["angle_deg"]))
		cos_theta = math.cos(theta)
		sin_theta = math.sin(theta)
		dx = x_grid - float(fiber["x"])
		dy = y_grid - float(fiber["y"])
		x_local = cos_theta * dx + sin_theta * dy
		y_local = -sin_theta * dx + cos_theta * dy
		mask[(x_local / float(fiber["a"])) ** 2 + (y_local / float(fiber["b"])) ** 2 <= 1.0] = 1.0

	if int(flipud) == 1:
		mask = np.flipud(mask)

	return mask


def fibers_from_metadata(metadata):
	axes = metadata.get("fiber_axes")
	centers = metadata.get("fiber_centers", [])
	angles = metadata.get("fiber_angles_deg")
	if axes is not None:
		if angles is None:
			angles = [0.0] * len(axes)
		return [
			{
				"x": float(center[0]),
				"y": float(center[1]),
				"a": float(axis_pair[0]),
				"b": float(axis_pair[1]),
				"angle_deg": float(angle_deg),
			}
			for center, axis_pair, angle_deg in zip(centers, axes, angles)
		]

	radius = float(metadata["fiber_radius"])
	return [
		{
			"x": float(center[0]),
			"y": float(center[1]),
			"a": radius,
			"b": radius,
			"angle_deg": 0.0,
		}
		for center in centers
	]


def fibers_to_metadata_fields(fibers, fiber_shape):
	shape = normalize_fiber_shape(fiber_shape)
	fields = {
		"fiber_shape": shape,
		"fiber_centers": [[float(fiber["x"]), float(fiber["y"])] for fiber in fibers],
		"fiber_axes": [[float(fiber["a"]), float(fiber["b"])] for fiber in fibers],
		"fiber_angles_deg": [float(fiber["angle_deg"]) for fiber in fibers],
		"fiber_radius": float(equivalent_radius(fibers[0])) if fibers else None,
	}
	return fields
