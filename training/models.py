#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.GELU(),
		)

	def forward(self, x):
		return self.block(x)


class DownBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.pool = nn.MaxPool2d(2)
		self.conv = ConvBlock(in_channels, out_channels)

	def forward(self, x):
		return self.conv(self.pool(x))


class UpBlock(nn.Module):
	def __init__(self, in_channels, skip_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.conv = ConvBlock(out_channels + skip_channels, out_channels)

	def forward(self, x, skip):
		x = self.up(x)
		diff_y = skip.size(2) - x.size(2)
		diff_x = skip.size(3) - x.size(3)
		if diff_x != 0 or diff_y != 0:
			x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class ForwardSurrogateNet(nn.Module):
	def __init__(self, in_channels, out_field_channels=0, out_global_channels=0, base_channels=32):
		super().__init__()
		if out_field_channels <= 0 and out_global_channels <= 0:
			raise ValueError("At least one output head must be enabled.")

		self.out_field_channels = int(out_field_channels)
		self.out_global_channels = int(out_global_channels)

		self.inc = ConvBlock(in_channels, base_channels)
		self.down1 = DownBlock(base_channels, base_channels * 2)
		self.down2 = DownBlock(base_channels * 2, base_channels * 4)
		self.down3 = DownBlock(base_channels * 4, base_channels * 8)
		self.bottleneck = DownBlock(base_channels * 8, base_channels * 16)

		if self.out_field_channels > 0:
			self.up1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
			self.up2 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
			self.up3 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
			self.up4 = UpBlock(base_channels * 2, base_channels, base_channels)
			self.field_head = nn.Conv2d(base_channels, self.out_field_channels, kernel_size=1)

		if self.out_global_channels > 0:
			self.global_head = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Flatten(),
				nn.Linear(base_channels * 16, base_channels * 8),
				nn.GELU(),
				nn.Linear(base_channels * 8, self.out_global_channels),
			)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.bottleneck(x4)

		outputs = {}
		if self.out_field_channels > 0:
			y = self.up1(x5, x4)
			y = self.up2(y, x3)
			y = self.up3(y, x2)
			y = self.up4(y, x1)
			outputs["fields"] = self.field_head(y)

		if self.out_global_channels > 0:
			outputs["global"] = self.global_head(x5)

		return outputs
