# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gathers all feature generators.

Calculates features based on input data and returns flattened output."""
from mxnet import nd
import numpy as np


def spatial_features(reshape, grid_spacing=1.):
    """Returns spatial coordinate features.

    The parameter grid_spacing scales the output grid. If grid_spacing = 0 is
    chosen, the grid is normalized with respect to width and height.
    """

    def generate(data):
        batch_size, _, height, width = data.shape
        spatial_mesh = generate_spatial_mesh(height, width, grid_spacing)
        spatial_mesh = spatial_mesh.repeat(repeats=batch_size, axis=0)
        if reshape:
            spatial_mesh = spatial_mesh.reshape([batch_size, 2, -1])
        return spatial_mesh

    return generate


def activation_features(reshape, scale_factor=1.0):
    """Returns image data as features."""

    def generate(data):
        if reshape:
            batch_size, num_channels = data.shape[0:2]
            data = data.reshape([batch_size, num_channels, -1])
        return scale_factor * data

    return generate


def stack_features(list_features):
    """Stacks several features by concatenating 2nd dimension."""

    def stack(data):
        composed_features = [feature(data) for feature in list_features]
        return nd.concat(*composed_features, dim=1)

    return stack


def generate_spatial_mesh(height, width, grid_spacing):
    """Generates spatial xy-mesh for given height and width.

    grid_spacing == 0 returns features normalized by width and height.
    """
    if grid_spacing == 0:
        x_coordinates = np.arange(0, width) / (width - 1)
        y_coordinates = np.arange(0, height) / (height - 1)
    else:
        x_coordinates = np.arange(0, width) * grid_spacing
        y_coordinates = np.arange(0, height) * grid_spacing

    # Transform coordinates to zero mean.
    x_coordinates = x_coordinates - np.mean(x_coordinates)
    y_coordinates = y_coordinates - np.mean(y_coordinates)

    x_mesh, y_mesh = np.meshgrid(x_coordinates, y_coordinates)
    x_mesh = x_mesh.reshape([1, 1, height, width])
    y_mesh = y_mesh.reshape([1, 1, height, width])
    spatial_mesh = np.concatenate((x_mesh, y_mesh), axis=1)
    return nd.array(spatial_mesh)
