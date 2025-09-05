# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model utilities for Neural Assets."""

from __future__ import annotations

from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

NpOrJaxArray = TypeVar('NpOrJaxArray', jnp.ndarray, np.ndarray)


def flatten_batch_axes(
    x: NpOrJaxArray, num_data_axes: int
) -> tuple[NpOrJaxArray, tuple[int, ...]]:
  """Flatten all leading batch dim to only one batch dim."""
  assert num_data_axes >= 1, f'{num_data_axes=} must be >= 1'
  batch_shape = x.shape[:-num_data_axes]
  if len(batch_shape) > 1:
    x = x.reshape((np.prod(batch_shape),) + x.shape[-num_data_axes:])
  return x, batch_shape


def unflatten_batch_axes(
    x: NpOrJaxArray, batch_shape: tuple[int, ...]
) -> NpOrJaxArray:
  """Undo flatten_batch_axes."""
  if len(batch_shape) > 1:
    x = x.reshape(batch_shape + x.shape[1:])
  return x


def roi_align(
    feature_map: jnp.ndarray, bbox: jnp.ndarray, output_size: int
) -> jnp.ndarray:
  """Extracts a fixed-size feature map for an ROI from a larger feature map.

  Function adapted from
  https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/layers.py

  Compared to the original code, we use a different bbox format here.

  Args:
    feature_map: [h, w, c] map of features from which to crop a region of
      interest.
    bbox: [y0, x0, y1, x1] normalized to (0, 1), bbox defining the region of
      interest.
    output_size: Size of the output feature map.

  Returns:
    Crop of size [output_size, output_size, c] taken from feature_map.
  """
  input_height, input_width, c = feature_map.shape
  output_height = output_width = output_size

  y0, x0, y1, x1 = jnp.split(bbox, 4, axis=-1)
  w = x1 - x0
  h = y1 - y0
  w = jnp.maximum(w, 1e-6)
  h = jnp.maximum(h, 1e-6)
  x_scale = output_width / (w * input_width)
  y_scale = output_height / (h * input_height)

  return jax.image.scale_and_translate(
      feature_map,
      shape=(output_height, output_width, c),
      spatial_dims=(0, 1),
      scale=jnp.concatenate((y_scale, x_scale)),
      translation=jnp.concatenate(
          (-y0 * output_height / h, -x0 * output_width / w)
      ),
      method='linear',
      precision=jax.lax.Precision('fastest'),
  )


def get_roi_align_features(
    bboxes: jnp.ndarray,  # [B, n, 4]
    feature_maps: jnp.ndarray,  # [B, h, w, c]
    size: int,
) -> jnp.ndarray:  # [B, n, s, s, c]
  """Apply RoIAlign to extract fix-sized per-bbox feature maps."""
  # Handle arbitrary leading batch dim.
  bboxes, batch_shape = flatten_batch_axes(bboxes, num_data_axes=2)
  feature_maps, _ = flatten_batch_axes(
      feature_maps, num_data_axes=feature_maps.ndim - len(batch_shape)
  )
  # Apply RoIAlign.
  assert feature_maps.ndim == 4, f'Unsupported {feature_maps.shape=}'
  roi_align_image = jax.vmap(roi_align, in_axes=[None, 0, None])
  roi_align_batch = jax.vmap(roi_align_image, in_axes=[0, 0, None])
  roi_features = roi_align_batch(feature_maps, bboxes, size)
  return unflatten_batch_axes(roi_features, batch_shape=batch_shape)


def boxes_to_sparse_segmentations(
    boxes: NpOrJaxArray,
    height: int,
    width: int,
    np_backbone: Any = jnp,
) -> NpOrJaxArray:
  """Converts bounding boxes into sparse segmentations.

  Args:
    boxes: A bounding box tensor of shape [..., N, 4] in TF format (i.e.,
      normalized coordinates [ymin, xmin, ymax, xmax] in [0, 1]^4.
    height: The frame height.
    width: The frame width.
    np_backbone: numpy module: Either the regular numpy package or jax.numpy.

  Returns:
    A sparse segmentations tensor of shape [..., N, H, W].
  """
  batch_shape = boxes.shape[:-2]
  n = boxes.shape[-2]
  boxes = np_backbone.reshape(boxes, (-1, n, 4))
  # Convert the normalized into absolute coordinates.
  boxes_absolute = np_backbone.round(
      boxes
      * np_backbone.array([height, width, height, width], np_backbone.float32)[
          np_backbone.newaxis, np_backbone.newaxis
      ]
  ).astype(np_backbone.int32)
  ymin, xmin, ymax, xmax = np_backbone.split(boxes_absolute, 4, axis=-1)
  # Generate yx-grid and mask the boxes.
  grid_x, grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
  sparse_segmentations = np_backbone.logical_and(
      np_backbone.logical_and(
          grid_y >= ymin[..., np_backbone.newaxis],
          grid_y < ymax[..., np_backbone.newaxis],
      ),
      np_backbone.logical_and(
          grid_x >= xmin[..., np_backbone.newaxis],
          grid_x < xmax[..., np_backbone.newaxis],
      ),
  )
  sparse_segmentations = np_backbone.reshape(
      sparse_segmentations, batch_shape + (n, height, width)
  ).astype(np_backbone.uint8)
  return sparse_segmentations
