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

"""Data preprocessing operations."""

import dataclasses
from typing import Dict, Optional, Sequence, Union

import einops
import numpy as np
import tensorflow as tf
import utils_3d


Features = Dict[str, Union[tf.Tensor, 'Features']]
NOTRACK_BOX = (0.0,) * 4
NOTRACK_BOX_3D = (0.0,) * 10
NOTRACK_PROJ_4_CORNER = (0.0,) * 12

CONDITIONING_TOKENS_MASK_KEY = 'valid_object_mask'
UNCONDITIONING_TOKENS_MASK_KEY = 'negative_object_mask'
UNCOND_BBOXES_KEY = 'negative_bboxes'
UNCOND_OBJECT_POSES_KEY = 'negative_object_poses'


def dense_to_sparse_segmentations(
    segmentation: tf.Tensor, max_instances: int
) -> tf.Tensor:
  """Converts a dense segmentation into a sparse segmentation representation.

  Args:
    segmentation: A dense segmentation tensor of shape [..., H, W] and dtype
      tf.int32, see modalities.SEGMENTATIONS.
    max_instances: The maximum number of instance masks/segment IDs.

  Returns:
    A binary sparse segmentations tensor of shape [..., max_instances, H, W],
    see modalities.SPARSE_SEGMENTATIONS.

  Raises:
    ValueError if segmentation.dtype is not tf.int32.
  """
  if segmentation.dtype != tf.int32:
    raise ValueError(
        f'Invalid dtype: segmentation.dtype expected={tf.int32}, '
        f'actual={segmentation.dtype}'
    )
  sparse_segmentation = tf.cast(
      tf.one_hot(
          segmentation - 1,  # Remove the background label.
          max_instances,
          axis=len(segmentation.shape) - 2,
          on_value=1,
          off_value=0,
      ),
      tf.uint8,
  )
  return sparse_segmentation


def sparse_segmentations_to_boxes(sparse_segmentations: tf.Tensor) -> tf.Tensor:
  """Converts sparse segmentation into bounding boxes.

  Args:
    sparse_segmentations: A sparse segmentation tensor of shape [..., N, H, W],
      see modalities.SPARSE_SEGMENTATIONS.

  Returns:
    The bounding boxes corresponding to the sparse segmentations, [..., N, 4].
  """
  vertical = tf.cast(tf.reduce_any(sparse_segmentations > 0, axis=-1), tf.int32)
  box_top = tf.reduce_sum(
      tf.cast(tf.cumsum(vertical, axis=-1) == 0, tf.int32), axis=-1
  )
  box_bottom = tf.reduce_sum(
      tf.cast(tf.cumsum(vertical, axis=-1, reverse=True) > 0, tf.int32), axis=-1
  )

  horizontal = tf.cast(
      tf.reduce_any(sparse_segmentations > 0, axis=-2), tf.int32
  )
  box_left = tf.reduce_sum(
      tf.cast(tf.cumsum(horizontal, axis=-1) == 0, tf.int32), axis=-1
  )
  box_right = tf.reduce_sum(
      tf.cast(tf.cumsum(horizontal, axis=-1, reverse=True) > 0, tf.int32),
      axis=-1,
  )

  h = tf.cast(sparse_segmentations.shape[-2], tf.float32)
  w = tf.cast(sparse_segmentations.shape[-1], tf.float32)
  boxes = tf.stack(
      [
          tf.cast(box_top, tf.float32) / h,
          tf.cast(box_left, tf.float32) / w,
          tf.cast(box_bottom, tf.float32) / h,
          tf.cast(box_right, tf.float32) / w,
      ],
      axis=-1,
  )

  # Special case: Empty sparse_segmentations result in [1, 1, 0, 0] but should
  # be NOTRACK_BOX.
  boxes = tf.where(
      boxes == tf.constant([1, 1, 0, 0], tf.float32),
      tf.constant(NOTRACK_BOX, tf.float32),
      boxes,
  )

  return boxes


def boxes_to_sparse_segmentations(
    boxes: tf.Tensor, height: int, width: int
) -> tf.Tensor:
  """Converts bounding boxes into sparse segmentations.

  Args:
    boxes: A bounding box tensor of shape [..., N, 4] in TF format (i.e.,
      normalized coordinates [ymin, xmin, ymax, xmax] in [0, 1]^4.
    height: The frame height.
    width: The frame width.

  Returns:
    A sparse segmentations tensor of shape [..., N, H, W].
  """
  batch_shape = boxes.shape[:-2]
  n = boxes.shape[-2]
  boxes = tf.reshape(boxes, (-1, n, 4))
  # Convert the normalized into absolute coordinates.
  boxes_absolute = tf.cast(
      tf.round(
          boxes
          * tf.constant([height, width, height, width], tf.float32)[
              tf.newaxis, tf.newaxis
          ]
      ),
      tf.int32,
  )
  ymin, xmin, ymax, xmax = tf.split(boxes_absolute, 4, axis=-1)
  # Generate yx-grid and mask the boxes.
  grid_y = tf.tile(
      tf.range(height)[tf.newaxis, tf.newaxis, :, tf.newaxis],
      boxes_absolute.shape[:2] + (1, width),
  )
  grid_x = tf.tile(
      tf.range(width)[tf.newaxis, tf.newaxis, tf.newaxis, :],
      boxes_absolute.shape[:2] + (height, 1),
  )
  sparse_segmentations = tf.logical_and(
      tf.logical_and(
          grid_y >= ymin[..., tf.newaxis], grid_y < ymax[..., tf.newaxis]
      ),
      tf.logical_and(
          grid_x >= xmin[..., tf.newaxis], grid_x < xmax[..., tf.newaxis]
      ),
  )
  sparse_segmentations = tf.cast(
      tf.reshape(sparse_segmentations, batch_shape + (n, height, width)),
      tf.uint8,
  )
  return sparse_segmentations


def _get_padding_bboxes(
    num_bboxes: int, is_2d: bool = True, is_proj_4_corner: bool = False
) -> tf.Tensor:
  """Prepare placeholder bboxes for `tf.where` call."""
  if is_2d:
    pad_box = NOTRACK_BOX
  else:
    if is_proj_4_corner:
      pad_box = NOTRACK_PROJ_4_CORNER
    else:
      pad_box = NOTRACK_BOX_3D
  return tf.tile(
      tf.constant(pad_box, tf.float32)[tf.newaxis, :],
      multiples=(num_bboxes, 1),
  )  # [N, 4/5/10/12]


def _compute_bbox_padding(bboxes: tf.Tensor) -> tf.Tensor:
  """Get bbox padding mask from bboxes."""
  # True: real bbox; False: padded bbox
  return tf.reduce_any(
      tf.not_equal(bboxes, tf.constant(NOTRACK_BOX, tf.float32)), -1
  )  # [*N]


def compute_bbox_areas(bboxes: tf.Tensor) -> tf.Tensor:
  """Computes aspect ratio and area of bboxes."""
  y0, x0, y1, x1 = tf.split(bboxes, num_or_size_splits=4, axis=-1)  # [*N, 1]
  h = y1 - y0
  w = x1 - x0
  areas = h * w
  return areas  # [*N, 1]


def sample_idx_within_range(max_num, min_interval, max_interval):
  """Sample two idx within [0, max_num] and desired interval."""
  delta_t = tf.random.get_global_generator().uniform(
      (),
      minval=min_interval,
      maxval=max_interval + 1,
      dtype=tf.int32,
  )
  t1 = tf.random.get_global_generator().uniform(
      (), minval=0, maxval=max_num - delta_t, dtype=tf.int32
  )
  t2 = t1 + delta_t
  do_swap = tf.random.get_global_generator().uniform(()) < 0.5
  final_t1 = tf.where(do_swap, t2, t1)
  final_t2 = tf.where(do_swap, t1, t2)
  return final_t1, final_t2


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GetFramesWithMatchingObjects:
  """Given a sequence of video frames, sample two frames and match objects.

  This is used to create training pairs for conditional generation. Therefore,
  the order of objects should be consistent across frames. E.g., `bboxes[0]` in
  frame1 should represent the same object as `bboxes[0]` in frame2.
  This is usually guaranteed when the input bboxes/masks have a time dimension.
  """

  seq_len: int
  min_interval: int = 1
  max_interval: int = -1
  relevant_keys: Sequence[str] = ('video', 'bboxes')  # data with a time dim
  # For each `key` here, we will return a `src_key` field and a `tgt_key` field
  t1_key: Optional[str] = None
  t2_key: Optional[str] = None
  # We might use other functions to sample idx e.g. finding big paired objects

  def __post_init__(self):
    assert self.seq_len > 1, f'{self.seq_len=} must be greater than 1'
    assert self.min_interval >= 0, f'{self.min_interval=} must be non-negative'
    assert (
        self.min_interval < self.seq_len
    ), f'{self.min_interval=} must be less than {self.seq_len=}'
    if self.max_interval < 0:
      object.__setattr__(self, 'max_interval', self.seq_len - 1)
    assert (self.t1_key is None and self.t2_key is None) or (
        self.t1_key is not None and self.t2_key is not None
    ), (
        f'{self.t1_key=} and {self.t2_key=} must be None or specified at the'
        ' same time'
    )

  def __call__(self, features: Features) -> Features:
    if self.t1_key is not None and self.t2_key is not None:
      t1 = features[self.t1_key]
      t2 = features[self.t2_key]
    else:
      t1, t2 = sample_idx_within_range(
          max_num=self.seq_len,
          min_interval=self.min_interval,
          max_interval=self.max_interval,
      )
    for key in self.relevant_keys:
      # We assume time is the first dimension
      features[f'src_{key}'] = features[key][t1]
      features[f'tgt_{key}'] = features[key][t2]
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MultiSegmentationsToBoxes:
  """Adds a bbox feature derived from segmentation masks.

  The mask can be either sparse segmentation or dense segmentation.
  """

  is_dense_segmentation: bool = True
  segmentations_key: str = 'segmentations'
  boxes_key: str = 'bboxes'
  boxes_padding_key: str = 'bboxes_padding'
  max_instances: int = 100

  def __call__(self, features: Features) -> Features:
    if self.is_dense_segmentation:
      dense_segmentations = features[self.segmentations_key]  # [..., H, W]
      sparse_segmentations = dense_to_sparse_segmentations(
          dense_segmentations, self.max_instances
      )  # [..., max_instances, H, W], uint8
    else:
      sparse_segmentations = features[self.segmentations_key]
    bboxes = sparse_segmentations_to_boxes(sparse_segmentations)
    features[self.boxes_key] = bboxes
    features[self.boxes_padding_key] = _compute_bbox_padding(bboxes)

    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class BoxesToForegroundMask:
  """Convert object bboxes to a boolean foreground mask."""

  image_size: tuple[int, int]
  boxes_key: str = 'bboxes'
  fg_mask_key: str = 'fg_mask'

  def __call__(self, features: Features) -> Features:
    h, w = self.image_size
    bboxes = features[self.boxes_key]  # [..., N, 4]
    sparse_segmentations = boxes_to_sparse_segmentations(
        bboxes, height=h, width=w
    )  # [..., N, H, W]
    fg_mask = tf.reduce_any(tf.greater(sparse_segmentations, 0), axis=-3)
    features[self.fg_mask_key] = fg_mask  # [..., H, W]
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MaskOutImage:
  """Use a mask to mask out certain regions of an image."""

  image_key: str = 'image'
  mask_out_image_key: str = 'mask_out_image'
  mask_key: str = 'segmentations'
  background_value: float = 0.5

  def __call__(self, features: Features) -> Features:
    image = features[self.image_key]  # [..., H, W, C]
    mask = features[self.mask_key]  # [..., H, W]
    mask = mask[..., tf.newaxis]  # [..., H, W, 1]
    image = tf.where(mask, tf.ones_like(image) * self.background_value, image)
    features[self.mask_out_image_key] = image
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class AddGlobalBox:
  """Appends a global bbox that covers the whole image to existing bboxes.

  NOTE: The global bbox is appended as the last bbox for each sample.
  """

  boxes_key: str = 'bboxes'
  boxes_padding_key: str = 'bboxes_padding'
  boxes_3d_key: Optional[str] = None
  is_proj_4_corner: bool = False
  is_camera_pose: bool = False

  def __post_init__(self):
    assert not (self.is_camera_pose and self.is_proj_4_corner)

  def __call__(self, features: Features) -> Features:
    bboxes = features[self.boxes_key]  # [N, 4/5]
    global_bbox = (0.0, 0.0, 1.0, 1.0)  # (y0, x0, y1, x1) normalized
    global_bbox = tf.constant(global_bbox, tf.float32)[tf.newaxis, :]
    bboxes = tf.concat([bboxes, global_bbox], axis=0)  # [N+1, 4/5]
    features[self.boxes_key] = bboxes
    features[self.boxes_padding_key] = _compute_bbox_padding(bboxes)
    # May also need to append a 3D bbox
    if self.boxes_3d_key is not None:
      bboxes_3d = features[self.boxes_3d_key]  # [N, 10/12]
      if self.is_camera_pose:
        identity_rmat = np.eye(3, dtype=np.float32).reshape((-1,))
        zero_trans = np.zeros(3, dtype=np.float32)
        dummy_bbox_3d = np.concatenate([identity_rmat, zero_trans], axis=0)
      elif self.is_proj_4_corner:
        dummy_bbox_3d = NOTRACK_PROJ_4_CORNER
      else:
        dummy_bbox_3d = NOTRACK_BOX_3D
      dummy_bbox_3d = tf.constant(dummy_bbox_3d, tf.float32)[tf.newaxis, :]
      bboxes_3d = tf.concat([bboxes_3d, dummy_bbox_3d], axis=0)  # [N+1, 10]
      features[self.boxes_3d_key] = bboxes_3d
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class AddGlobal3DBoxFromCameraPose:
  """Appends relative camera pose between 2 frames as background 3D bbox."""

  boxes_3d_key: str = 'bboxes_3d'  # should be in proj_4_corner format!
  src_camera2world_key: Optional[str] = None
  tgt_camera2world_key: Optional[str] = None

  def __post_init__(self):
    assert (
        self.src_camera2world_key is None and self.tgt_camera2world_key is None
    ) or (
        self.src_camera2world_key is not None
        and self.tgt_camera2world_key is not None
    )

  def _get_rel_camera_pose(self, features):
    """Gets RT transformation between 2 frames **viewed from src frame**."""
    if self.src_camera2world_key is None:
      rmat, trans = tf.eye(3, dtype=tf.float32), tf.zeros(3, dtype=tf.float32)
      return tf.concat([tf.reshape(rmat, (-1,)), trans], axis=0)  # [12]
    src_camera2world = features[self.src_camera2world_key]  # [4, 4]
    tgt_camera2world = features[self.tgt_camera2world_key]  # [4, 4]
    # Get src_camera2tgt_camera
    tgt_world2camera = tf.linalg.inv(tgt_camera2world)
    src_camera2tgt_camera = tgt_world2camera @ src_camera2world
    rmat, trans = src_camera2tgt_camera[:3, :3], src_camera2tgt_camera[:3, 3]
    return tf.concat([tf.reshape(rmat, (-1,)), trans], axis=0)  # [12]

  def __call__(self, features: Features) -> Features:
    bboxes_3d = features[self.boxes_3d_key]  # [N, 12]
    global_bbox_3d = self._get_rel_camera_pose(features)
    bboxes_3d = tf.concat([bboxes_3d, global_bbox_3d[tf.newaxis]], axis=0)
    features[self.boxes_3d_key] = bboxes_3d
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class RandomDropBoxes:
  """Drop bboxes randomly for classifier-free guidance at inference time."""

  max_instances: int
  drop_prob: float = 0.1
  boxes_key: str = 'bboxes'
  boxes_padding_key: str = 'bboxes_padding'
  boxes_3d_key: Optional[str] = None
  is_proj_4_corner: bool = False

  def _get_dummy_boxes(self):
    """Get dummy boxes."""
    dummy_bboxes = _get_padding_bboxes(self.max_instances)
    dummy_bboxes_3d = _get_padding_bboxes(
        self.max_instances,
        is_2d=False,
        is_proj_4_corner=self.is_proj_4_corner,
    )
    return dummy_bboxes, dummy_bboxes_3d

  def _drop_boxes(self, features: Features) -> Features:
    """Drop boxes by replacing them with dummy ones."""
    dummy_bboxes, dummy_bboxes_3d = self._get_dummy_boxes()
    features[self.boxes_key] = dummy_bboxes
    features[self.boxes_padding_key] = tf.ones(
        (self.max_instances,), dtype=tf.bool
    )
    if self.boxes_3d_key is not None:
      features[self.boxes_3d_key] = dummy_bboxes_3d
    return features

  def __call__(self, features: Features) -> Features:
    do_drop = tf.random.get_global_generator().uniform(()) < self.drop_prob
    # Need this copy to avoid a weird bug of `tf.cond`
    ori_features = {k: v for k, v in features.items()}
    features = tf.cond(
        do_drop, lambda: self._drop_boxes(features), lambda: ori_features
    )
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class GetMOViCamera2World:
  """Get MOVi camera2world matrix from camera position and rotation."""

  camera_positions_key: str = 'camera_positions'
  camera_quats_key: str = 'camera_quats'
  camera2world_key: str = 'camera2world'

  def __call__(self, features: Features) -> Features:
    cam_pos = features[self.camera_positions_key]  # [T, 3]
    cam_quat = features[self.camera_quats_key]  # [T, 4]
    cam2world = utils_3d.batch_get_matrix_world(cam_pos, cam_quat)  # [T,4,4]
    features[self.camera2world_key] = cam2world
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MOVi3DBboxWorldToCamera:
  """Convert MOVi 3D bbox from world coordinate to camera coordinate."""

  seq_len: int
  bboxes_3d_key: str = 'bboxes_3d'
  bboxes_quats_key: str = 'bboxes_quats'
  camera_positions_key: str = 'camera_positions'
  camera_quats_key: str = 'camera_quats'

  def __call__(self, features: Features) -> Features:
    bboxes_3d = features[self.bboxes_3d_key]  # [T, N, 8, 3]
    bboxes_quats = features[self.bboxes_quats_key]  # [T, N, 4]
    camera_bboxes_3d, camera_bboxes_quats = utils_3d.batch_world2camera(
        features[self.camera_positions_key],  # [T, 3]
        features[self.camera_quats_key],  # [T, 4]
        tf.reshape(bboxes_3d, (self.seq_len, -1, 3)),  # [T, (N*8), 3]
        bboxes_quats,
    )
    camera_bboxes_3d = tf.reshape(camera_bboxes_3d, (self.seq_len, -1, 8, 3))
    features[self.bboxes_3d_key] = camera_bboxes_3d
    features[self.bboxes_quats_key] = camera_bboxes_quats
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MOVi3DBboxToVec:
  """Convert MOVi 3D bbox to (centers, sizes, quaternions) format."""

  seq_len: int
  max_instances: int
  bboxes_3d_key: str = 'bboxes_3d'
  bboxes_quats_key: str = 'bboxes_quats'

  def __call__(self, features: Features) -> Features:
    bboxes_3d = features[self.bboxes_3d_key]  # [T, n, 8, 3]
    bboxes_quats = features[self.bboxes_quats_key]  # [T, n, 4]
    centers, sizes, quats = utils_3d.batch_decompose_3d_bboxes(
        bboxes_3d, bboxes_quats
    )
    bboxes_3d = tf.concat([centers, sizes, quats], axis=-1)  # [T, n, 10]
    # Pad to a fixed number
    bboxes_3d = bboxes_3d[:, : self.max_instances]
    pad_num = self.max_instances - tf.shape(bboxes_3d)[1]
    pad_bboxes_3d = tf.tile(
        tf.constant(NOTRACK_BOX_3D, dtype=tf.float32)[None, None, :],
        multiples=(self.seq_len, pad_num, 1),
    )  # [T, N-n, 10]
    bboxes_3d = tf.concat([bboxes_3d, pad_bboxes_3d], axis=1)  # [T, N, 10]
    bboxes_3d.set_shape((self.seq_len, self.max_instances, 10))
    features[self.bboxes_3d_key] = bboxes_3d
    return features


@dataclasses.dataclass(kw_only=True, frozen=True, eq=True)
class MOVi3DBboxCornerToImage:
  """Project MOVi 3D bbox corners (p0,1,2,4) to image coords + depth.

  This makes the image coords to have a value range in [0, 1].
  While the depth is still in [0, inf], it is proportional to bbox size.
  3D bbox can be either 8x3 corner format or 10-D vector format.
  """

  seq_len: int
  max_instances: int
  image_size: tuple[int, int]
  bboxes_3d_key: str = 'bboxes_3d'
  is_corner_format: bool = True  # if in [N, 8, 3] format, otherwise is [N, 10]
  focal_length_key: Optional[str] = None
  sensor_width_key: Optional[str] = None
  intrinsic_key: Optional[str] = None

  def __post_init__(self):
    if self.focal_length_key is not None:
      assert self.sensor_width_key is not None
      assert self.intrinsic_key is None
    if self.sensor_width_key is not None:
      assert self.focal_length_key is not None
      assert self.intrinsic_key is None
    if self.intrinsic_key is not None:
      assert self.focal_length_key is None
      assert self.sensor_width_key is None

  def __call__(self, features: Features) -> Features:
    bboxes_3d = features[self.bboxes_3d_key]  # [T, n, 8, 3]
    if not self.is_corner_format:
      centers, sizes, quats = tf.split(bboxes_3d, [3, 3, 4], axis=-1)
      bboxes_3d = utils_3d.batch_construct_3d_bboxes(centers, sizes, quats)
    h, w = self.image_size
    focal_length = (
        features[self.focal_length_key]
        if self.focal_length_key is not None
        else None
    )
    sensor_width = (
        features[self.sensor_width_key]
        if self.sensor_width_key is not None
        else None
    )
    intrinsic = (
        features[self.intrinsic_key] if self.intrinsic_key is not None else None
    )
    # We take the following 4 corners, which fully specify a 3D bbox
    # p0 = bboxes_3d[:, :, 0, :]  # [T, n, 3], (x0, y0, z0)
    # p1 = bboxes_3d[:, :, 1, :]  # [T, n, 3], (x0, y0, z1)
    # p2 = bboxes_3d[:, :, 2, :]  # [T, n, 3], (x0, y1, z0)
    # p4 = bboxes_3d[:, :, 4, :]  # [T, n, 3], (x1, y0, z0)
    # In theory, 3 corners are enough for a 3D bbox as it gives 9DoF
    # Using 4 corners make conversion back to 8 corners format much easier
    # Also, people have over-parameterized representations for e.g. rotation:
    # https://arxiv.org/abs/1812.07035
    proj_coords = []
    for i in [0, 1, 2, 4]:
      corner = bboxes_3d[:, :, i, :]  # [T, n, 3]
      image_coords, depth = utils_3d.batch_camera2image(
          point3d=corner,  # [T, n, 3]
          intrinsics=intrinsic,
          focal_length=focal_length,
          sensor_width=sensor_width,
          width=w,
          height=h,
      )  # [T, n, 3], [T, n, 1]
      proj_coords.append(image_coords[:, :, :2])
      proj_coords.append(depth)
    bboxes_3d = tf.concat(proj_coords, axis=-1)  # [T, n, (2+1)*4=12]
    # Pad to a fixed number
    bboxes_3d = bboxes_3d[:, : self.max_instances]
    pad_num = self.max_instances - tf.shape(bboxes_3d)[1]
    pad_bboxes_3d = tf.tile(
        tf.constant(NOTRACK_PROJ_4_CORNER, dtype=tf.float32)[None, None, :],
        multiples=(self.seq_len, pad_num, 1),
    )  # [T, N-n, 12]
    bboxes_3d = tf.concat([bboxes_3d, pad_bboxes_3d], axis=1)  # [T, N, 12]
    bboxes_3d.set_shape((self.seq_len, self.max_instances, 12))
    features[self.bboxes_3d_key] = bboxes_3d
    return features


def preprocess_gv_movi_example(
    batch: Features,
    max_instances: int,
    resolution: int = 256,
    drop_cond_prob: float = 0.1,
):
  """Extract necessary fields from gv MOVi data batch."""
  # Video and object masks
  video = batch['video']  # [T, H, W, 3]
  video = tf.cast(video, np.float32) / 255.0
  video = tf.image.resize(video, (resolution, resolution), method='area')
  masks = batch['segmentations']  # [T, H, W, 1]
  masks = tf.image.resize(masks, (resolution, resolution), method='nearest')
  masks = tf.cast(masks[..., 0], np.int32)  # [T, H, W]
  # 3D bboxes
  bboxes_3d = batch['instances']['bboxes_3d']  # [N, T, 8, 3]
  bboxes_3d = einops.rearrange(bboxes_3d, pattern='n t ... -> t n ...')
  bboxes_quats = batch['instances']['quaternions']  # [N, T, 4]
  bboxes_quats = einops.rearrange(bboxes_quats, pattern='n t ... -> t n ...')
  batch = {
      'image': video,  # [T, H, W, 3]
      'segmentations': masks,  # [T, H, W]
      'bboxes_3d': bboxes_3d,  # [T, N, 8, 3]
      'bboxes_3d_quats': bboxes_quats,  # [T, N, 4]
      'camera_focal_length': batch['camera']['focal_length'],  # float
      'camera_sensor_width': batch['camera']['sensor_width'],  # float
      'camera_positions': batch['camera']['positions'],  # [T, 3]
      'camera_quaternions': batch['camera']['quaternions'],  # [T, 4]
  }
  # Get camera2world matrix
  fn = GetMOViCamera2World(
      camera_positions_key='camera_positions',
      camera_quats_key='camera_quaternions',
      camera2world_key='camera2world',
  )
  batch = fn(batch)  # `camera2world`: [T, 4, 4]
  # Transform 3D bboxes to camera coordinate
  fn = MOVi3DBboxWorldToCamera(
      seq_len=24,
      bboxes_3d_key='bboxes_3d',
      bboxes_quats_key='bboxes_3d_quats',
      camera_positions_key='camera_positions',
      camera_quats_key='camera_quaternions',
  )
  batch = fn(batch)
  # From corner format [T, N, 8, 3] to (center, size, quat) format [T, N, 10]
  fn = MOVi3DBboxToVec(
      seq_len=24,
      max_instances=max_instances,
      bboxes_3d_key='bboxes_3d',
      bboxes_quats_key='bboxes_3d_quats',
  )
  batch = fn(batch)
  # Object masks to 2D bboxes
  fn = MultiSegmentationsToBoxes(
      segmentations_key='segmentations',
      boxes_key='bboxes',  # [T, N, 4], (y0, x0, y1, x1) normalized
      boxes_padding_key='bboxes_padding',  # [T, N]
      max_instances=max_instances,
  )
  batch = fn(batch)
  # Gather results so far
  batch = {
      'image': batch['image'],  # [T, H, W, 3]
      'bboxes': batch['bboxes'],  # [T, N, 4]
      'bboxes_3d': batch['bboxes_3d'],  # [T, N, 10]
      'camera_focal_length': batch['camera_focal_length'],  # float
      'camera_sensor_width': batch['camera_sensor_width'],  # float
      'camera2world': batch['camera2world'],  # [T, 4, 4]
  }
  # Sample two frames from the video
  sample_keys = ('image', 'bboxes', 'bboxes_3d', 'camera2world')
  fn = GetFramesWithMatchingObjects(
      seq_len=24, min_interval=0, relevant_keys=sample_keys
  )
  batch = fn(batch)
  for k in sample_keys:
    _ = batch.pop(k)
  # Project 3D bbox corners to the image plane as our pose representation
  fn = MOVi3DBboxCornerToImage(
      seq_len=1,
      max_instances=max_instances,
      image_size=(resolution, resolution),
      bboxes_3d_key='bboxes_3d',
      is_corner_format=False,
      focal_length_key='camera_focal_length',
      sensor_width_key='camera_sensor_width',
  )
  src_batch = {
      'bboxes_3d': batch['src_bboxes_3d'][None],
      'camera_focal_length': batch['camera_focal_length'],
      'camera_sensor_width': batch['camera_sensor_width'],
  }
  batch['src_bboxes_3d'] = fn(src_batch)['bboxes_3d'][0]
  tgt_batch = {
      'bboxes_3d': batch['tgt_bboxes_3d'][None],
      'camera_focal_length': batch['camera_focal_length'],
      'camera_sensor_width': batch['camera_sensor_width'],
  }
  batch['tgt_bboxes_3d'] = fn(tgt_batch)['bboxes_3d'][0]
  # Mask out objects using foreground mask to create background images
  fn = BoxesToForegroundMask(
      image_size=(resolution, resolution),
      boxes_key='bboxes',
      fg_mask_key='fg_mask',
  )
  fg_mask = fn({'bboxes': batch['src_bboxes']})['fg_mask']  # [H, W]
  fn = MaskOutImage(
      image_key='image',
      mask_out_image_key='bg_image',
      mask_key='fg_mask',
      background_value=0.5,
  )
  bg_image = fn({'image': batch['src_image'], 'fg_mask': fg_mask})['bg_image']
  batch['src_bg_image'] = bg_image  # [H, W, 3]
  # Append a bbox covering the whole image
  # This will increase the number of bboxes by 1
  fn = AddGlobalBox(boxes_key='bboxes')
  batch['src_bboxes'] = fn({'bboxes': batch['src_bboxes']})['bboxes']
  batch['src_bboxes_padding'] = _compute_bbox_padding(batch['src_bboxes'])
  batch['tgt_bboxes'] = fn({'bboxes': batch['tgt_bboxes']})['bboxes']
  # Add global 3D bbox based on camera poses
  # This will increase the number of 3D bboxes by 1
  fn = AddGlobal3DBoxFromCameraPose(
      boxes_3d_key='src_bboxes_3d'
  )  # src frame has identity pose
  batch = fn(batch)
  fn = AddGlobal3DBoxFromCameraPose(
      boxes_3d_key='tgt_bboxes_3d',
      src_camera2world_key='src_camera2world',
      tgt_camera2world_key='tgt_camera2world',
  )  # tgt frame uses relative camera pose
  batch = fn(batch)
  _ = batch.pop('src_camera2world')
  _ = batch.pop('tgt_camera2world')
  # Randomly drop conditioning tokens (2D/3D bboxes) in training for CFG
  # Only src 2D bbox and tgt 3D bbox are used for conditioning
  if drop_cond_prob > 0:
    fn = RandomDropBoxes(
        # +1 because we add a global background bbox
        max_instances=max_instances + 1,
        drop_prob=drop_cond_prob,
        boxes_key='src_bboxes',
        boxes_padding_key='src_bboxes_padding',
        boxes_3d_key='tgt_bboxes_3d',
        is_proj_4_corner=True,
    )
    batch = fn(batch)
  return batch
