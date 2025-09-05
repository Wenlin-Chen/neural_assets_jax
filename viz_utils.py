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

"""Visualization utilities for Neural Assets."""

from __future__ import annotations

import os
from typing import Optional

import cv2
import einops
import mediapy as media
import numpy as np
import preprocessing
import tensorflow as tf
import utils_3d


def to_numpy(data):
  if isinstance(data, tf.Tensor):
    return data.numpy()
  elif isinstance(data, (list, tuple)):
    return type(data)([to_numpy(d) for d in data])
  elif isinstance(data, dict):
    return type(data)({k: to_numpy(v) for k, v in data.items()})
  else:
    raise ValueError(f'Unsupported data type: {type(data)}')


def deepcopy_np(array):
  return np.array(array).copy() if array is not None else None


def convert4vis(images):
  """Convert images to np.ndarray in uint8 for visualization."""
  images = np.array(images)
  if images.dtype == np.uint8:
    return images
  images = np.round(images.clip(0, 1) * 255.0).astype(np.uint8)
  return images


def show_video(video, height=None, fps=8, codec='gif', save_path=None):
  """Show a video and potentially save it."""
  # video: [T, H, W, C]
  media.show_video(video, height=height, fps=fps, codec=codec)
  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    media.write_video(save_path, video, fps=fps, codec=codec)


def show_images(images, save_path=None):
  """Show images and potentially save it."""
  # images: [N, H, W, C]
  images = einops.rearrange(images, 'n h w c -> h (n w) c')
  media.show_image(images, height=images.shape[-3])
  if save_path is not None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
  media.write_image(save_path, images, fmt='png')


def draw_bbox(
    image_lst, bboxes_lst, scores_lst=None, show=True, save_path=None, **kwargs
):
  """Draw bboxes on images."""
  image_lst = [convert4vis(image) for image in image_lst]
  if scores_lst is None:
    scores_lst = [None] * len(bboxes_lst)
  bbox_images = np.stack([
      show_bbox_on_image(
          image=deepcopy_np(image),
          bboxes=deepcopy_np(bboxes),
          scores=deepcopy_np(scores),
          **kwargs,
      )
      for image, bboxes, scores in zip(image_lst, bboxes_lst, scores_lst)
  ])
  if show:
    show_images(bbox_images, save_path=save_path)
  else:
    return bbox_images


def draw_bbox_3d(
    image_lst,
    bboxes3d_lst,
    focal_length_lst=None,
    sensor_width_lst=None,
    camera2image_lst=None,
    show=True,
    is_proj_4_corner=False,
    has_background_bbox=False,
    save_path=None,
    **kwargs,
):
  """Draw bboxes on images."""
  images = np.stack([convert4vis(image) for image in image_lst], axis=0).copy()
  bboxes3d = deepcopy_np(bboxes3d_lst)
  focal_length = deepcopy_np(focal_length_lst)
  sensor_width = deepcopy_np(sensor_width_lst)
  camera2image = deepcopy_np(camera2image_lst)
  bbox3d_images = batch_show_3d_bbox_on_image(
      images=images,
      bboxes=bboxes3d,
      focal_length=focal_length,
      sensor_width=sensor_width,
      intrinsics=camera2image,
      is_proj_4_corner=is_proj_4_corner,
      has_background_bbox=has_background_bbox,
      **kwargs,
  )
  if show:
    show_images(bbox3d_images, save_path=save_path)
  else:
    return bbox3d_images


def show_bbox_on_image(
    image: np.ndarray,  # [H, W, C]
    bboxes: np.ndarray,  # [N, 4]
    scores: Optional[np.ndarray] = None,  # [N]
    bbox_color: tuple[int, int, int] = (0, 255, 0),  # Green
    score_color: tuple[int, int, int] = (0, 0, 0),  # Black
    line_thickness=1,
) -> np.ndarray:
  """Draws object bboxes on an image, potentially with detection scores."""
  assert image.dtype == np.uint8, 'Image must be uint8'
  assert bboxes.min() >= 0 and bboxes.max() <= 1, 'Bboxes must be normalized'
  h, w = image.shape[0], image.shape[1]
  for box_idx, box in enumerate(bboxes):  # (y0, x0, y1, x1) normalized
    if (box == np.array(preprocessing.NOTRACK_BOX)).all():
      continue
    pt1 = (round(box[1] * w), round(box[0] * h))
    pt2 = (round(box[3] * w), round(box[2] * h))
    cv2.rectangle(image, pt1, pt2, color=bbox_color, thickness=line_thickness)
    if scores is not None:
      score = scores[box_idx]
      cv2.putText(
          image,
          f'{score:.2f}',
          (pt1[0], pt1[1] - 1),
          cv2.FONT_HERSHEY_SIMPLEX,
          fontScale=0.5,
          color=score_color,
          thickness=1,
      )
  return image


def batch_show_3d_bbox_on_image(
    images: np.ndarray,  # [B, H, W, C]
    bboxes: np.ndarray,  # [B, N, 3], (xc, yc, zc)
    # `bboxes` can also be:
    # - [B, N, 10]: concat of (center, size, quat)
    # - [B, N, 12]: 4 projected corners (p0, p1, p2, p4)
    bboxes_size: Optional[np.ndarray] = None,  # [B, N, 3], (w, h, d)
    bboxes_quat: Optional[np.ndarray] = None,  # [B, N, 4], quaternion
    focal_length: Optional[np.ndarray] = None,  # [B]
    sensor_width: Optional[np.ndarray] = None,  # [B]
    intrinsics: Optional[np.ndarray] = None,  # [B, 3, 3]
    bbox_color: tuple[int, int, int] = (0, 255, 0),  # Green
    is_proj_4_corner: bool = False,  # `bboxes` is 4 projected corners
    has_background_bbox: bool = False,  # skip the last `bboxes` which is bg
) -> np.ndarray:
  """Project 3D bboxes to 2D images."""
  assert images.dtype == np.uint8, 'Image must be uint8'
  _, h, w, _ = images.shape
  if is_proj_4_corner:
    # Projected p0, p1, p2, p4
    assert bboxes.shape[2] == 12, f'Invalid {bboxes.shape=}'
    assert bboxes_size is None and bboxes_quat is None
    unproj_corners = []
    for i in range(4):
      # bboxes[0:2] is p0's projected image coords, bboxes[2:3] is its depth
      # bboxes[3:5] is p1's projected image coords, bboxes[5:6] is its depth
      # ...
      corner = utils_3d.batch_image2camera(
          image_coords=bboxes[..., i * 3 : i * 3 + 2],
          depth=bboxes[..., i * 3 + 2 : i * 3 + 3],
          width=w,
          height=h,
          focal_length=focal_length,
          sensor_width=sensor_width,
          intrinsics=intrinsics,
      )
      unproj_corners.append(corner)
    p0 = unproj_corners[0]
    p1 = unproj_corners[1]
    p2 = unproj_corners[2]
    p4 = unproj_corners[3]
    bboxes_3d = utils_3d.construct_3d_bboxes_from_p0124(p0, p1, p2, p4)
  else:
    if bboxes_size is None or bboxes_quat is None:
      assert bboxes_size is None and bboxes_quat is None
      assert bboxes.shape[2] == 10
      bboxes_center, bboxes_size, bboxes_quat = np.split(bboxes, (3, 6), axis=2)
    else:
      bboxes_center = bboxes
    bboxes_3d = utils_3d.construct_3d_bboxes(
        centers=bboxes_center, sizes=bboxes_size, quaternions=bboxes_quat
    )
  # bboxes_3d: [B, N, 8, 3]
  if has_background_bbox:
    # NOTE: we assume that background bbox is always concatenated at the end
    # We skip it in visualization
    bboxes_3d = bboxes_3d[:, :-1, :, :]
  camera_dict = {
      'focal_length': focal_length,
      'sensor_width': sensor_width,
      'intrinsic': intrinsics,
  }
  images = show_3d_bbox_on_image(
      images=images,
      bboxes_3d=bboxes_3d,
      cameras=camera_dict,
      bbox_color=bbox_color,
      is_world_coord=False,  # assume already in camera coords
  )
  return images


def draw_projected_3d_bbox(
    image,
    proj_corners,
    proj_centers=None,
    line_color=(0, 255, 0),
    pt_color=(255, 0, 0),
    line_thickness=1,
):
  """Draw a projected 3d bbox on a 2d image."""
  # image: [H, W, C]
  # proj_corners: [N, 8, 3]
  # proj_centers: [N, 3]
  assert image.dtype == np.uint8, 'Image must be uint8'
  h, w, _ = image.shape
  # Note that projected points are in (x, y, z) not (y, x, z)!
  assert (proj_corners[:, :, 2] == 1.0).all(), '`proj_corners` not Z-normalized'
  proj_corners = proj_corners[:, :, :2]  # we only need (x, y)
  if proj_centers is not None:
    assert (proj_centers[:, 2] == 1.0).all(), '`proj_centers` not Z-normalized'
    proj_centers = proj_centers[:, :2]
  corner_pairs = (
      (0, 1),
      (0, 2),
      (2, 3),
      (1, 3),
      (4, 5),
      (4, 6),
      (6, 7),
      (5, 7),
      (0, 4),
      (1, 5),
      (2, 6),
      (3, 7),
  )
  for i, proj_pt in enumerate(proj_corners):  # [8, 2]
    if (proj_pt == 0.0).all():
      continue
    for corner_pair in corner_pairs:
      pt1, pt2 = proj_pt[corner_pair[0]], proj_pt[corner_pair[1]]
      pt1 = (round(pt1[0] * w), round(pt1[1] * h))
      pt2 = (round(pt2[0] * w), round(pt2[1] * h))
      cv2.line(image, pt1, pt2, color=line_color, thickness=line_thickness)
    if proj_centers is not None:
      pt = proj_centers[i]
      pt = (round(pt[0] * w), round(pt[1] * h))
      cv2.circle(image, pt, h // 100, color=pt_color, thickness=-1)
  return image


def get_camera_dict(cameras, frame_idx):
  """Create a camera dict for visualization from Kubric returned `cameras`."""
  # Basic camera parameters may be shared across frames
  camera_dict = {
      'focal_length': None,
      'sensor_width': None,
      'intrinsic': None,
  }
  focal_length = cameras.get('focal_length', None)
  if focal_length is not None:
    if isinstance(focal_length, (int, float)) or not focal_length.shape:
      camera_dict['focal_length'] = float(focal_length)
    else:
      camera_dict['focal_length'] = focal_length[frame_idx]
  sensor_width = cameras.get('sensor_width', None)
  if sensor_width is not None:
    if isinstance(sensor_width, (int, float)) or not sensor_width.shape:
      camera_dict['sensor_width'] = float(sensor_width)
    else:
      camera_dict['sensor_width'] = sensor_width[frame_idx]
  intrinsic = cameras.get('intrinsic', None)
  if intrinsic is not None:
    if len(intrinsic.shape) == 2:  # 3x3 or 4x4
      camera_dict['intrinsic'] = intrinsic
    else:
      assert len(intrinsic.shape) == 3, f'Invalid {intrinsic.shape=}'
      camera_dict['intrinsic'] = intrinsic[frame_idx]
  # Camera pose may not needed if points are already in camera coordinate
  # If needed, it should be specified per frame
  if 'positions' in cameras:
    camera_dict['position'] = cameras['positions'][frame_idx]
  if 'quaternions' in cameras:
    camera_dict['quaternion'] = cameras['quaternions'][frame_idx]
  return camera_dict


def show_3d_bbox_on_image(
    images,
    bboxes_3d,
    cameras,
    bboxes_center_3d=None,
    bbox_color=(0, 255, 0),  # Green
    center_color=(255, 0, 0),  # Red
    is_world_coord=True,
):
  """Project 3D bboxes to 2D, then show them on an image."""
  # images: [B, H, W, C]
  # bboxes_3d: [B, N, 8, 3]
  # cameras: a dict with camera metadata
  # bboxes_center_3d: [B, N, 3]
  b, h, w, _ = images.shape
  for i in range(b):
    image = images[i]  # [H, W, C]
    bbox_3d = bboxes_3d[i]  # [N, 8, 3]
    camera = get_camera_dict(cameras, i)
    proj_corners = utils_3d.project_3d_point(
        camera=camera,
        point3d=bbox_3d,
        width=w,
        height=h,
        is_world_coord=is_world_coord,
    )
    if bboxes_center_3d is not None:
      proj_centers = utils_3d.project_3d_point(
          camera=camera,
          point3d=bboxes_center_3d[i],
          width=w,
          height=h,
          is_world_coord=is_world_coord,
      )
    else:
      proj_centers = None
    images[i] = draw_projected_3d_bbox(
        image=image.copy(),
        proj_corners=proj_corners,
        proj_centers=proj_centers,
        line_color=bbox_color,
        pt_color=center_color,
    )
  return images

