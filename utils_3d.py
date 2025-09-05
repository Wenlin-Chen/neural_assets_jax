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

r"""Utility functions for 3D-related transformations.

Some code borrowed from
https://github.com/google-research/kubric/blob/main/challenges/point_tracking/dataset.py

3D bbox (after rotated back with quaternion) format in Kubric is:
- Y up, X left, Z from back to camera

                                up y
                                  ^
                                  |
                                  |
   p6=(x1, y1, z0) + ------------ + p2=(x0, y1, z0)
                  /|            / |
                 / |         p3/  |
p7=(x1, y1, z1) + ----------- +   + p0=(x0, y0, z0)
                |  /      .   |  /
                | / origin    | /
left x <------- + ----------- + p1=(x0, y0, z1)
            p5=(x1, y0, z1)  /
                            /
                     front z
"""

import model_utils
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


def normalize_quat(q):
  """Get a unit quaternion."""
  # q: [*N, 4]
  return tf.math.l2_normalize(q, axis=-1)


def batch_quat_mul(q1, q2):
  """Multiply two quaternions."""
  # both have shape [*N, 4]
  q1 = normalize_quat(q1)
  q2 = normalize_quat(q2)
  w1, x1, y1, z1 = tf.split(q1, 4, axis=-1)
  w2, x2, y2, z2 = tf.split(q2, 4, axis=-1)
  return tf.concat(
      [
          w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ],
      axis=-1,
  )


def batch_quat_inverse(q):
  """Inverse of a quaternion."""
  # q: [*N, 4]
  q = normalize_quat(q)
  w, x, y, z = tf.split(q, 4, axis=-1)
  return tf.concat([w, -x, -y, -z], axis=-1)


def batch_quat2rmat(quaternion):
  """Convert [*N, 4] quaternion to [*N, 3, 3] rotation matrix."""
  # quaternion: [*N, 4]
  quaternion = normalize_quat(quaternion)
  q0, q1, q2, q3 = tf.split(quaternion, 4, axis=-1)

  # Compute the rotation matrix
  rotation_matrix = tf.stack(
      [
          tf.concat(
              [
                  1 - 2 * (q2**2 + q3**2),
                  2 * (q1 * q2 - q0 * q3),
                  2 * (q0 * q2 + q1 * q3),
              ],
              axis=-1,
          ),
          tf.concat(
              [
                  2 * (q1 * q2 + q0 * q3),
                  1 - 2 * (q1**2 + q3**2),
                  2 * (q2 * q3 - q0 * q1),
              ],
              axis=-1,
          ),
          tf.concat(
              [
                  2 * (q1 * q3 - q0 * q2),
                  2 * (q0 * q1 + q2 * q3),
                  1 - 2 * (q1**2 + q2**2),
              ],
              axis=-1,
          ),
      ],
      axis=-2,
  )

  return rotation_matrix


def batch_rmat2quat(rotation_matrix):
  """Convert [*N, 3, 3] rotation matrix to [*N, 4] quaternion."""
  quaternion = tfg.quaternion.from_rotation_matrix(rotation_matrix)
  # [qx, qy, qz, qw] to [qw, qx, qy, qz]
  qx, qy, qz, qw = tf.split(quaternion, 4, axis=-1)
  return tf.concat([qw, qx, qy, qz], axis=-1)


def get_axis_aligned_3d_bboxes(bboxes_3d, quaternions):
  """Rotate 3D bboxes to be axis-aligned."""
  # bboxes_3d: [*N, 8, 3]
  # quaternions: [*N, 4]
  rotation_matrixs = batch_quat2rmat(quaternions)  # [*N, 3, 3]
  aabb_3d = bboxes_3d @ rotation_matrixs  # [*N, 8, 3]
  return aabb_3d


def get_3d_bboxes_size(bboxes_3d, quaternions):
  """Get the width/height/depth of 3D rotated bboxes."""
  # bboxes_3d: [*N, 8, 3]
  # quaternions: [*N, 4]
  aabb_3d = get_axis_aligned_3d_bboxes(bboxes_3d, quaternions)  # [*N, 8, 3]
  width = aabb_3d[..., 4, 0] - aabb_3d[..., 0, 0]
  height = aabb_3d[..., 2, 1] - aabb_3d[..., 0, 1]
  depth = aabb_3d[..., 1, 2] - aabb_3d[..., 0, 2]
  return tf.stack([width, height, depth], axis=-1)  # [*N, 3]


def batch_construct_3d_bboxes(centers, sizes, quaternions):
  """Construct bbox_3d (8 corners) from components."""
  # centers: [*N, 3]
  # sizes: [*N, 3]
  # quaternions: [*N, 4]
  # First construct the corresponding axis-aligned bounding box
  rmats = batch_quat2rmat(quaternions)  # [*N, 3, 3]
  centers = (centers[..., None, :] @ rmats)[..., 0, :]  # [*N, 3]
  x0, y0, z0 = tf.split(centers - sizes / 2.0, 3, axis=-1)  # [*N, 1]
  x1, y1, z1 = tf.split(centers + sizes / 2.0, 3, axis=-1)  # [*N, 1]
  p0 = tf.concat([x0, y0, z0], axis=-1)
  p1 = tf.concat([x0, y0, z1], axis=-1)
  p2 = tf.concat([x0, y1, z0], axis=-1)
  p3 = tf.concat([x0, y1, z1], axis=-1)
  p4 = tf.concat([x1, y0, z0], axis=-1)
  p5 = tf.concat([x1, y0, z1], axis=-1)
  p6 = tf.concat([x1, y1, z0], axis=-1)
  p7 = tf.concat([x1, y1, z1], axis=-1)
  aabb_3d = tf.stack([p0, p1, p2, p3, p4, p5, p6, p7], axis=-2)  # [*N, 8, 3]
  # Now rotate back
  bboxes_3d = aabb_3d @ tf.linalg.matrix_transpose(rmats)  # [*N, 8, 3]
  return bboxes_3d


def batch_decompose_3d_bboxes(bboxes_3d, quaternions):
  """Get centers, sizes, and quaternions from bboxes_3d."""
  # bboxes_3d: [*N, 8, 3]
  # quaternions: [*N, 4]
  centers = tf.reduce_mean(bboxes_3d, axis=-2)  # [*N, 3]
  sizes = get_3d_bboxes_size(bboxes_3d, quaternions)  # [*N, 3]
  quats = normalize_quat(quaternions)  # [*N, 4]
  return centers, sizes, quats  # [*N, 3/3/4]


def batch_get_matrix_world(position, quaternion):
  """Get the transformation matrix in world coordinates."""
  # position: [*N, 3]
  # quaternion: [*N, 4]
  rotation_matrix = batch_quat2rmat(quaternion)  # [*N, 3, 3]
  dummy = tf.zeros_like(rotation_matrix[..., :1, :])  # [*N, 1, 3]
  transformation = tf.concat([rotation_matrix, dummy], axis=-2)  # [*N, 4, 3]
  dummy = tf.ones_like(position[..., 0:1])  # [*N, 1]
  position_4d = tf.concat([position, dummy], axis=-1)  # [*N, 4]
  transformation = tf.concat([transformation, position_4d[..., None]], axis=-1)
  return transformation  # [*N, 4, 4]


def batch_world2camera(cam_pos, cam_quat, point3d, quats=None):
  """Transform 3D points from world coordinate to camera coordinate."""
  # cam_pos: [*B, 3]
  # cam_quat: [*B, 4]
  # point3d: [*B, N, 3]
  # quats: [*B, M, 4]
  matrix_world = batch_get_matrix_world(cam_pos, cam_quat)  # [*B, 4, 4]
  homo_transform = tf.linalg.inv(matrix_world)
  homo_transform = tf.linalg.matrix_transpose(homo_transform)
  point4d = tf.concat([point3d, tf.ones_like(point3d[..., 0:1])], axis=-1)
  projected = point4d @ homo_transform  # [*B, N, 4] @ [*B, 4, 4] -> [*B, N, 4]
  if quats is not None:
    cam_quat = cam_quat[..., None, :]  # [*B, 1, 4]
    quats = batch_quat_mul(batch_quat_inverse(cam_quat), quats)
  return projected[..., :3], quats


def batch_camera2world(point3d, cam_pos=None, cam_quat=None, cam2world=None):
  """Transform 3D points from world coordinate to camera coordinate."""
  # point3d: [*B, N, 3]
  # quats: [*B, M, 4]
  # cam_pos: [*B, 3]
  # cam_quat: [*B, 4]
  # cam2world: [*B, 4, 4]
  if cam2world is None:
    homo_transform = cam2world
  else:
    homo_transform = batch_get_matrix_world(cam_pos, cam_quat)  # [*B, 4, 4]
  homo_transform = tf.linalg.matrix_transpose(homo_transform)
  point4d = tf.concat([point3d, tf.ones_like(point3d[..., 0:1])], axis=-1)
  projected = point4d @ homo_transform  # [*B, N, 4] @ [*B, 4, 4] -> [*B, N, 4]
  return projected[..., :3]


def batch_get_intrinsics(focal_length, sensor_width, width, height):
  """Get camera intrinsics."""
  # NOTE: this function is only compatible with MOVi (Blender) camera model
  # focal_length, sensor_width: [*N]
  # width, height: int
  f_x = focal_length / sensor_width
  sensor_height = sensor_width * width / height
  f_y = focal_length / sensor_height
  ones = tf.ones_like(f_x)
  zeros = tf.zeros_like(f_x)
  p_x = p_y = ones / 2.0
  return tf.stack(
      [
          tf.stack([f_x, zeros, -p_x], axis=-1),
          tf.stack([zeros, -f_y, -p_y], axis=-1),
          tf.stack([zeros, zeros, -ones], axis=-1),
      ],
      axis=-2,
  )


def batch_camera2image(
    point3d,
    intrinsics=None,
    focal_length=None,
    sensor_width=None,
    width=None,
    height=None,
):
  """Project 3D points (already in camera coordinate) to [0, 1] image space."""
  # NOTE: this function is different from `camera2image()` in utils_3d.py
  # Here, we always return the depth of 3D points, while in utils_3d.py, we only
  # return the projected image-space coordinates by default
  # point3d: [*B, N, 3]
  # intrinsics: [*B, 3, 3]
  # others are float or [*B]
  point4d = tf.concat([point3d, tf.ones_like(point3d[..., 0:1])], axis=-1)
  # point4d: [*B, N, 4]
  if intrinsics is not None:
    assert (
        focal_length is None and sensor_width is None
    ), 'either directly input intrinsics, or compute it from parameters'
  else:
    # Extend batch dim for scalars, make them [*B]
    focal_length = tf.ones_like(point3d[..., 0, 0]) * focal_length
    sensor_width = tf.ones_like(point3d[..., 0, 0]) * sensor_width
    width = tf.ones_like(point3d[..., 0, 0]) * width
    height = tf.ones_like(point3d[..., 0, 0]) * height
    # So we get intrinsics of shape [*B, 3, 3]
    intrinsics = batch_get_intrinsics(focal_length, sensor_width, width, height)
  homo_intrinsics = tf.concat(
      [intrinsics, tf.zeros_like(intrinsics[..., 0:1])], axis=-1
  )  # [*B, 3, 4]
  homo_intrinsics = tf.linalg.matrix_transpose(homo_intrinsics)  # [*B, 4, 3]
  projected = point4d @ homo_intrinsics  # [*B, N, 3]
  # Perform Z-normalization
  depth = projected[..., 2:3]  # [*B, N, 1]
  image_coords = projected[..., 0:2] / (depth + 1e-6)  # [*B, N, 2]
  # Concat with a Z-dim for compatibility
  # For padded points (0, 0, 0), we still make their Z as 1
  image_coords = tf.concat(
      [image_coords, tf.ones_like(image_coords[..., 0:1])], axis=-1
  )  # [*B, N, 3]
  return image_coords, depth


def batch_image2camera(
    image_coords,
    depth,
    intrinsics=None,
    focal_length=None,
    sensor_width=None,
    width=None,
    height=None,
):
  """Unproject points from [0, 1] image space to 3D camera coordinates."""
  # NOTE: this is designed to be the inverse of `batch_camera2image()`
  # image_coords: [*B, N, 2/3]
  # depth: [*B, N, 1]
  # intrinsics: [*B, 3, 3]
  # others are float or [*B]
  # Recover points before Z-normalization
  image_coords = image_coords[..., :2]  # only need projected (x, y)
  point3d = tf.concat(
      [image_coords, tf.ones_like(image_coords[..., 0:1])], axis=-1
  )  # [*B, N, 3]
  point3d = point3d * depth
  point4d = tf.concat([point3d, tf.ones_like(point3d[..., 0:1])], axis=-1)
  # point4d: [*B, N, 4]
  if intrinsics is not None:
    assert (
        focal_length is None
        and sensor_width is None
        and width is None
        and height is None
    ), 'either directly input intrinsics, or compute it from parameters'
  else:
    # Extend batch dim for scalars, make them [*B]
    focal_length = tf.ones_like(image_coords[..., 0, 0]) * focal_length
    sensor_width = tf.ones_like(image_coords[..., 0, 0]) * sensor_width
    width = tf.ones_like(image_coords[..., 0, 0]) * width
    height = tf.ones_like(image_coords[..., 0, 0]) * height
    # So we get intrinsic of shape [*B, 3, 3]
    intrinsics = batch_get_intrinsics(focal_length, sensor_width, width, height)
  homo_intrinsics = tf.concat(
      [intrinsics, tf.zeros_like(intrinsics[..., 0:1])], axis=-1
  )  # [*B, 3, 4]
  # Make it [*B, 4, 4] so that we can apply matrix inverse
  dummy = tf.concat(
      [
          tf.zeros_like(homo_intrinsics[..., 0]),
          tf.ones_like(homo_intrinsics[..., 0, 0:1]),
      ],
      axis=-1,
  )  # [*B, 4]
  homo_intrinsics = tf.concat([homo_intrinsics, dummy[..., None, :]], axis=-2)
  homo_intrinsics = tf.linalg.inv(tf.linalg.matrix_transpose(homo_intrinsics))
  unproj4d = point4d @ homo_intrinsics  # [*B, N, 4]
  return unproj4d[..., :3]  # [*B, N, 3], (x, y, z)


def get_intrinsics(focal_length, sensor_width, width, height):
  """Get camera intrinsics."""
  f_x = focal_length / sensor_width
  sensor_height = sensor_width * width / height
  f_y = focal_length / sensor_height
  p_x = 1 / 2.0
  p_y = 1 / 2.0
  return np.array([
      [f_x, 0, -p_x],
      [0, -f_y, -p_y],
      [0, 0, -1],
  ])


def quat2rmat(quaternion, eps=1e-6):
  """Convert a [4,] quaternion to a [3, 3] rotation matrix."""
  q0, q1, q2, q3 = quaternion

  # Normalize the quaternion
  norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2) + eps
  q0 /= norm
  q1 /= norm
  q2 /= norm
  q3 /= norm

  # Compute the rotation matrix
  rotation_matrix = np.array([
      [
          1 - 2 * (q2**2 + q3**2),
          2 * (q1 * q2 - q0 * q3),
          2 * (q0 * q2 + q1 * q3),
      ],
      [
          2 * (q1 * q2 + q0 * q3),
          1 - 2 * (q1**2 + q3**2),
          2 * (q2 * q3 - q0 * q1),
      ],
      [
          2 * (q1 * q3 - q0 * q2),
          2 * (q0 * q1 + q2 * q3),
          1 - 2 * (q1**2 + q2**2),
      ],
  ])

  return rotation_matrix


def get_matrix_world(position, quaternion):
  """Get the transformation matrix in world coordinates."""
  rotation_matrix = quat2rmat(quaternion)
  transformation = np.eye(4)
  transformation[:3, :3] = rotation_matrix
  transformation[:3, 3] = position
  return transformation


def world2camera(cam_pos, cam_quat, point3d, quats=None):
  """Transform 3D points from world coordinate to camera coordinate."""
  # cam_pos: [3]
  # cam_quat: [4]
  # point3d: [*N, 3]
  # quats: [*N, 4]
  point3d, batch_shape = model_utils.flatten_batch_axes(
      point3d, num_data_axes=1)
  matrix_world = get_matrix_world(cam_pos, cam_quat)
  homo_transform = np.linalg.inv(matrix_world)
  point4d = np.concatenate([point3d, np.ones_like(point3d[:, 0:1])], axis=1)
  projected = point4d @ homo_transform.T
  projected = model_utils.unflatten_batch_axes(projected, batch_shape)
  if quats is not None:
    quats, batch_shape = model_utils.flatten_batch_axes(quats, num_data_axes=1)
    quats = batch_quat_mul(batch_quat_inverse(cam_quat[None]), quats)
    quats = model_utils.unflatten_batch_axes(quats, batch_shape)
  return projected[..., :3], quats


def camera2image(
    point3d,
    width,
    height,
    focal_length=None,
    sensor_width=None,
    intrinsic=None,
    return_depth=False,
):
  """Project 3D points (already in camera coordinate) to [0, 1] image space."""
  # point3d: [*N, 3]
  # intrinsic: [3, 3]
  # others are float
  point3d, batch_shape = model_utils.flatten_batch_axes(
      point3d, num_data_axes=1)
  point4d = np.concatenate([point3d, np.ones_like(point3d[:, 0:1])], axis=1)
  # point4d: [*N, 4]
  if intrinsic is not None:
    assert intrinsic.shape == (3, 3), f'Invalid {intrinsic.shape=}'
    assert focal_length is None and sensor_width is None, (
        'Camera intrinsic and focal_length/sensor_width should not be provided'
        ' at the same time'
    )
  else:
    intrinsic = get_intrinsics(focal_length, sensor_width, width, height)
  homo_intrinsics = np.zeros((3, 4), dtype=np.float32)
  homo_intrinsics[:, :3] = intrinsic
  projected = point4d @ homo_intrinsics.T
  projected = model_utils.unflatten_batch_axes(projected, batch_shape)
  # Perform Z-normalization
  # For padded points (0, 0, 0), we still make their Z as 1
  depth = projected[..., 2:3]
  image_coords = projected / (depth + 1e-6)
  image_coords[..., 2] = np.sign(image_coords[..., 2] + 1e-6)
  if return_depth:
    return image_coords, depth
  return image_coords


def project_3d_point(camera, point3d, width, height, is_world_coord=True):
  """Project points in world coordinates to image space coordinate in [0, 1]."""
  if is_world_coord:
    point3d, _ = world2camera(camera['position'], camera['quaternion'], point3d)
  image_coords = camera2image(
      point3d,
      width,
      height,
      focal_length=camera.get('focal_length', None),
      sensor_width=camera.get('sensor_width', None),
      intrinsic=camera.get('intrinsic', None),
  )
  return image_coords


def construct_3d_bboxes_from_p0124(p0, p1, p2, p4):
  """Construct bbox_3d (8 corners) from 4 corners."""
  # p: [*N, 3]
  diff_x = p4 - p0
  diff_y = p2 - p0
  diff_z = p1 - p0
  p3 = p0 + diff_y + diff_z
  p5 = p1 + diff_x
  p6 = p2 + diff_x
  p7 = p3 + diff_x
  bboxes_3d = np.stack([p0, p1, p2, p3, p4, p5, p6, p7], axis=-2)  # [*N, 8, 3]
  return bboxes_3d
