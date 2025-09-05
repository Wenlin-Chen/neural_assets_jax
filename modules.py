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

"""Modules for Neural Assets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TypedDict, TypeVar

from big_vision.models import vit
import diffusion
import einops
from flax import linen as nn
import jax
import jax.numpy as jnp
import model_utils
import numpy as np
import preprocessing


NpOrJaxArray = TypeVar('NpOrJaxArray', jnp.ndarray, np.ndarray)


class ControllableGeneratorReturn(TypedDict):
  """Output shapes of ControllableGenerator."""

  diff: jnp.ndarray  # [B, h, w, c], GT added noise
  pred_diff: jnp.ndarray  # [B, h, w, c], predicted noise

  # Inputs used to extract conditioning tokens, mostly for visualization
  # purposes, e.g., check if the bboxes are correct
  src_bboxes: Optional[jnp.ndarray] = None  # [B, n, 4]
  conditioning_object_poses: Optional[jnp.ndarray] = None  # [B, n, co]


class ControllableGenerator(nn.Module):
  """Wrapper for a conditional generative model."""

  generator: diffusion.DiffuserDiffusionWrapper
  conditioning_encoder: ConditioningEncoder
  conditioning_neck: FeedForwardNeck

  def _get_conditioning_tokens(
      self,
      tgt_object_poses: jnp.ndarray,  # [B, n, co]
      src_images: jnp.ndarray,  # [B, H, W, C]
      src_bboxes: jnp.ndarray,  # [B, n, 4]
      src_bg_images: Optional[jnp.ndarray] = None,  # [B, H, W, C]
  ) -> tuple[jnp.ndarray, ConditioningEncoderReturn]:
    # Encode conditioning inputs to tokens
    cond_dict = self.conditioning_encoder(
        tgt_object_poses=tgt_object_poses,
        src_images=src_images,
        src_bboxes=src_bboxes,
        src_bg_images=src_bg_images,
    )

    # Process encoded tokens as the model's conditioning inputs
    conditioning_tokens = self.conditioning_neck(conditioning_dict=cond_dict)
    # [B, nc, dc]

    return conditioning_tokens, cond_dict

  @nn.compact
  def __call__(
      self,
      tgt_images: jnp.ndarray,  # [B, H, W, C]
      tgt_object_poses: jnp.ndarray,  # [B, n, co]
      src_images: jnp.ndarray,  # [B, H, W, C]
      src_bboxes: jnp.ndarray,  # [B, n, 4]
      src_bg_images: Optional[jnp.ndarray] = None,  # [B, H, W, C]
  ) -> ControllableGeneratorReturn:
    # Encode inputs to conditioning_tokens
    conditioning_tokens, cond_dict = self._get_conditioning_tokens(
        tgt_object_poses=tgt_object_poses,
        src_images=src_images,
        src_bboxes=src_bboxes,
        src_bg_images=src_bg_images,
    )

    # Condition the generator on conditioning_tokens to reconstruct tgt_images
    # NOTE: Diffusers models always expect images to be in [B, C, H, W] format
    # [B, H, W, C] -> [B, C, H, W]
    tgt_images = jnp.transpose(tgt_images, (0, 3, 1, 2))
    result_dict = self.generator(
        images=tgt_images,
        conditioning_tokens=conditioning_tokens,
    )

    return self.postprocess_model_output(result_dict, cond_dict)

  def postprocess_model_output(
      self,
      generator_output: Dict[str, Any],
      cond_dict: ConditioningEncoderReturn,
  ) -> ControllableGeneratorReturn:
    """Convert model output to desired format."""
    result_dict = {
        # For loss computation
        'diff': generator_output['diff'],
        'pred_diff': generator_output['pred_diff'],
        # For visualization
        'src_bboxes': cond_dict['src_bboxes'],
        'conditioning_object_poses': cond_dict['conditioning_object_poses'],
    }

    return result_dict


class MLP(nn.Module):
  """A simple MLP."""

  hidden_size: int
  output_size: Optional[int] = None
  num_hidden_layers: int = 1

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    output_size = self.output_size or inputs.shape[-1]

    x = inputs
    for _ in range(self.num_hidden_layers):
      x = nn.Dense(self.hidden_size)(x)
      x = nn.gelu(x)
    x = nn.Dense(output_size)(x)

    return x


class ConditioningEncoderReturn(TypedDict):
  """Output shapes of ConditioningEncoder."""

  appearance_tokens: jnp.ndarray  # [B, _n, da]
  object_pose_tokens: Optional[jnp.ndarray] = None  # [B, _n, do]

  # Inputs used to extract conditioning tokens, mostly for visualization
  # purposes, e.g., check if the bboxes are correct
  src_bboxes: Optional[jnp.ndarray] = None  # [B, n, 4]
  conditioning_object_poses: Optional[jnp.ndarray] = None  # [B, n, co]


class ConditioningEncoder(nn.Module):
  """Wrapper for different conditioning encoders.

  This module takes in different conditioning inputs (e.g. image, object bbox,
  3D pose), and returns a dict of features as conditioning tokens.
  """

  appearance_encoder: nn.Module
  object_pose_encoder: nn.Module

  # We may mask out background pixels when encoding object tokens
  mask_out_bg_for_appearance: bool = True
  background_value: float = 0.5
  # The background is different from foreground objects, so we need a special
  # positional token for it. Current options include:
  # - 'mlp': apply an MLP on the input background positional token. For example,
  #          we may compute the relative camera pose between two frames and
  #          flatten it in data pre-processing, then we apply MLP on it.
  background_pos_enc_type: Optional[str] = None

  def _prepare_bg_pos_enc(self, pos_input: jnp.ndarray) -> jnp.ndarray:
    """We may need a special positional token for background modeling."""
    if self.background_pos_enc_type is None:
      return pos_input
    elif self.background_pos_enc_type == 'mlp':
      # NOTE: we assume that in data pre-processing, we append a tensor as the
      # background positional token, which can be e.g. relative camera pose.
      bg_pos = pos_input[..., -1:, :]  # [*B, 1, c]
      pos_dim = pos_input.shape[-1]  # c
      bg_pos = MLP(
          hidden_size=pos_dim * 2, output_size=pos_dim, num_hidden_layers=1
      )(bg_pos)
      pos_input = jnp.concatenate(
          [pos_input[..., :-1, :], bg_pos], axis=-2
      )  # [*B, n-1, c] + [*B, 1, c] = [*B, n, c]
    else:
      raise ValueError(f'Unknown {self.background_pos_enc_type=}')
    return pos_input

  @nn.compact
  def __call__(
      self,
      tgt_object_poses: jnp.ndarray,  # [B, n, co]
      src_images: jnp.ndarray,  # [B, H, W, C]
      src_bboxes: jnp.ndarray,  # [B, n, 4]
      src_bg_images: Optional[jnp.ndarray] = None,  # [B, H, W, C]
  ) -> ConditioningEncoderReturn:
    result_dict = {
        'src_bboxes': src_bboxes,
        'conditioning_object_poses': tgt_object_poses,
    }

    # Extract conditioning inputs from the image
    appearance_token_src = src_images  # input for appearance token encoding

    # Encode object appearance information.
    # Maybe mask out background pixels when encoding object tokens
    if self.mask_out_bg_for_appearance:
      h, w = src_images.shape[-3:-1]
      # Get coarse object masks from bboxes
      # When using `src_bg_images`, we will append a global bbox at the end
      # So need to remove it here when creating foreground mask
      fg_bboxes = (
          src_bboxes if src_bg_images is None else src_bboxes[..., :-1, :]
      )
      fg_masks = model_utils.boxes_to_sparse_segmentations(fg_bboxes, h, w)
      # [*B, n, H, W] -> [*B, H, W, 1]
      fg_masks = jnp.any(jnp.greater(fg_masks, 0), axis=-3)[..., None]
      appearance_token_src = jnp.where(
          fg_masks,
          src_images,
          jnp.ones_like(src_images) * self.background_value,
      )
    # NOTE: if we use `src_bg_images` to extract background tokens, we assume
    # that a global bbox and a global object_pose is already appended to input
    # `src_bboxes` and `tgt_object_poses`.
    result_dict['appearance_tokens'] = self.appearance_encoder(
        images=appearance_token_src,
        bboxes=src_bboxes,
        bg_images=src_bg_images,
    )

    # Encode object pose token.
    tgt_object_poses = self._prepare_bg_pos_enc(tgt_object_poses)
    result_dict['object_pose_tokens'] = self.object_pose_encoder(
        tgt_object_poses
    )

    return result_dict


class RoIAlignAppearanceEncoder(nn.Module):
  """An appearance encoder that uses the RoIAligned features of object bbox."""

  shape: tuple[int, int]  # [max_num_objects, feature_dim]
  image_backbone: nn.Module

  # How to aggregate per-object feature maps (e.g. 28x28) to appearance tokens
  roi_align_size: int = 7
  aggregate_method: str = 'mean'
  # 'mean': mean-pooling
  # 'max': max-pooling
  # 'flatten': flatten the spatial dimension into the token dimension

  def _aggregate_obj_features(
      self,
      bboxes: jnp.ndarray,  # [B, n, 4]
      roi_features: jnp.ndarray,  # [B, n, s, s, c]
  ) -> jnp.ndarray:  # [B, _n, _c]
    """Aggregate per-bbox feature maps to get object appearance tokens."""
    if self.aggregate_method == 'mean':  # mean-pool
      obj_features = jnp.mean(roi_features, axis=(-3, -2))
    elif self.aggregate_method == 'max':  # max-pool
      obj_features = jnp.max(roi_features, axis=(-3, -2))
    elif self.aggregate_method == 'flatten':  # flatten into token dimension
      obj_features = einops.rearrange(
          roi_features, '... n s1 s2 d -> ... (n s1 s2) d'
      )
    else:
      raise ValueError(f'Unknown pooling method: {self.aggregate_method}.')

    # Special case: Empty bboxes result in taking image_features[..., 0, 0, :]
    # We set them to zeros here
    is_non_empty = jnp.any(
        jnp.not_equal(bboxes, jnp.array(preprocessing.NOTRACK_BOX)),
        axis=-1,
        keepdims=True,
    )  # [*B, n, 1]
    # Duplicate it to [*B, (n*s*s), 1] when token number is more than 1
    if self.aggregate_method == 'flatten':
      is_non_empty = einops.repeat(
          is_non_empty,
          '... n 1 -> ... (n repeat) 1',
          repeat=self.roi_align_size**2,
      )
    obj_features = jnp.where(
        is_non_empty, obj_features, jnp.zeros_like(obj_features)
    )
    return obj_features

  def _extract_obj_features(
      self,
      images: jnp.ndarray,  # [B, H, W, C]
      bboxes: jnp.ndarray,  # [B, n, 4]
  ) -> jnp.ndarray:  # [B, _n, _c]
    """Extract object-centric features via RoIAlign using 2D bboxes."""
    # Extract image features
    image_features = self.image_backbone(images)
    # Shape: [*B, w', h', c]

    # Apply RoIAlign to get per-bbox feature maps
    roi_features = model_utils.get_roi_align_features(
        bboxes, image_features, size=self.roi_align_size
    )
    # Shape: [*B, n, s, s, c]

    # Aggregate per-bbox feature maps to get object appearance tokens
    obj_features = self._aggregate_obj_features(bboxes, roi_features)
    # Shape: [*B, n or (n*s*s), c]
    return obj_features

  @nn.compact
  def __call__(
      self,
      images: jnp.ndarray,  # [B, H, W, C]
      bboxes: jnp.ndarray,  # [B, n, 4]
      bg_images: Optional[jnp.ndarray] = None,  # [B, H, W, C]
  ) -> jnp.ndarray:  # [B, _n, d]
    bboxes = jax.lax.stop_gradient(bboxes)
    assert (
        bboxes.shape[-2] == self.shape[-2]
    ), f'Expected {self.shape[-2]} bboxes, but got {bboxes.shape[-2]}'

    if bg_images is None:
      app_features = self._extract_obj_features(images, bboxes)
    else:
      # We assume the first (n-1) bboxes are for foreground objects
      # The last bbox is a global bbox for background features
      obj_features = self._extract_obj_features(images, bboxes[..., :-1, :])
      bg_features = self._extract_obj_features(bg_images, bboxes[..., -1:, :])
      app_features = jnp.concatenate([obj_features, bg_features], axis=-2)
    # Shape: [*B, n or (n*s*s), c]

    # Project to the desired feature dim
    return nn.Dense(self.shape[-1])(app_features)


class MLPPoseEncoder(nn.Module):
  """A module that encodes object pose with MLP as conditioning inputs."""

  mlp_module: nn.Module
  # Repeat bbox encoding, e.g., when using multiple appearance tokens per asset
  duplicate_factor: int = 1

  @nn.compact
  def __call__(self, bboxes: jnp.ndarray) -> jnp.ndarray:
    pose_features = self.mlp_module(bboxes)
    if self.duplicate_factor > 1:
      pose_features = einops.repeat(
          pose_features,
          '... n d -> ... (n repeat) d',
          repeat=self.duplicate_factor,
      )
    return pose_features


class FeedForwardNeck(nn.Module):
  """A simple module for converting encoded conditioning to generator inputs."""

  feed_forward_module: nn.Module

  @nn.compact
  def __call__(
      self, conditioning_dict: ConditioningEncoderReturn
  ) -> jnp.ndarray:
    # Process encoded tokens into one feature tensor
    conds = [
        conditioning_dict['appearance_tokens'],
        conditioning_dict['object_pose_tokens'],
    ]
    # Each entry has shape [*B, n, c]

    # Concatenate conditioning tokens along the channel dimension, which
    # explicitly associates conditioning tokens of the same object
    conditioning_tokens = jnp.concatenate(conds, axis=-1)

    # Fuse the appearance and the pose tokens
    conditioning_tokens = self.feed_forward_module(conditioning_tokens)

    return conditioning_tokens  # [B, num_tokens, token_dim]


class DINOViT(nn.Module):
  """DINO v1 ViT model.

  DINO v1: https://arxiv.org/abs/2104.14294
  Implementation forked from https://github.com/google-research/big_vision
  """

  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: int = 3072
  num_heads: int = 12

  in_vrange: tuple[float, float] = (0.0, 1.0)
  use_imagenet_value_range: bool = True
  frozen_model: bool = True

  def _preprocess(self, image: jnp.ndarray) -> jnp.ndarray:
    """Preprocess input image."""
    # Resize to 224x224.
    image = jax.image.resize(image, (image.shape[0], 224, 224, 3), 'bilinear')

    # Normalize to [0, 1]
    vmin, vmax = self.in_vrange
    image = (image - vmin) / (vmax - vmin)

    # Optionally adjust to ImageNet pre-trained value range
    if self.use_imagenet_value_range:
      image = image - jnp.asarray((0.485, 0.456, 0.406))
      image = image / jnp.asarray((0.229, 0.224, 0.225))

    return image

  @nn.compact
  def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
    image = self._preprocess(image)  # [B, H, W, C]

    # Patch extraction
    x = nn.Conv(
        self.width,
        self.patch_size,
        strides=self.patch_size,
        padding='VALID',
        name='embedding',
    )(image)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # Add [CLS] token
    cls = self.param('cls', nn.initializers.zeros, (1, 1, c), x.dtype)
    x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    # Add posemb
    l = h * w + 1
    posemb = vit.get_posemb(self, 'learn', l, c, 'pos_embedding', x.dtype)
    x = x + posemb

    x, _ = vit.Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        name='Transformer',
    )(x, deterministic=True)

    # Remove cls token and unflatten
    x = x[:, 1:, :]
    x = jnp.reshape(x, (n, h, w, c))
    if self.frozen_model:
      x = jax.lax.stop_gradient(x)

    return x  # [B, h, w, c]

  @classmethod
  def from_variant_str(cls, version: str, variant: str, **kwargs) -> DINOViT:
    if version == 'v1':
      assert variant in ('B/16', 'B/8'), 'DINO v1 only supports B/16 and B/8.'
    else:
      raise ValueError(
          f'Unknown version: {version}, version should be either v1 or v2.'
      )

    v, patch = variant.split('/')
    patch = (int(patch), int(patch))

    if v == 'B':
      kwargs.update({
          'width': 768,
          'depth': 12,
          'mlp_dim': 3072,
          'num_heads': 12,
      })
    else:
      raise ValueError('Only supports DINO with ViT-B.')

    return cls(patch_size=patch, **kwargs)
