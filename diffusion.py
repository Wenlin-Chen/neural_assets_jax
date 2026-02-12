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

"""Flax-implemented Stable Diffusion from Diffusers."""

from __future__ import annotations

from typing import TypedDict

import diffusers
from flax import linen as nn
import jax
import jax.numpy as jnp


class DiffuserDiffusionLossReturn(TypedDict):
  """Output shapes of DiffuserDiffusion's training forward pass."""

  diff: jnp.ndarray  # [B, h, w, c], GT added noise
  pred_diff: jnp.ndarray  # [B, h, w, c], predicted noise


class DiffuserDiffusionWrapper(nn.Module):
  """Interface between diffuser models and our modules."""

  model_name: str = 'stable_diffusion_v2_1'

  def setup(self):
    if self.model_name == 'stable_diffusion_v2_1':
      model_name = 'flax/stable-diffusion-2-1'
    else:
      raise ValueError(f'Unknown model name {self.model_name}')
    self.vae, pretrained_vae_params = diffusers.FlaxAutoencoderKL.from_pretrained(
        model_name, subfolder='vae'
    )
    self.unet, pretrained_unet_params = (
        diffusers.FlaxUNet2DConditionModel.from_pretrained(
            model_name, subfolder='unet'
        )
    )

    # Register pretrained UNet/VAE params as Flax parameters so they are trainable
    # and will be handled by Optax + Flax checkpointing.
    self.vae_params = self.param('vae_params', lambda _: pretrained_vae_params)
    self.unet_params = self.param('unet_params', lambda _: pretrained_unet_params)
    self.noise_scheduler = diffusers.FlaxDDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule='scaled_linear',
        num_train_timesteps=1000,
    )
    self.noise_scheduler_state = self.noise_scheduler.create_state()

  def __call__(
      self,
      images: jnp.ndarray,  # [B, H, W, C]
      conditioning_tokens: jnp.ndarray,  # [B, n, d]
  ) -> DiffuserDiffusionLossReturn:
    # Training pass, code adapted from:
    # https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py

    # Convert images to latent space
    vae_outputs = self.vae.apply(
        {'params': self.vae_params},
        images,
        deterministic=True,
        method=self.vae.encode,
    )
    sample_rng = self.make_rng('diffusion')
    latents = vae_outputs.latent_dist.sample(sample_rng)
    # (NHWC) -> (NCHW)
    latents = jnp.transpose(latents, (0, 3, 1, 2))
    latents = latents * self.vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise_rng, timestep_rng = jax.random.split(sample_rng)
    noise = jax.random.normal(noise_rng, latents.shape)
    # Sample a random timestep for each image
    bsz = latents.shape[0]
    timesteps = jax.random.randint(
        timestep_rng,
        (bsz,),
        0,
        self.noise_scheduler.config.num_train_timesteps,
    )

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = self.noise_scheduler.add_noise(
        self.noise_scheduler_state, latents, noise, timesteps
    )

    # Replace the text embedding with our Neural Assets for conditioning
    encoder_hidden_states = conditioning_tokens

    # Predict the noise residual and compute loss
    model_pred = self.unet.apply(
        {'params': self.unet_params},
        noisy_latents,
        timesteps,
        encoder_hidden_states,
        train=True,
    ).sample

    # Get the target for loss depending on the prediction type
    pred_type = self.noise_scheduler.config.prediction_type
    if pred_type == 'epsilon':
      target = noise
    elif pred_type == 'v_prediction':
      target = self.noise_scheduler.get_velocity(
          self.noise_scheduler_state, latents, noise, timesteps
      )
    else:
      raise ValueError(f'Unknown prediction type {pred_type}')

    loss_dict = {
        'diff': target,
        'pred_diff': model_pred,
    }

    return loss_dict
