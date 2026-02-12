# --- Imports ---

import os
# Set CUDA_VISIBLE_DEVICES to '0' for the first GPU, '1' for the second, etc.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import diffusion
from etils.lazy_imports import *  # pylint: disable=wildcard-import
import modules
import preprocessing
import viz_utils


# --- Build the RAW MOVi Dataset ---

# Dataset settings (mirrors the notebook defaults)
VARIANT = "e"
RESOLUTION = 256
BATCH_SIZE = 64


def _get_max_obj_num(variant: str) -> int:
  """Max number of objects in the dataset."""
  if variant in ["a", "b", "c"]:
    return 10
  elif variant in ["e", "f"]:
    return 23
  else:
    raise ValueError(f"Invalid MOVi variant: {variant}")


def load_movi():
  """Build the MOVi dataset."""
  ds_name = f"movi_{VARIANT}/{RESOLUTION}x{RESOLUTION}:1.0.0"
  ds_builder = tfds.builder(ds_name, data_dir="/home/chn5pk/fast_901/datasets/")
  ds = ds_builder.as_dataset(split="train", shuffle_files=True)
  ds_iter = iter(ds)
  data = next(ds_iter)
  return ds, data


def main():

  train_ds, sample = load_movi()
  tf_bboxes_3d = sample["instances"]["bboxes_3d"]  # [N, T, 8, 3]
  tf_bboxes_3d = einops.rearrange(tf_bboxes_3d, "n t ... -> t n ...")
  
  # --- Apply Our Data Pre-processing Pipeline ---

  preproc_fn = lambda x: preprocessing.preprocess_gv_movi_example(
      x,
      max_instances=_get_max_obj_num(variant=VARIANT),
      resolution=RESOLUTION,
      drop_cond_prob=0.1,
  )
  train_loader = train_ds.map(preproc_fn).batch(batch_size=BATCH_SIZE)
  train_loader_iter = iter(train_loader)
  batch = next(train_loader_iter)
  batch = viz_utils.to_numpy(batch)


  # --- Build the Neural Assets Model ---

  # We use SD v2.1 as our base generator
  model_name = "stable_diffusion_v2_1"
  generator = diffusion.DiffuserDiffusionWrapper(model_name=model_name)
  hidden_size = 1024  # the cross-attention dim in the denoising U-Net

  # Learnable appearance tokens + pose tokens
  token_dim = hidden_size // 2
  # We will do RoIAlign to extract 2x2 feature maps as object appearance tokens
  roi_align_size = 2
  # We use DINO as our visual encoder
  dino_version, dino_variant = "v1", "B/8"
  conditioning_encoder = modules.ConditioningEncoder(
      appearance_encoder=modules.RoIAlignAppearanceEncoder(
          # +1 because we add a global background bbox
          shape=(_get_max_obj_num(variant=VARIANT) + 1, token_dim),
          image_backbone=modules.DINOViT.from_variant_str(
              version=dino_version,
              variant=dino_variant,
              in_vrange=(0, 1),
              use_imagenet_value_range=True,
              frozen_model=False,  # we fine-tune DINO
          ),
          roi_align_size=roi_align_size,  # feature map size: (28, 28)
          aggregate_method="flatten",  # flatten the 2x2 feature map
      ),
      # Treat 3D bbox as object pose
      object_pose_encoder=modules.MLPPoseEncoder(
          mlp_module=modules.MLP(
              hidden_size=token_dim * 2,
              output_size=token_dim,
              num_hidden_layers=1,
          ),
          # Duplicate bbox tokens to match the length of appearance tokens
          duplicate_factor=roi_align_size**2,
      ),
      # Mask out background pixels when encoding object tokens
      mask_out_bg_for_appearance=True,
      background_value=0.5,
      # Map the relative camera pose with a MLP
      # This serves as the pose token for the background
      background_pos_enc_type="mlp",
  )
  # Fuse appearance and pose tokens with a neck module
  conditioning_neck = modules.FeedForwardNeck(
      feed_forward_module=nn.Dense(hidden_size),
  )

  # Full Model
  ns_model = modules.ControllableGenerator(
      generator=generator,
      conditioning_encoder=conditioning_encoder,
      conditioning_neck=conditioning_neck,
  )

  # --- Model Forward & Loss Computation ---

  input_dict = {
      "tgt_images": batch["tgt_image"],
      "tgt_object_poses": batch["tgt_bboxes_3d"],
      "src_images": batch["src_image"],
      "src_bboxes": batch["src_bboxes"],
      "src_bg_images": batch["src_bg_image"],
  }
  output_dict, params = ns_model.init_with_output(jax.random.key(0), **input_dict)
  # Compute the denoising loss
  loss = (output_dict["diff"] - output_dict["pred_diff"]) ** 2
  loss = loss.mean()
  print("Training loss: ", loss)


if __name__ == "__main__":
  main()
