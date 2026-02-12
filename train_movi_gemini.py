# --- Imports ---

import os
# Set CUDA_VISIBLE_DEVICES to '0' for the first GPU, '1' for the second, etc.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
import optax # Added for optimization
from flax.training import checkpoints
import diffusion
from etils.lazy_imports import * # pylint: disable=wildcard-import
import modules
import preprocessing
import viz_utils
from flax import traverse_util


# --- Build the RAW MOVi Dataset ---

# Dataset settings (mirrors the notebook defaults)
VARIANT = "e"
RESOLUTION = 256
BATCH_SIZE = 2


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

  print(f"JAX Backend: {jax.default_backend()}")
  print(f"JAX Devices: {jax.devices()}")

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
  # Repeat the dataset to allow for continuous training loop
  train_loader = train_ds.map(preproc_fn, num_parallel_calls=tf.data.AUTOTUNE).repeat().batch(batch_size=BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
  
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
              frozen_model=False,  # we fine-tune DINO (Correctly set to False)
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

  # --- Initialize Parameters ---
  
  # We grab a single batch just to initialize shapes
  init_batch = viz_utils.to_numpy(next(iter(train_loader)))
  init_inputs = {
      "tgt_images": init_batch["tgt_image"],
      "tgt_object_poses": init_batch["tgt_bboxes_3d"],
      "src_images": init_batch["src_image"],
      "src_bboxes": init_batch["src_bboxes"],
      "src_bg_images": init_batch["src_bg_image"],
  }
  
  # Initialize parameters (including DINO and Neural Assets components)
  rng = jax.random.key(0)
  init_rng, train_rng = jax.random.split(rng)
  _, params = ns_model.init_with_output(init_rng, **init_inputs)

  print("DEBUG: Forcing parameters to GPU...")
  try:
    # This forces the arrays to VRAM. 
    # If this fails or VRAM doesn't jump to >4GB, your CUDA/cuDNN is broken.
    params = jax.device_put(params, jax.devices()[0])
    print("DEBUG: Success! Parameters are on GPU.")
  except Exception as e:
    print(f"DEBUG: FAILED to put params on GPU. Error: {e}")
    exit()

  # --- Optimizer Setup ---
  
  # 1. Define Schedulers
  # Linearly warm up for 1000 steps, then stay constant
  warmup_steps = 1000
    
  # Low LR for pre-trained backbones (Image Generator + Visual Encoder)
  backbone_lr_schedule = optax.warmup_constant_schedule(
    init_value=0.0, 
    peak_value=5e-5, 
    warmup_steps=warmup_steps
  )
    
  # High LR for new layers (MLPs, Linear Projections)
  rest_lr_schedule = optax.warmup_constant_schedule(
    init_value=0.0, 
    peak_value=1e-3, 
    warmup_steps=warmup_steps
  )

  # 2. Partition Parameters (Create a label tree)
  # We flatten the params dictionary to inspect keys easily
  flat_params = traverse_util.flatten_dict(params)
  flat_labels = {}

  for path, _ in flat_params.items():
    # 'path' is a tuple of keys, e.g., ('generator', 'unet', 'conv_in', ...)
    # We check top-level keys or specific sub-module names
        
    # 'generator' covers the Stable Diffusion Wrapper
    # 'image_backbone' covers the DINO visual encoder inside conditioning_encoder
    if 'generator' in path or 'image_backbone' in path:
      flat_labels[path] = 'backbone'
    else:
      # Everything else (MLPs, Neck, Pose Encoders) gets the higher LR
      flat_labels[path] = 'rest'

  # Unflatten to recreate the PyTree structure matching 'params'
  param_labels = traverse_util.unflatten_dict(flat_labels)

  # 3. Create the Optimizer Chain
  tx = optax.chain(
    # Apply gradient clipping of 1.0 globally
    optax.clip_by_global_norm(1.0),
    # Apply specific optimizers based on the labels we defined above
    optax.multi_transform(
      {
        'backbone': optax.adamw(learning_rate=backbone_lr_schedule),
        'rest':     optax.adamw(learning_rate=rest_lr_schedule),
      },
      param_labels
    )
  )
  opt_state = tx.init(params)

  # --- Checkpointing Setup ---

  CKPT_DIR = os.path.abspath("/home/chn5pk/fast_901/checkpoints/")
  if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

  # --- Training Step Definition ---

  @jax.jit
  def train_step(params, opt_state, batch, rng):
      
      def loss_fn(params):
          # Reconstruct input dict from batch
          inputs = {
              "tgt_images": batch["tgt_image"],
              "tgt_object_poses": batch["tgt_bboxes_3d"],
              "src_images": batch["src_image"],
              "src_bboxes": batch["src_bboxes"],
              "src_bg_images": batch["src_bg_image"],
          }
          
          # RNG handling: split for dropout/diffusion noise if required by internals
          step_rng = jax.random.fold_in(rng, 0)
          
          # Run model forward pass
          # We use 'sample' or 'dropout' rngs depending on model internals, 
          # passing generic 'params' rng covers most Flax bases.
          output_dict = ns_model.apply(params, **inputs, rngs={'params': step_rng, 'dropout': step_rng})
          
          # Compute loss
          loss = (output_dict["diff"] - output_dict["pred_diff"]) ** 2
          return loss.mean()

      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, new_opt_state = tx.update(grads, opt_state, params)
      new_params = optax.apply_updates(params, updates)
      return new_params, new_opt_state, loss

  # --- Training Loop ---
  
  print("Starting training...")
  MAX_STEPS = 10000
  
  for step, batch in enumerate(train_loader):
      if step >= MAX_STEPS:
          break
          
      batch = viz_utils.to_numpy(batch)
      train_rng, step_rng = jax.random.split(train_rng)
      
      params, opt_state, loss = train_step(params, opt_state, batch, step_rng)
      
      if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

      # --- SAVE CHECKPOINT EVERY 1000 STEPS ---
      if step > 0 and (step + 1) % 1000 == 0:
        print(f"Saving checkpoint at step {step+1}...")
        # We bundle params and opt_state together to save them in one file/structure
        state_to_save = {'params': params, 'opt_state': opt_state}
        checkpoints.save_checkpoint(
          ckpt_dir=CKPT_DIR,
          target=state_to_save,
          step=step+1,
          keep=100,
          overwrite=True
        )

if __name__ == "__main__":
  main()