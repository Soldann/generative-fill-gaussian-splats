import torch
import numpy as np
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import json
from pathlib import Path

ROTATION_STEPS = 321
OUTPUT_DIR = Path("submodules/lyra/Lyra-2/assets/custom_trajectory_examples/example_3")

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = [OUTPUT_DIR / "eth.jpg"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # remove batch dimension
    extrinsic = extrinsic.squeeze(0)
    intrinsic = intrinsic.squeeze(0)

    print(extrinsic.shape, intrinsic.shape)
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]

    extrinsic_w2c = torch.eye(4, device=extrinsic.device).repeat(len(extrinsic), 1, 1)
    extrinsic_w2c[:, :3, :3] = R.transpose(1, 2)
    extrinsic_w2c[:, :3, 3] = -(R.transpose(1, 2) @ t.unsqueeze(-1)).squeeze(-1)

if extrinsic_w2c.shape[0] == 1:
    # Generate a set of camera poses by rotating the camera 360 degrees around the y axis
    extrinsic_w2c = extrinsic_w2c.repeat(ROTATION_STEPS, 1, 1)
    # Rotate the last pose around the y axis ROTATION_STEPS times to generate a full 360 degree rotation
    angles = torch.linspace(0, 2 * torch.pi, ROTATION_STEPS, device=extrinsic_w2c.device)
    for i in range(1, ROTATION_STEPS):
        R_y = torch.tensor([[torch.cos(angles[i]), 0, torch.sin(angles[i])],
                            [0, 1, 0],
                            [-torch.sin(angles[i]), 0, torch.cos(angles[i])]], device=extrinsic_w2c.device)
        extrinsic_w2c[i, :3, :3] = R_y @ extrinsic_w2c[0, :3, :3]
        extrinsic_w2c[i, :3, 3] = extrinsic_w2c[0, :3, 3]

    intrinsic = intrinsic.repeat(ROTATION_STEPS, 1, 1)

image_height = images.shape[-2]
image_width = images.shape[-1]

captions = {str(i): f"" for i in range(extrinsic_w2c.shape[0])} # empty captions for each pose

trajectory = {
    "w2c": extrinsic_w2c.cpu(),
    "intrinsics": intrinsic.cpu(),
    "image_height": image_height,
    "image_width": image_width
}

np.savez(OUTPUT_DIR / "trajectory.npz", **trajectory)

with open(OUTPUT_DIR / "captions.json", "w") as f:
    json.dump(captions, f)

print("DONE!")