import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import shutil
import torch
import PIL
import numpy as np
from vfi_utils import load_file_from_github_release
from vfi_models import rife, film

frame_0 = torch.from_numpy(np.array(PIL.Image.open("demo_frames/anime0.png").convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)
frame_1 = torch.from_numpy(np.array(PIL.Image.open("demo_frames/anime1.png").convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)

if os.path.exists("test_result"):
    shutil.rmtree("test_result")

# Test RIFE
print("Testing RIFE VFI...")
vfi_node_class = rife.RIFE_VFI()
ckpt_name = "rife47.pth"
result = vfi_node_class.vfi(
    ckpt_name, 
    torch.cat([frame_0, frame_1], dim=0).cuda(), 
    multiplier=2, 
    clear_cache_after_n_frames=10,
    fast_mode=True,
    ensemble=True,
    scale_factor=1.0
)[0]
print(f"RIFE: Generated {result.size(0)} frames")
print(f"Shape: {result.shape}")

# Test FILM
print("\nTesting FILM VFI...")
vfi_node_class = film.FILM_VFI()
ckpt_name = "film_net_fp32.pt"
result = vfi_node_class.vfi(
    ckpt_name, 
    torch.cat([frame_0, frame_1], dim=0).cuda(), 
    multiplier=2,
    clear_cache_after_n_frames=10
)[0]
print(f"FILM: Generated {result.size(0)} frames")
print(f"Shape: {result.shape}")

print("\nAll tests passed!")
