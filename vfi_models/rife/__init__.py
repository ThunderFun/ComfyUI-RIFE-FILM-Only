import torch
from torch.utils.data import DataLoader
import pathlib
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames, InterpolationStateList
import typing
from comfy.model_management import get_torch_device, soft_empty_cache
import comfy.utils

try:
    from comfy.cli_args import enables_dynamic_vram
    DYNAMIC_VRAM_AVAILABLE = True
except ImportError:
    DYNAMIC_VRAM_AVAILABLE = False
    def enables_dynamic_vram():
        return False
import re
from functools import cmp_to_key
from packaging import version
import gc
import sys
import time

class VFIProgressBar:
    """A progress bar that displays both in ComfyUI UI and terminal"""
    def __init__(self, total, desc="RIFE VFI"):
        self.total = total
        self.n = 0
        self.desc = desc
        self.start_time = time.perf_counter()
        self.comfy_pbar = comfy.utils.ProgressBar(total)
        self._print_terminal()
    
    def update(self, n=1):
        self.n += n
        self.comfy_pbar.update(n)
        self._print_terminal()
    
    def _print_terminal(self):
        if self.total > 0:
            percent = 100 * (self.n / float(self.total))
            bar_length = 40
            filled_length = int(bar_length * self.n // self.total)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f'\r{self.desc}: [{bar}] {percent:.1f}%')
            sys.stdout.flush()
            if self.n >= self.total:
                elapsed = time.perf_counter() - self.start_time
                sys.stdout.write(f'\n{self.desc} completed in {elapsed:.1f}s\n')
                sys.stdout.flush()

MODEL_TYPE = pathlib.Path(__file__).parent.name

CKPT_NAME_VER_DICT = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0",
    "rife42.pth": "4.2",
    "rife43.pth": "4.3",
    "rife44.pth": "4.3",
    "rife45.pth": "4.5",
    "rife46.pth": "4.6",
    "rife47.pth": "4.7",
    "rife48.pth": "4.7",
    "rife49.pth": "4.7",
    "rife417.pth": "4.17",
    "rife426.pth": "4.26",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0"
    # Arch 4.10 doesn't work due to state dict mismatch
    # "rife410.pth": "4.10",
    # "rife411.pth": "4.10",
    # "rife412.pth": "4.10"
}


class RIFE_VFI:
    """Real-Time Intermediate Flow Estimation video frame interpolation.

    RIFE uses an IFNet to directly estimate the intermediate optical flow and
    fusion mask between two frames via a coarse-to-fine stack of IFBlocks, then
    reconstructs the in-between frame. See the RIFE paper in Papers/ for details.
    """

    DESCRIPTION = ("RIFE: Real-time Intermediate Flow Estimation for Video Frame Interpolation. "
                    "Synthesizes new frames between each pair of input frames using a flow-based "
                    "IFNet. Supports arbitrary-timestep interpolation, so it scales to any frame "
                    "multiplier. Hover the individual options below for what each one does.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (
                    sorted(list(CKPT_NAME_VER_DICT.keys()), key=lambda ckpt_name: version.parse(CKPT_NAME_VER_DICT[ckpt_name]), reverse=True),
                    {"default": "rife47.pth", "tooltip":
                        "RIFE model weights to use. The list is sorted newest-first by architecture version. "
                        "Newer versions (e.g. 4.26, 4.17, 4.7) generally produce higher quality, while older "
                        "ones (4.0-4.3) are the only ones that support the RefineNet step controlled by 'fast_mode'. "
                        "The 'sudo_rife4_269...' entry is a community fine-tuned variant built on the 4.0 architecture."}
                ),
                "frames": ("IMAGE", {"tooltip":
                    "The input video frames to interpolate between. Connect an IMAGE batch here. "
                    "At least 2 frames are required; RIFE synthesizes the frames that go between each consecutive pair."}),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip":
                    "How many frame pairs to process before clearing the GPU (CUDA) cache to avoid out-of-memory errors. "
                    "Lower values are safer for high-resolution inputs or large multipliers but slightly slower due to more "
                    "frequent cache clears. Raise it (e.g. 50-100) if you have spare VRAM and want maximum speed."}),
                "multiplier": ("INT", {"default": 2, "min": 1, "tooltip":
                    "How many times to multiply the frame rate. 2x inserts one new frame between each pair (doubles fps); "
                    "4x inserts three, etc. Because RIFE supports arbitrary timesteps via its temporal encoding, large "
                    "multipliers are handled directly without recursively re-interpolating. E.g. 60 input frames x 2 = 120 output."}),
                "fast_mode": ("BOOLEAN", {"default": True, "tooltip":
                    "When ON (default), skips the RefineNet post-processing pass for maximum speed. "
                    "When OFF, enables RefineNet (a ContextNet + U-Net) which refines high-frequency detail and reduces "
                    "artifacts - this roughly doubles compute. NOTE: RefineNet only exists in architectures 4.0, 4.2 and 4.3; "
                    "on 4.5 and newer this option has no effect (those models refine within the IFNet already)."}),
                "ensemble": ("BOOLEAN", {"default": True, "tooltip":
                    "When ON, each IFBlock is run twice (once with the frame order as given, once reversed) and the predicted "
                    "flows/masks are averaged. This improves quality and robustness at the cost of roughly doubling inference time. "
                    "Forced OFF automatically for the 4.26 architecture, which does not implement ensemble averaging."}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0, "tooltip":
                    "Controls the coarse-to-fine resolution pyramid used inside the IFBlocks. Internally the frame is downscaled "
                    "by [8/x, 4/x, 2/x, 1/x] at successive stages (1/x of native at the finest stage). "
                    "1.0 = native RIFE behaviour. Higher values (2.0, 4.0) process at lower internal resolution -> faster and "
                    "lighter on VRAM but blurrier. Lower values (0.5, 0.25) raise the internal resolution -> sharper but slower "
                    "and more memory-hungry."})
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", {"tooltip":
                    "Optional. Connect a 'Make Interpolation State List' node to selectively skip or include specific frame "
                    "pairs for interpolation. If left unconnected, every consecutive pair of frames is interpolated."})
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        fast_mode: bool = True,
        ensemble = False,
        scale_factor = 1.0,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        # Local import of the model definition to avoid circular imports
        from .rife_arch import IFNet

        # Resolve the checkpoint path and instantiate the model
        model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]

        # 4.26 disables ensemble at runtime
        if arch_ver == "4.26":
            ensemble = False

        interpolation_model = IFNet(arch_ver=arch_ver)
        
        # Use assign=True for dynamic VRAM support (zero-copy loading)
        assign_enabled = DYNAMIC_VRAM_AVAILABLE and enables_dynamic_vram()

        state_dict = torch.load(model_path)
        # Some RIFE checkpoints bundle training-only modules that are not part of
        # the inference IFNet (e.g. a "teacher" network used for knowledge
        # distillation and a "caltime" timestep-calibration MLP). These show up
        # as unexpected keys and make load_state_dict fail. Strip any key that
        # does not map to a model parameter/buffer so we can keep strict loading,
        # which still catches genuine parameter mismatches.
        model_keys = set(interpolation_model.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
        interpolation_model.load_state_dict(state_dict, assign=assign_enabled)

        device = get_torch_device()
        interpolation_model.eval().to(device)

        # Free the CPU checkpoint copy now that the model lives on-device.
        del state_dict
        gc.collect()

        # Convert input frames from NHWC to NCHW and ensure float32 dtype
        frames = preprocess_frames(frames)
        dtype = torch.float32

        # Prepare per-frame multipliers (one per frame pair)
        num_pairs = len(frames) - 1
        if isinstance(multiplier, int):
            multipliers = [int(multiplier)] * num_pairs
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (num_pairs - len(multipliers))

        # Scale list for multi-scale processing (4.26 uses 5 stages instead of 4)
        if arch_ver == "4.26":
            scale_list = [16 / scale_factor, 8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        else:
            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        frames_processed_since_cache_clear = 0

        # Build a list of interpolation tasks across all frame pairs. Each task is a
        # tuple of (pair_idx, dt) representing the pair index and timestep fraction.
        # Pairs that are skipped via optional_interpolation_states have no tasks.
        tasks: typing.List[typing.Tuple[int, float]] = []
        num_tasks_per_pair: typing.Dict[int, int] = {}
        # Output layout: for each non-skipped pair we reserve a run of slots for
        # its interpolated frames. mid_offsets[pair_idx] is the output index where
        # the first interpolated frame for that pair should be written.
        mid_offsets: typing.Dict[int, int] = {}
        total_output = 0
        for pair_idx in range(len(frames) - 1):
            total_output += 1  # leading original frame
            if optional_interpolation_states is not None and optional_interpolation_states.is_frame_skipped(pair_idx):
                num_tasks_per_pair[pair_idx] = 0
                continue
            m = multipliers[pair_idx]
            n = max(m - 1, 0)
            num_tasks_per_pair[pair_idx] = n
            if n > 0:
                mid_offsets[pair_idx] = total_output
            total_output += n
            for step in range(1, m):
                tasks.append((pair_idx, step / m))
        total_output += 1  # trailing original frame

        # Preallocate a single contiguous output buffer. Interpolated frames are
        # written directly into their final positions during the task loop, avoiding
        # an intermediate dict accumulation and torch.cat at the end.
        output_frames = torch.zeros(
            total_output, *frames.shape[1:], dtype=dtype, device="cpu"
        )

        # Place every original frame in its final position up front (cheap copies).
        fill_pos = 0
        for pair_idx in range(len(frames) - 1):
            output_frames[fill_pos] = frames[pair_idx]
            fill_pos += 1
            if pair_idx in mid_offsets:
                fill_pos += multipliers[pair_idx] - 1
        output_frames[fill_pos] = frames[-1]

        mid_written: typing.Dict[int, int] = {pair_idx: 0 for pair_idx in mid_offsets}

        pbar = VFIProgressBar(len(tasks), desc="RIFE VFI")

        pos = 0
        while pos < len(tasks):
            # Always process a single task at a time since batching is disabled.
            batch_tasks = tasks[pos : pos + 1]
            frame0_list: typing.List[torch.Tensor] = []
            frame1_list: typing.List[torch.Tensor] = []
            timestep_list: typing.List[float] = []
            for (pair_idx, dt) in batch_tasks:
                frame0_list.append(frames[pair_idx:pair_idx+1])
                frame1_list.append(frames[pair_idx+1:pair_idx+2])
                timestep_list.append(dt)
            # Move frames to device
            frame0_batch = torch.cat(frame0_list, dim=0).to(device).to(dtype)
            frame1_batch = torch.cat(frame1_list, dim=0).to(device).to(dtype)
            timestep_tensor = torch.tensor(timestep_list, dtype=dtype, device=device).view(-1, 1, 1, 1)

            with torch.no_grad():
                middle_frames = interpolation_model(
                    frame0_batch,
                    frame1_batch,
                    timestep_tensor,
                    scale_list,
                    False,       # training=False (inference mode)
                    fast_mode,   # fastmode
                    ensemble,    # ensemble
                ).clamp(0, 1)

            middle_frames_cpu = middle_frames.detach().to(dtype=dtype, device="cpu")

            for idx, (pair_idx, _dt) in enumerate(batch_tasks):
                write_pos = mid_offsets[pair_idx] + mid_written[pair_idx]
                output_frames[write_pos] = middle_frames_cpu[idx]
                mid_written[pair_idx] += 1
                num_tasks_per_pair[pair_idx] -= 1
                if num_tasks_per_pair[pair_idx] == 0:
                    frames_processed_since_cache_clear += 1
                    if frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                        soft_empty_cache()
                        frames_processed_since_cache_clear = 0
                        gc.collect()
            pbar.update(len(batch_tasks))
            pos += len(batch_tasks)

        # Free the model and GPU cache before the final CPU-side rearrange.
        del interpolation_model
        soft_empty_cache()
        gc.collect()

        # postprocess_frames allocates a second full-size tensor (NCHW -> NHWC).
        # Release the original buffer immediately to keep peak RAM at ~2x.
        out_images = postprocess_frames(output_frames)
        del output_frames
        gc.collect()
        return (out_images,)
