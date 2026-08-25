import torch
import pathlib
from vfi_utils import (
    load_file_from_github_release,
    preprocess_frames,
    InterpolationStateList,
    VFIProgressBar,
)
import typing
import operator
from packaging import version
from comfy.model_management import get_torch_device, soft_empty_cache

try:
    from comfy.cli_args import enables_dynamic_vram
    DYNAMIC_VRAM_AVAILABLE = True
except ImportError:
    DYNAMIC_VRAM_AVAILABLE = False
    def enables_dynamic_vram():
        return False

try:
    from comfy.model_management import processing_interrupted, InterruptProcessingException
except ImportError:
    def processing_interrupted():
        return False
    class InterruptProcessingException(Exception):
        pass

import gc

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

# Cached models: ckpt_name -> (model, model_dtype). Mirrors the FILM node's
# MODEL_CACHE so repeated runs skip the disk read, the weight copy and the
# host->device upload entirely. One entry per ckpt: switching precision
# evicts the previous copy instead of holding both.
MODEL_CACHE = {}

PRECISION_OPTIONS = ["fp32", "bf16", "fp16"]
_DTYPE_BY_PRECISION = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def clear_model_cache():
    """Drop every cached RIFE model and its warp grids."""
    global MODEL_CACHE
    from .rife_arch import clear_warp_grid_cache
    for ckpt_name in list(MODEL_CACHE.keys()):
        model, _ = MODEL_CACHE.pop(ckpt_name)
        del model
    clear_warp_grid_cache()
    soft_empty_cache()
    gc.collect()


def _resolve_model_dtype(precision, device):
    """Map the precision option to a torch dtype, with hardware fallbacks.

    bf16 is offered on Ampere+ (fp32 exponent range, no flow overflow at
    large motion); fp16 is the legacy fast path and can overflow flows for
    extreme motion at high resolution. Non-CUDA devices always run fp32.
    """
    if device.type != "cuda":
        if precision != "fp32":
            print(f"RIFE VFI: precision '{precision}' requires CUDA; forcing fp32.")
        return torch.float32
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("RIFE VFI: bf16 not supported on this GPU; forcing fp32.")
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _load_model(ckpt_name, model_dtype, device):
    """Build the IFNet for a ckpt and load its weights on CPU.

    Returns the model on `device` in `model_dtype`. Loading happens on CPU
    (map_location="cpu") so a CUDA-saved checkpoint can't spike VRAM during
    the state-dict copy, and the dtype cast runs before the device transfer
    so the GPU never holds both fp32 and low-precision weight copies.
    """
    from .rife_arch import IFNet

    model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)

    interpolation_model = IFNet(arch_ver=CKPT_NAME_VER_DICT[ckpt_name])

    # assign=True is zero-copy loading (the params ARE the state-dict
    # tensors) used by ComfyUI's dynamic-VRAM mode. Only meaningful in fp32
    # and only on the first load of a cached model.
    assign_enabled = (
        DYNAMIC_VRAM_AVAILABLE
        and enables_dynamic_vram()
        and model_dtype == torch.float32
    )

    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        # Older torch or a checkpoint pickling non-tensor objects (the
        # community sudo_rife4_269 build). Fall back to the legacy loader.
        state_dict = torch.load(model_path, map_location="cpu")

    # Some checkpoints bundle training-only modules (a "teacher" network and
    # a "caltime" timestep-calibration MLP). Strip keys that do not map to a
    # model parameter/buffer so strict loading still catches real mismatches.
    model_keys = set(interpolation_model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    interpolation_model.load_state_dict(state_dict, assign=assign_enabled)

    interpolation_model.eval()
    if model_dtype != torch.float32:
        interpolation_model = interpolation_model.to(dtype=model_dtype)
    interpolation_model = interpolation_model.to(device)

    del state_dict
    gc.collect()
    return interpolation_model


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
                    "Upper bound on how many frame pairs to process between GPU cache clears. The cache is also cleared "
                    "automatically when free VRAM drops below ~15%, so this is a safety cap rather than the only trigger. "
                    "Lower values are safer for high-resolution inputs or large multipliers but slightly slower; raise it "
                    "(e.g. 50-100) if you have spare VRAM and want maximum speed."}),
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
                    "and more memory-hungry."}),
                "precision": (PRECISION_OPTIONS, {"default": "fp32", "tooltip":
                    "Inference precision. fp32 (default) is the reference behaviour. bf16 (Ampere+ GPUs only) halves activation "
                    "and weight VRAM and runs faster, with quality close to fp32; it shares fp32's exponent range, so large-motion "
                    "flows do not overflow. fp16 is a legacy fast path that can overflow extreme motion at high resolution - "
                    "prefer bf16 when available. On non-CUDA devices this option is ignored and fp32 is used."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip":
                    "Keep the model in VRAM between runs so repeated executions skip loading entirely (like the FILM node). "
                    "Disable this if you are short on VRAM or run under ComfyUI's dynamic-VRAM mode and want the model released "
                    "after each run."}),
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
        clear_cache_after_n_frames: int = 10,
        multiplier: typing.SupportsInt = 2,
        fast_mode: bool = True,
        ensemble: bool = False,
        scale_factor: float = 1.0,
        precision: str = "fp32",
        keep_model_loaded: bool = True,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        device = get_torch_device()
        arch_ver = CKPT_NAME_VER_DICT[ckpt_name]
        if arch_ver == "4.26":
            ensemble = False

        model_dtype = _resolve_model_dtype(precision, device)

        # Model loading (cached per ckpt and precision)
        cached = MODEL_CACHE.get(ckpt_name)
        if cached is not None and cached[1] == model_dtype:
            interpolation_model = cached[0]
        else:
            if cached is not None:
                # Precision switch: drop the old copy before loading the new one.
                model, _ = MODEL_CACHE.pop(ckpt_name)
                del model
                gc.collect()
            interpolation_model = _load_model(ckpt_name, model_dtype, device)
            if keep_model_loaded:
                MODEL_CACHE[ckpt_name] = (interpolation_model, model_dtype)

        # ComfyUI normally hands us CPU tensors, but tolerate GPU-resident inputs.
        if frames.device.type != "cpu":
            frames = frames.cpu()

        # Convert input frames from NHWC to NCHW (a strided view, no copy).
        frames_nchw = preprocess_frames(frames)
        num_pairs = len(frames_nchw) - 1
        if num_pairs < 1:
            raise ValueError("RIFE VFI requires at least 2 input frames.")

        # multiplier may arrive as int, numpy integer, or a list of per-pair ints.
        try:
            m_scalar = operator.index(multiplier)
            multipliers = [m_scalar] * num_pairs
        except TypeError:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (num_pairs - len(multipliers))

        # Scale list for multi-scale processing (4.26 uses 5 stages instead of 4)
        if arch_ver == "4.26":
            scale_list = [16 / scale_factor, 8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        else:
            scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]

        h, w = frames_nchw.shape[2], frames_nchw.shape[3]
        c = frames_nchw.shape[1]

        # Task layout: for each non-skipped pair with multiplier > 1, reserve a
        # run of output slots for its interpolated frames.
        mid_offsets = {}
        num_mids = {}
        pair_tasks = []  # (pair_idx, [dt, ...])
        total_output = 0
        for pair_idx in range(num_pairs):
            total_output += 1  # leading original frame
            m = multipliers[pair_idx]
            if (optional_interpolation_states is not None
                    and optional_interpolation_states.is_frame_skipped(pair_idx)) or m <= 1:
                continue
            dts = [step / m for step in range(1, m)]
            mid_offsets[pair_idx] = total_output
            num_mids[pair_idx] = len(dts)
            total_output += len(dts)
            pair_tasks.append((pair_idx, dts))
        total_output += 1  # trailing original frame

        # Output buffer is allocated directly in NHWC fp32 (ComfyUI IMAGE
        # layout), so the final rearrange in postprocess_frames (a second
        # full-size tensor) never happens. Every slot is written exactly once.
        output_frames = torch.empty(total_output, h, w, c, dtype=torch.float32, device="cpu")

        # Place every original frame in its final position up front.
        frames_nhwc3 = frames[..., :3]
        fill_pos = 0
        for pair_idx in range(num_pairs):
            output_frames[fill_pos] = frames_nhwc3[pair_idx]
            fill_pos += 1
            fill_pos += num_mids.get(pair_idx, 0)
        output_frames[fill_pos] = frames_nhwc3[-1]

        total_tasks = sum(len(dts) for _, dts in pair_tasks)
        pbar = VFIProgressBar(total_tasks, desc="RIFE VFI")

        # Pinned staging buffer for GPU->CPU transfers (CUDA only): the D2H
        # copy goes into pinned memory, then the output slot write is a CPU
        # memcpy with the dtype cast (bf16/fp16 -> fp32).
        staging = None
        if device.type == "cuda":
            staging = torch.empty(c, h, w, dtype=model_dtype, pin_memory=True)

        frames_processed_since_cache_clear = 0
        gpu_frame = None       # last uploaded frame (carry-over)
        gpu_frame_idx = -1

        try:
            with torch.inference_mode():
                for pair_idx, dts in pair_tasks:
                    # Reuse the previous pair's frame1 as this pair's frame0,
                    # cutting host->device traffic roughly in half. Uploads are
                    # ordered (pairs are processed in order), so the carry-over
                    # is valid even when pairs in between are skipped.
                    if gpu_frame is not None and gpu_frame_idx == pair_idx:
                        frame0_batch = gpu_frame
                    else:
                        frame0_batch = frames_nchw[pair_idx:pair_idx + 1].to(
                            device=device, dtype=model_dtype
                        )
                    frame1_batch = frames_nchw[pair_idx + 1:pair_idx + 2].to(
                        device=device, dtype=model_dtype
                    )
                    # One device-side timestep tensor per pair, sliced per task.
                    timesteps = torch.tensor(dts, dtype=model_dtype, device=device).view(-1, 1, 1, 1)

                    write_pos = mid_offsets[pair_idx]
                    for i, _dt in enumerate(dts):
                        middle_frames = interpolation_model(
                            frame0_batch,
                            frame1_batch,
                            timesteps[i:i + 1],
                            scale_list,
                            False,       # training=False (inference mode)
                            fast_mode,   # fastmode
                            ensemble,    # ensemble
                        ).clamp(0, 1)

                        if staging is not None:
                            staging.copy_(middle_frames[0], non_blocking=True)
                            torch.cuda.current_stream(device).synchronize()
                            output_frames[write_pos + i].copy_(staging.permute(1, 2, 0))
                        else:
                            output_frames[write_pos + i].copy_(
                                middle_frames[0]
                                .permute(1, 2, 0)
                                .to(device="cpu", dtype=torch.float32)
                            )
                        pbar.update(1)

                    gpu_frame = frame1_batch
                    gpu_frame_idx = pair_idx + 1

                    # Cache clear: a hard cap on processed pairs plus automatic
                    # clearing under VRAM pressure (the allocator reuses blocks
                    # in steady state, so this only fires when needed or on the cap).
                    frames_processed_since_cache_clear += 1
                    pressure = False
                    if device.type == "cuda":
                        free_b, total_b = torch.cuda.mem_get_info(device)
                        pressure = free_b < total_b * 0.15
                    if pressure or frames_processed_since_cache_clear >= clear_cache_after_n_frames:
                        soft_empty_cache()
                        frames_processed_since_cache_clear = 0
                        gc.collect()

                    if processing_interrupted():
                        raise InterruptProcessingException()
        finally:
            # Drop references so a cancelled/errored run releases VRAM
            # deterministically instead of waiting on GC.
            gpu_frame = None
            staging = None
            cached = None
            if not keep_model_loaded:
                from .rife_arch import clear_warp_grid_cache
                MODEL_CACHE.pop(ckpt_name, None)
                del interpolation_model
                clear_warp_grid_cache()
            if device.type == "cuda":
                soft_empty_cache()
            gc.collect()

        return (output_frames,)