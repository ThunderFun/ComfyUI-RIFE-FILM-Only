import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import bisect
import numpy as np
import typing
from vfi_utils import (
    InterpolationStateList,
    load_file_from_github_release,
    preprocess_frames,
    postprocess_frames,
)
import pathlib
import gc
import comfy.utils
import sys
import time
import warnings

warnings.filterwarnings(
    "ignore",
    message="Using padding='same' with even kernel lengths and odd dilation",
)

class VFIProgressBar:
    def __init__(self, total, desc="FILM VFI"):
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
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            sys.stdout.write(f"\r{self.desc}: [{bar}] {percent:.1f}%")
            sys.stdout.flush()
            if self.n >= self.total:
                elapsed = time.perf_counter() - self.start_time
                sys.stdout.write(f"\n{self.desc} completed in {elapsed:.1f}s\n")
                sys.stdout.flush()


MODEL_TYPE = pathlib.Path(__file__).parent.name
DEVICE = get_torch_device()
MODEL_CACHE = {}


def clear_model_cache():
    global MODEL_CACHE
    for ckpt_name in list(MODEL_CACHE.keys()):
        model, _ = MODEL_CACHE[ckpt_name]
        del model
        del MODEL_CACHE[ckpt_name]
    MODEL_CACHE = {}
    soft_empty_cache()
    gc.collect()


def build_bisection_schedule(inter_frames: int) -> list:
    """Precompute the greedy bisection schedule for subdividing a frame pair.

    Returns a list of (start_idx, end_idx, target, dt) tuples that mirrors the
    exact ordering the original per-step greedy loop would produce. Uses
    ``torch.linspace`` (fp32) for numeric consistency with the upstream code.
    """
    if inter_frames <= 0:
        return []

    n_total = inter_frames + 2
    splits = torch.linspace(0, 1, n_total)

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))
    schedule: list = []

    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None])
                     / (ends[:, None] - starts[:, None]) - 0.5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        dt_val = ((splits[remains[step]] - splits[idxes[start_i]])
                  / (splits[idxes[end_i]] - splits[idxes[start_i]])).item()
        schedule.append((idxes[start_i], idxes[end_i], remains[step], float(dt_val)))

        insert_pos = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_pos, remains[step])
        del remains[step]

    return schedule


@torch.inference_mode()
def inference(
    model,
    img_batch_1: torch.Tensor,
    img_batch_2: torch.Tensor,
    inter_frames: int,
    model_dtype: torch.dtype,
    device: torch.device,
    schedule: list | None = None,
    forward_fn=None,
) -> list:
    """Generate ``inter_frames`` intermediate frames between two inputs.

    Processes tasks sequentially (one model forward per intermediate frame).
    The schedule is precomputed on CPU by ``build_bisection_schedule``.

    Parameters
    ----------
    schedule : precomputed bisection schedule; ``None`` to compute internally.
    forward_fn : ``(x0, x1, dt) -> Tensor``; ``None`` to use ``model``.
    """
    if schedule is None:
        schedule = build_bisection_schedule(inter_frames)
    forward = forward_fn or (lambda x0, x1, dt: model(x0, x1, dt))

    n_total = inter_frames + 2
    results: list[torch.Tensor | None] = [None] * n_total
    results[0] = img_batch_1
    results[n_total - 1] = img_batch_2

    # FILM is trained for midpoint interpolation only (t=0.5).
    dt_half = torch.tensor([[0.5]], device=device, dtype=model_dtype)

    for start, end, target, _dt in schedule:
        pred = forward(results[start], results[end], dt_half)
        results[target] = pred.clamp(0, 1)

    return [r for r in results if r is not None]


class FILM_VFI:
    """Frame Interpolation for Large Motion (FILM) video frame interpolation.

    FILM uses a multi-scale pyramid network to synthesize the midpoint frame
    between two inputs (t=0.5 only). For multipliers > 2 it recursively
    subdivides using a bisection schedule, always picking the largest remaining
    gap first.
    """

    DESCRIPTION = ("FILM: Frame Interpolation for Large Motion. Synthesizes new frames between each pair "
                    "of input frames using a multi-scale pyramid network. Unlike RIFE, FILM only supports "
                    "midpoint interpolation (t=0.5) and recursively subdivides to reach higher multipliers. "
                    "Works well for large motions. Hover the options below for details.")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt", "film_net_fp16.pt"], {"tooltip":
                    "Model precision variant. fp32 (full precision) produces slightly more accurate results; "
                    "fp16 (half precision) uses less VRAM and runs faster on GPUs with good fp16 support."}),
                "frames": ("IMAGE", {"tooltip":
                    "The input video frames to interpolate between. Connect an IMAGE batch here. "
                    "At least 2 frames are required; FILM synthesizes the frames that go between each consecutive pair."}),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000, "tooltip":
                    "How many frame pairs to process before clearing the GPU (CUDA) cache to avoid out-of-memory errors. "
                    "Lower values are safer for high-resolution inputs or large multipliers but slightly slower due to more "
                    "frequent cache clears. Raise it (e.g. 50-100) if you have spare VRAM and want maximum speed."}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000, "tooltip":
                    "How many times to multiply the frame rate. 2x inserts one new frame between each pair (doubles fps); "
                    "4x inserts three, etc. FILM only does midpoint interpolation (t=0.5), so for multipliers > 2 it uses a "
                    "bisection schedule that recursively subdivides the largest remaining gap. E.g. 60 input frames x 2 = 120 output."}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", {"tooltip":
                    "Optional. Connect a 'Make Interpolation State List' node to selectively skip or include specific frame "
                    "pairs for interpolation. If left unconnected, every consecutive pair of frames is interpolated."})
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    @torch.inference_mode()
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames=10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs,
    ):
        interpolation_states = optional_interpolation_states
        device = get_torch_device()

        # Model loading (cached)
        if ckpt_name not in MODEL_CACHE:
            soft_empty_cache()
            gc.collect()

            model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval()
            model = model.to(device)

            try:
                model_dtype = next(model.parameters()).dtype
            except StopIteration:
                model_dtype = torch.float16 if "fp16" in ckpt_name else torch.float32

            MODEL_CACHE[ckpt_name] = (model, model_dtype)

        model, model_dtype = MODEL_CACHE[ckpt_name]
        output_dtype = torch.float32

        frames_nchw = preprocess_frames(frames)
        num_input_frames = len(frames_nchw)

        if isinstance(multiplier, int):
            multipliers = [multiplier] * (num_input_frames - 1)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (num_input_frames - len(multipliers) - 1)

        total_output_frames = sum(multipliers) + 1
        output_frames = torch.zeros(
            total_output_frames, *frames_nchw.shape[1:], dtype=output_dtype, device="cpu"
        )
        out_len = 0

        pbar = VFIProgressBar(num_input_frames - 1, desc="FILM VFI")
        frames_processed = 0

        for frame_itr in range(num_input_frames - 1):
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                output_frames[out_len] = frames_nchw[frame_itr : frame_itr + 1]
                out_len += 1
                pbar.update(1)
                continue

            n_inter = multipliers[frame_itr] - 1

            frame_0 = frames_nchw[frame_itr : frame_itr + 1].to(device, non_blocking=True).to(model_dtype)
            frame_1 = frames_nchw[frame_itr + 1 : frame_itr + 2].to(device, non_blocking=True).to(model_dtype)

            schedule = build_bisection_schedule(n_inter)
            results = inference(
                model, frame_0, frame_1, n_inter, model_dtype, device,
                schedule=schedule,
            )

            for i, f in enumerate(results[:-1]):
                output_frames[out_len] = f.detach().to(dtype=output_dtype)
                out_len += 1

            del results, frame_0, frame_1

            frames_processed += 1
            if frames_processed >= clear_cache_after_n_frames:
                soft_empty_cache()
                frames_processed = 0

            pbar.update(1)

        # Final frame
        output_frames[out_len] = frames_nchw[-1:].to(dtype=output_dtype)
        out_len += 1

        soft_empty_cache()
        gc.collect()

        return (postprocess_frames(output_frames[:out_len]),)


NODE_CLASS_MAPPINGS = {
    "FILM VFI": FILM_VFI,
}
