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
    compute_fps_schedule,
    resolve_fps_mode,
    multiplier_output_fps,
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

# Verdicts of _model_responds_to_dt, keyed by id(model) with the model held
# as a value so the id cannot be recycled while the verdict is alive.
_DT_RESPONSE_CACHE = {}


def clear_model_cache():
    global MODEL_CACHE
    for ckpt_name in list(MODEL_CACHE.keys()):
        model, _ = MODEL_CACHE[ckpt_name]
        del model
        del MODEL_CACHE[ckpt_name]
    MODEL_CACHE = {}
    _DT_RESPONSE_CACHE.clear()
    soft_empty_cache()
    gc.collect()


def _model_responds_to_dt(model, device, model_dtype):
    """Empirically check whether a model's output depends on its dt input.

    See _load_timeaware_model for why the shipped export cannot be used
    directly. Models that do respond are used as-is.
    """
    verdict = _DT_RESPONSE_CACHE.get(id(model))
    if verdict is not None:
        return verdict[1]
    g = torch.Generator().manual_seed(0)
    x0 = torch.rand(1, 3, 64, 64, generator=g).to(device, model_dtype)
    x1 = torch.rand(1, 3, 64, 64, generator=g).to(device, model_dtype)
    with torch.inference_mode():
        a = model(x0, x1, torch.tensor([[0.25]], device=device, dtype=model_dtype))
        b = model(x0, x1, torch.tensor([[0.75]], device=device, dtype=model_dtype))
    responds = (a - b).abs().max().item() > 1e-6
    _DT_RESPONSE_CACHE[id(model)] = (model, responds)
    return responds


def _load_timeaware_model(ckpt_name, device):
    """Re-host the shipped FILM weights in a dt-wired architecture.

    The released film_net TorchScript export was trained only at t=0.5, so
    it hard-wires its flow-pyramid scaling to 0.5 and ignores the dt input
    entirely; exact fps retiming (frames at t != 0.5) is impossible with it
    directly. The same weights are loaded into the local Interpolator with
    fixed_midpoint=False, which scales the flow pyramids by the requested
    t, giving positionally exact frames for any ratio. Frames at timesteps
    other than 0.5 are extrapolations (quality is validated by
    tests/test_fps_real_model.py).

    The cached entry lives in MODEL_CACHE under "<ckpt>:timeaware" so
    clear_model_cache() releases it together with the base model.

    Raises RuntimeError if the checkpoint's weights don't map onto the
    local architecture (e.g. a foreign checkpoint format).
    """
    cache_key = f"{ckpt_name}:timeaware"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    from .film_arch import Interpolator

    model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
    jit_model = torch.jit.load(model_path, map_location="cpu")
    model_dtype = next(jit_model.parameters()).dtype

    rehosted = Interpolator(compile=False, fixed_midpoint=False)
    local_keys = set(rehosted.state_dict().keys())
    shipped_sd = {k: v for k, v in jit_model.state_dict().items() if k in local_keys}
    # Every real weight must transfer. The only permitted gaps are the two
    # dummy dtype/device marker buffers that exist solely for TorchScript.
    uncovered = [k for k in local_keys
                 if k not in shipped_sd
                 and not k.startswith("extract.extract_sublevels.target_")]
    if uncovered:
        raise RuntimeError(
            f"FILM VFI: checkpoint '{ckpt_name}' is not compatible with exact "
            f"fps mode ({len(uncovered)} weights missing: {uncovered[:3]}...). "
            f"Use the 'multiplier' input instead of source_fps/target_fps, or "
            f"switch to RIFE VFI for fractional frame-rate conversion.")
    rehosted.load_state_dict(shipped_sd, strict=False)
    rehosted.eval()
    rehosted = rehosted.to(dtype=model_dtype).to(device)

    print(f"FILM VFI: '{ckpt_name}' ignores its timestep input (hard-wired to t=0.5); "
          f"weights re-hosted with dt wired to the flow scaling for exact fps timing.")

    MODEL_CACHE[cache_key] = (rehosted, model_dtype)
    return rehosted, model_dtype


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


def inference_exact(model, img0, img1, dts, model_dtype, device, forward_fn=None):
    """Evaluate the interpolant directly at arbitrary timesteps.

    Unlike :func:`inference` (midpoint recursion, always t=0.5), each dt is
    fed to the model against the original pair, producing the frame at
    exactly that position on the source timeline. See _load_timeaware_model
    for why the shipped export cannot be used directly.

    ``forward_fn`` injection mirrors :func:`inference` for testing.
    """
    forward = forward_fn or (lambda x0, x1, dt: model(x0, x1, dt))
    outputs = []
    for dt in dts:
        dt_tensor = torch.tensor([[dt]], device=device, dtype=model_dtype)
        pred = forward(img0, img1, dt_tensor)
        outputs.append(pred.clamp(0, 1))
    return outputs


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
                "source_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001, "tooltip":
                    "Frame rate of the input frames (e.g. 24, 29.97, 23.976). Set BOTH this and 'target_fps' to enable "
                    "fps mode, which overrides 'multiplier': output frames are sampled on the exact target-fps timeline, "
                    "so fractional conversions like 24 -> 60 (2.5x) are retimed correctly instead of rounded. "
                    "NOTE: FILM was only trained at t=0.5; frames at other timesteps are positionally exact but are "
                    "model extrapolations. For quality-critical fractional conversions, RIFE (which supports arbitrary "
                    "timesteps natively) is recommended. Leave both at 0 to use 'multiplier'."}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001, "tooltip":
                    "Desired output frame rate. Fractional ratios (24 -> 60) and slowdowns (target < source) are "
                    "supported. Original frames that land on the target grid are copied verbatim; pairs excluded via "
                    "Interpolation States become hold-frames so the output timing stays exact. Feed the result to a "
                    "video saver with frame_rate = target_fps, or connect the node's frame_rate output. "
                    "Leave both fps inputs at 0 to use 'multiplier'."}),
                "video_info": ("VHS_VIDEOINFO", {"tooltip":
                    "Connect the 'video_info' output of a Video Helper Suite Load Video node to "
                    "auto-detect source_fps. Takes the loader's 'loaded_fps', the rate of the "
                    "frames actually loaded after force_rate and select_every_nth, so subsampled "
                    "loads stay timed correctly. With target_fps set, this enables fps mode; "
                    "otherwise 'multiplier' runs and the frame_rate output reports the detected "
                    "rate times the multiplier. Cannot be combined with a manual source_fps."}),
                "optional_interpolation_states": ("INTERPOLATION_STATES", {"tooltip":
                    "Optional. Connect a 'Make Interpolation State List' node to selectively skip or include specific frame "
                    "pairs for interpolation. If left unconnected, every consecutive pair of frames is interpolated."})
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "frame_rate")
    OUTPUT_TOOLTIPS = ("The interpolated frames.",
                       "Frame rate of the returned frames. target_fps in fps mode, otherwise "
                       "the source rate times the multiplier. 0.0 when the input rate is "
                       "unknown (no fps inputs and no video_info), or when skipped pairs or "
                       "uneven per-pair multipliers make the timing non-uniform.")
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-RIFE-FILM-Only/VFI"

    @torch.inference_mode()
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames=10,
        multiplier: typing.SupportsInt = 2,
        source_fps: float = 0.0,
        target_fps: float = 0.0,
        video_info: typing.Optional[dict] = None,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs,
    ):
        source_fps, fps_mode = resolve_fps_mode("FILM VFI", source_fps, target_fps, video_info)
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

        # fps mode: exact target-fps tick-grid retiming.
        if fps_mode:
            return self._vfi_fps(
                ckpt_name, model, model_dtype, output_dtype, frames_nchw,
                source_fps, target_fps, interpolation_states,
                clear_cache_after_n_frames, device,
            )

        if isinstance(multiplier, int):
            multipliers = [multiplier] * (num_input_frames - 1)
        else:
            multipliers = list(map(int, multiplier))
            multipliers += [2] * (num_input_frames - len(multipliers) - 1)
        any_skipped = interpolation_states is not None and any(
            interpolation_states.is_frame_skipped(i) for i in range(num_input_frames - 1))
        output_fps = multiplier_output_fps(source_fps, multipliers, any_skipped)

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

        return (postprocess_frames(output_frames[:out_len]), output_fps)

    def _vfi_fps(
        self,
        ckpt_name,
        model,
        model_dtype,
        output_dtype,
        frames_nchw,
        source_fps,
        target_fps,
        interpolation_states,
        clear_cache_after_n_frames,
        device,
    ):
        """Exact-timing fps conversion path (source_fps/target_fps inputs).

        Every output frame is evaluated directly against its bracketing input
        pair at the true target timestamp (dt = s_k - floor(s_k), exact via
        Fraction arithmetic in compute_fps_schedule); no midpoint recursion,
        so timing is exact for fractional ratios like 24 -> 60.

        If every requested dt is 0.5 the shipped model is used as-is;
        otherwise the weights are re-hosted by _load_timeaware_model (see
        that function for why and for the extrapolation caveat).
        """
        num_input_frames = len(frames_nchw)

        skip_pairs = set()
        if interpolation_states is not None:
            skip_pairs = {
                i for i in range(num_input_frames - 1)
                if interpolation_states.is_frame_skipped(i)
            }
        total_output, tasks, fills = compute_fps_schedule(
            num_input_frames, source_fps, target_fps, skip_pairs)
        print(f"FILM VFI: fps mode {source_fps} -> {target_fps} fps "
              f"(x{target_fps / source_fps:.6g}): {num_input_frames} -> {total_output} frames")

        needs_arbitrary_t = any(
            dt != 0.5 for _p, _out0, dts in tasks for dt in dts)
        if needs_arbitrary_t and not _model_responds_to_dt(model, device, model_dtype):
            model, model_dtype = _load_timeaware_model(ckpt_name, device)

        output_frames = torch.zeros(
            total_output, *frames_nchw.shape[1:], dtype=output_dtype, device="cpu"
        )

        # Original frames landing on ticks, hold-frames for skipped pairs,
        # and the appended final frame go to their exact slots up front.
        for out_idx, src_idx in fills:
            output_frames[out_idx] = frames_nchw[src_idx]

        total_tasks = sum(len(dts) for _, _out0, dts in tasks)
        pbar = VFIProgressBar(total_tasks, desc="FILM VFI")
        frames_processed = 0

        for pair_idx, out_start, dts in tasks:
            frame_0 = frames_nchw[pair_idx: pair_idx + 1].to(device, non_blocking=True).to(model_dtype)
            frame_1 = frames_nchw[pair_idx + 1: pair_idx + 2].to(device, non_blocking=True).to(model_dtype)

            mids = inference_exact(model, frame_0, frame_1, dts, model_dtype, device)

            for j, mid in enumerate(mids):
                output_frames[out_start + j] = mid.detach().to(dtype=output_dtype)
            del mids, frame_0, frame_1

            frames_processed += 1
            if frames_processed >= clear_cache_after_n_frames:
                soft_empty_cache()
                frames_processed = 0

            pbar.update(len(dts))

        soft_empty_cache()
        gc.collect()

        return (postprocess_frames(output_frames), target_fps)


NODE_CLASS_MAPPINGS = {
    "FILM VFI": FILM_VFI,
}
