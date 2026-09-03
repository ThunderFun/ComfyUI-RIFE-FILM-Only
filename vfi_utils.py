import yaml
import os
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
import torch
import typing
import traceback
import einops
import gc
import torchvision.transforms.functional as transform
from comfy.model_management import soft_empty_cache, get_torch_device
import comfy.utils
import sys
import time
from fractions import Fraction

import numpy as np

class VFIProgressBar:
    """A progress bar that displays both in ComfyUI UI and terminal"""
    def __init__(self, total, desc="Comfy-VFI"):
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



try:
    import folder_paths
    COMFY_FOLDER_PATHS_AVAILABLE = True
except ImportError:
    COMFY_FOLDER_PATHS_AVAILABLE = False

BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]

# Mirrors for checkpoints missing (404) from all GitHub release endpoints,
# tried before the base URLs. A plain string keeps the URL's file name; a dict
# {"url": ..., "file_name": ...} renames the download (Practical-RIFE ships
# v4.25 as "flownet.pkl"; the bytes are already a state_dict).
CKPT_URL_OVERRIDES = {
    "rife46.pth": "https://huggingface.co/windecay/SimpleSDXL2/resolve/main/SimpleModels/controlnet/rife/rife46.pth",
    "rife425.pth": {
        "url": "https://huggingface.co/Upsampler/rife-4-25/resolve/main/flownet.pkl",
        "file_name": "rife425.pth",
    },
    "sudo_rife4_269.662_testV1_scale1.pth": "https://huggingface.co/licyk/sd-upscaler-models/resolve/main/ESRGAN/sudo_rife4_269.662_testV1_scale1.pth",
}

config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml not found. Download it from https://github.com/ThunderFun/ComfyUI-RIFE-FILM-Only")
DEVICE = get_torch_device()

class InterpolationStateList():

    def __init__(self, frame_indices: typing.List[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list
        
    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_frame_in_list or not self.is_skip_list and not is_frame_in_list
    

class MakeInterpolationStateList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frame_indices": ("STRING", {"multiline": True, "default": "1,2,3"}),
                "is_skip_list": ("BOOLEAN", {"default": True},),
            },
        }
    
    RETURN_TYPES = ("INTERPOLATION_STATES",)
    FUNCTION = "create_options"
    CATEGORY = "ComfyUI-RIFE-FILM-Only/VFI"

    def create_options(self, frame_indices: str, is_skip_list: bool):
        raw_split = frame_indices.split(',')
        try:
            frame_indices_list = [int(item) for item in raw_split]
        except ValueError:
            frame_indices_list = [int(item.strip()) for item in raw_split]
        
        interpolation_state_list = InterpolationStateList(
            frame_indices=frame_indices_list,
            is_skip_list=is_skip_list,
        )
        return (interpolation_state_list,)
        
        
def get_ckpt_container_path(model_type):
    if COMFY_FOLDER_PATHS_AVAILABLE:
        try:
            # Use ComfyUI's standard models/checkpoints directory
            checkpoints_paths = folder_paths.get_folder_paths("checkpoints")
            if checkpoints_paths:
                # Create path: models/checkpoints/vfi_models/{model_type}
                comfy_vfi_path = os.path.join(checkpoints_paths[0], "vfi_models", model_type)
                os.makedirs(comfy_vfi_path, exist_ok=True)
                return os.path.abspath(comfy_vfi_path)
        except Exception as e:
            print(f"ComfyUI-RIFE-FILM-Only: Failed to use ComfyUI folder paths ({e}), falling back to local directory")
    

    # Fallback to original behavior
    return os.path.abspath(os.path.join(os.path.dirname(__file__), config["ckpts_path"], model_type))

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Download a file from a URL if not already cached locally.

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    if file_name is None:
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def load_file_from_github_release(model_type, ckpt_name):
    error_strs = []
    urls = []  # (url, file_name) pairs
    if ckpt_name in CKPT_URL_OVERRIDES:
        override = CKPT_URL_OVERRIDES[ckpt_name]
        if isinstance(override, dict):
            urls.append((override["url"], override.get("file_name")))
        else:
            urls.append((override, None))
    urls.extend(
        (base_model_download_url + ckpt_name, None)
        for base_model_download_url in BASE_MODEL_DOWNLOAD_URLS
    )

    for i, (url, file_name) in enumerate(urls):
        try:
            return load_file_from_url(url, get_ckpt_container_path(model_type), file_name=file_name)
        except Exception:
            traceback_str = traceback.format_exc()
            if i < len(urls) - 1:
                print("Failed! Trying another endpoint.")
            error_strs.append(f"Error when downloading from: {url}\n\n{traceback_str}")

    error_str = '\n\n'.join(error_strs)
    raise Exception(f"Tried all download URLs for {ckpt_name} but none succeeded. Error log:\n\n{error_str}")
                

def load_file_from_direct_url(model_type, url):
    return load_file_from_url(url, get_ckpt_container_path(model_type))

def preprocess_frames(frames):
    return einops.rearrange(frames[..., :3], "n h w c -> n c h w")

def postprocess_frames(frames):
    return einops.rearrange(frames, "n c h w -> n h w c")[..., :3].cpu()

def assert_batch_size(frames, batch_size=2, vfi_name=None):
    subject_verb = "Most VFI models require" if vfi_name is None else f"VFI model {vfi_name} requires"
    assert len(frames) >= batch_size, f"{subject_verb} at least {batch_size} frames to work with, only found {frames.shape[0]}. Please check the frame input using PreviewImage."


_FPS_LIMIT_DENOMINATOR = 1_000_000


def _parse_fps(value, name):
    """Parse a user fps into an exact rational.

    Fraction(str(x)) captures the decimal the user typed (or the full float
    repr) exactly; limit_denominator then snaps NTSC-style rates to their
    true low-denominator forms (23.976023976023978 -> 24001/1001, 29.97 ->
    2997/100) so ratios like 23.976 -> 47.952 come out as *exact* integers
    instead of accumulating float drift that would turn original-frame
    copies into spurious model calls.
    """
    try:
        frac = Fraction(str(float(value))).limit_denominator(_FPS_LIMIT_DENOMINATOR)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if frac <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return frac


def resolve_fps_mode(vfi_name, source_fps, target_fps, video_info=None):
    """Merge an optional VHS video_info dict into source_fps and pick the mode.

    VHS loaders report the rate of the frames they actually delivered as
    'loaded_fps', after force_rate and select_every_nth. The IMAGE batch
    contains frames at that rate, so it becomes source_fps. Setting
    source_fps manually while video_info is connected is ambiguous and raises.

    fps mode (output sampled on the exact target-fps grid) requires both
    rates. source_fps alone runs multiplier mode and scales the frame_rate
    output; target_fps alone is unusable and raises.

    Returns (source_fps, fps_mode).
    """
    if video_info is not None:
        if source_fps > 0:
            raise ValueError(
                f"{vfi_name}: source_fps is set manually and video_info is connected; "
                "reset source_fps to 0 to use the fps detected from video_info.")
        source_fps = float(video_info.get("loaded_fps", 0))
        if source_fps <= 0:
            raise ValueError(f"{vfi_name}: connected video_info has no usable 'loaded_fps'.")
    if target_fps > 0 and source_fps <= 0:
        raise ValueError(
            f"{vfi_name}: target_fps is set but the input frame rate is unknown; "
            "set source_fps or connect a Video Helper Suite video_info.")
    return source_fps, source_fps > 0 and target_fps > 0


def multiplier_output_fps(source_fps, multipliers, any_skipped=False):
    """Frame rate of a multiplier-mode output batch.

    The source rate times the uniform per-pair multiplier. 0.0 when the input
    rate is unknown, per-pair multipliers are uneven, or skipped pairs make
    the timing non-uniform.
    """
    if source_fps <= 0 or not multipliers or len(set(multipliers)) != 1 or any_skipped:
        return 0.0
    return source_fps * multipliers[0]


def compute_fps_schedule(
    num_frames: int,
    source_fps: float,
    target_fps: float,
    skip_pairs: typing.Optional[typing.Set[int]] = None,
):
    """Build the exact output schedule for a source_fps -> target_fps conversion.

    Output frames are sampled on the target-fps tick grid: output tick ``k``
    corresponds to source-timeline position ``s_k = k * source_fps / target_fps``
    (in units of source frame intervals). When ``s_k`` lands exactly on a
    source frame that frame is copied verbatim (no model call); otherwise the
    frame is synthesized between the two bracketing source frames at
    ``dt = s_k - floor(s_k)``.

    All position math is done with :class:`fractions.Fraction`, so "is this
    tick an original frame?" is an exact integer test: no float drift, and
    equal in/out rates always produce a pure passthrough.

    Args:
        num_frames: number of input frames (>= 2).
        source_fps: fps of the input frames (> 0).
        target_fps: desired output fps (> 0). May be smaller than source_fps
            (retimed decimation); the same formula handles it.
        skip_pairs: pair indices that must not be model-interpolated (from an
            InterpolationStateList). Ticks inside a skipped pair become
            hold-frames (a copy of the pair's left source frame) so the total
            output count, and therefore the timing, is unchanged.

    Returns:
        (total_output, tasks, fills) where
        total_output: number of output frames.
        tasks: list of ``(pair_idx, first_out_idx, dts)`` with ``dts`` the
            ascending list of timesteps to synthesize for that pair, in pair
            order. Ticks of a pair are consecutive output indices, so the
            j-th dt of a task writes to ``first_out_idx + j``.
        fills: list of ``(out_idx, src_frame_idx)`` direct copies (original
            frames that land on ticks, plus hold-frames for skipped pairs),
            sorted by out_idx.

    The final input frame is always present in the output: either it lands
    exactly on the last tick, or one extra output slot is appended holding it.
    """
    if num_frames < 2:
        raise ValueError(f"fps conversion needs at least 2 input frames, got {num_frames}")
    fs = _parse_fps(source_fps, "source_fps")
    ft = _parse_fps(target_fps, "target_fps")

    if skip_pairs is None:
        skip_pairs = set()
    num_pairs = num_frames - 1
    bad = [p for p in skip_pairs if not 0 <= p < num_pairs]
    if bad:
        raise ValueError(f"skip_pairs contains out-of-range pair indices {bad} for {num_frames} frames")

    # Ratio of target ticks per source frame interval, exact.
    ratio = ft / fs
    # Source position of the last reachable tick: k <= (N-1) * ratio.
    span = (num_frames - 1) * ratio
    last_tick = span.numerator // span.denominator  # exact floor

    pair_dts: typing.Dict[int, typing.List[float]] = {}
    pair_first_out: typing.Dict[int, int] = {}
    fills: typing.List[typing.Tuple[int, int]] = []

    for k in range(last_tick + 1):
        s = k * fs / ft  # exact source position of tick k
        i = s.numerator // s.denominator  # floor -> left source frame
        dt = s - i
        if dt == 0:
            fills.append((k, i))
        elif i in skip_pairs:
            # Hold-frame: repeat the pair's left frame, timing preserved.
            fills.append((k, i))
        else:
            if i not in pair_dts:
                pair_dts[i] = []
                pair_first_out[i] = k
            pair_dts[i].append(dt.numerator / dt.denominator)

    tasks = [(i, pair_first_out[i], pair_dts[i]) for i in sorted(pair_dts.keys())]

    total_output = last_tick + 1
    if span.denominator != 1:
        # The last input frame falls between ticks; append it so the output
        # always ends on the final source frame.
        fills.append((last_tick + 1, num_frames - 1))
        total_output += 1

    return total_output, tasks, fills

def _generic_frame_loop(
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=True,
        dtype=torch.float16,
        final_logging=True):
    
    # Non-timestep recursive bisection (used by models without arbitrary-timestep support)
    def non_timestep_inference(frame0, frame1, n):        
        middle = return_middle_frame_function(frame0, frame1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame0, middle, n=n//2)
        second_half = non_timestep_inference(middle, frame1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    output_frames = torch.zeros(multiplier*frames.shape[0], *frames.shape[1:], dtype=dtype, device="cpu")
    out_len = 0

    number_of_frames_processed_since_last_cleared_cuda_cache = 0

    # Initialize progress bar (both UI and terminal)
    total_frames = len(frames) - 1
    pbar = VFIProgressBar(total_frames, desc="Comfy-VFI")
    
    for frame_itr in range(len(frames) - 1):
        frame0 = frames[frame_itr:frame_itr+1]
        output_frames[out_len] = frame0
        out_len += 1
        # Ensure that input frames are in fp32 - the same dtype as model
        frame0 = frame0.to(dtype=torch.float32)
        frame1 = frames[frame_itr+1:frame_itr+2].to(dtype=torch.float32)
        
        if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
            continue
    
        middle_frame_batches = []

        if use_timestep:
            for middle_i in range(1, multiplier):
                timestep = middle_i/multiplier
                
                middle_frame = return_middle_frame_function(
                    frame0.to(DEVICE), 
                    frame1.to(DEVICE),
                    timestep,
                    *return_middle_frame_function_args
                ).detach().cpu()
                middle_frame_batches.append(middle_frame.to(dtype=dtype))
        else:
            middle_frames = non_timestep_inference(frame0.to(DEVICE), frame1.to(DEVICE), multiplier - 1)
            middle_frame_batches.extend(torch.cat(middle_frames, dim=0).detach().cpu().to(dtype=dtype))
        
        for middle_frame in middle_frame_batches:
            output_frames[out_len] = middle_frame
            out_len += 1

        number_of_frames_processed_since_last_cleared_cuda_cache += 1
        if number_of_frames_processed_since_last_cleared_cuda_cache >= clear_cache_after_n_frames:
            soft_empty_cache()
            number_of_frames_processed_since_last_cleared_cuda_cache = 0
        
        gc.collect()

        pbar.update(1)
    
    if final_logging:
        print(f"Comfy-VFI done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
    output_frames[out_len] = frames[-1:]
    out_len += 1
    soft_empty_cache()
    return output_frames[:out_len]

def generic_frame_loop(
        model_name,
        frames,
        clear_cache_after_n_frames,
        multiplier: typing.Union[typing.SupportsInt, typing.List],
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states: InterpolationStateList = None,
        use_timestep=True,
        dtype=torch.float32):

    assert_batch_size(frames, vfi_name=model_name.replace('_', ' ').replace('VFI', ''))
    if type(multiplier) == int:
        return _generic_frame_loop(
            frames, 
            clear_cache_after_n_frames, 
            multiplier, 
            return_middle_frame_function, 
            *return_middle_frame_function_args, 
            interpolation_states=interpolation_states,
            use_timestep=use_timestep,
            dtype=dtype
        )
    if type(multiplier) == list:
        multipliers = list(map(int, multiplier))
        multipliers += [2] * (len(frames) - len(multipliers) - 1)
        frame_batches = []
        for frame_itr in range(len(frames) - 1):
            multiplier = multipliers[frame_itr]
            if multiplier == 0: continue
            frame_batch = _generic_frame_loop(
                frames[frame_itr:frame_itr+2], 
                clear_cache_after_n_frames, 
                multiplier, 
                return_middle_frame_function, 
                *return_middle_frame_function_args, 
                interpolation_states=interpolation_states,
                use_timestep=use_timestep,
                dtype=dtype,
                final_logging=False
            )
            if frame_itr != len(frames) - 2: # Not append last frame unless this batch is the last one
                frame_batch = frame_batch[:-1]
            frame_batches.append(frame_batch)
        output_frames = torch.cat(frame_batches)
        print(f"Comfy-VFI done! {len(output_frames)} frames generated at resolution: {output_frames[0].shape}")
        return output_frames
    raise NotImplementedError(f"multipiler of {type(multiplier)}")

class FloatToInt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0, 'min': 0, 'step': 0.01})
            }
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = "ComfyUI-RIFE-FILM-Only"

    def convert(self, float):
        if hasattr(float, "__iter__"):
            return (list(map(int, float)),)
        return (int(float),)
