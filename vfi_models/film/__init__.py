import torch
from comfy.model_management import get_torch_device, soft_empty_cache
import bisect
import numpy as np
import typing
from vfi_utils import InterpolationStateList, load_file_from_github_release, preprocess_frames, postprocess_frames
import pathlib
import gc
import comfy.utils
import sys
import warnings

warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths and odd dilation")

class VFIProgressBar:
    def __init__(self, total, desc="FILM VFI"):
        self.total = total
        self.n = 0
        self.desc = desc
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
                sys.stdout.write('\n')
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

@torch.inference_mode()
def inference(model, img_batch_1, img_batch_2, inter_frames, model_dtype, device):
    results = [img_batch_1, img_batch_2]
    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))
    
    splits = torch.linspace(0, 1, inter_frames + 2, device=device)
    
    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]
        
        dt_val = (splits[remains[step]] - splits[idxes[start_i]]) / (splits[idxes[end_i]] - splits[idxes[start_i]])
        dt = torch.tensor([[dt_val]], device=device, dtype=model_dtype)

        prediction = model(x0, x1, dt)
        
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1))
        del remains[step]
        
        if start_i > 0:
            old_tensor = results[start_i - 1]
            if old_tensor is not img_batch_1:
                old_tensor.cpu()
        
        if end_i < len(results) - 1:
            old_tensor = results[end_i + 1]
            if old_tensor is not img_batch_2:
                old_tensor.cpu()

    return results

class FILM_VFI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (["film_net_fp32.pt", "film_net_fp16.pt"], ),
                "frames": ("IMAGE", ),
                "clear_cache_after_n_frames": ("INT", {"default": 10, "min": 1, "max": 1000}),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 1000}),
            },
            "optional": {
                "optional_interpolation_states": ("INTERPOLATION_STATES", )
            }
        }

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "vfi"
    CATEGORY = "ComfyUI-Frame-Interpolation/VFI"

    @torch.inference_mode()
    def vfi(
        self,
        ckpt_name: typing.AnyStr,
        frames: torch.Tensor,
        clear_cache_after_n_frames = 10,
        multiplier: typing.SupportsInt = 2,
        optional_interpolation_states: InterpolationStateList = None,
        **kwargs
    ):
        interpolation_states = optional_interpolation_states
        device = get_torch_device()
        
        if ckpt_name not in MODEL_CACHE:
            soft_empty_cache()
            gc.collect()
            
            model_path = load_file_from_github_release(MODEL_TYPE, ckpt_name)
            model = torch.jit.load(model_path, map_location='cpu')
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
        output_frames = []
        
        pbar = VFIProgressBar(num_input_frames - 1, desc="FILM VFI")
        frames_processed = 0

        for frame_itr in range(num_input_frames - 1):
            if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
                output_frames.append(frames_nchw[frame_itr:frame_itr+1])
                pbar.update(1)
                continue
            
            frame_0 = frames_nchw[frame_itr:frame_itr+1].to(device, non_blocking=True).to(model_dtype)
            frame_1 = frames_nchw[frame_itr+1:frame_itr+2].to(device, non_blocking=True).to(model_dtype)
            
            results = inference(model, frame_0, frame_1, multipliers[frame_itr] - 1, model_dtype, device)
            
            for i, f in enumerate(results[:-1]):
                output_frames.append(f.detach().to(device="cpu", dtype=output_dtype, non_blocking=True))
                if i > 0:
                    del f
            
            del results
            del frame_0, frame_1
            
            frames_processed += 1
            if frames_processed >= clear_cache_after_n_frames:
                soft_empty_cache()
                gc.collect()
                frames_processed = 0

            pbar.update(1)

        output_frames.append(frames_nchw[-1:].to(dtype=output_dtype))
        
        out = torch.cat(output_frames, dim=0)
        del output_frames
        
        soft_empty_cache()
        gc.collect()
        
        return (postprocess_frames(out), )


NODE_CLASS_MAPPINGS = {
    "FILM VFI": FILM_VFI,
}
