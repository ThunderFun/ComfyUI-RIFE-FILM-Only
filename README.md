# ComfyUI Frame Interpolation - FILM and RIFE Only

A custom node set for Video Frame Interpolation in ComfyUI, containing only FILM and RIFE models.

## Fork Additions
This fork includes the following improvements from community pull requests:

* **ComfyUI Folder Paths** by [vitosans](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/pull/97) - Adds support for ComfyUI's standard `models/checkpoints` directory structure
* **RIFE Speed Up** by [JohnAlcatraz](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/pull/102) - Speeds up RIFE by 45x (36.4 seconds -> 0.8 seconds)

**FILM VFI Improvements:**
* Performance optimizations for faster processing
* Support for FP16 model: [`film_net_fp16.pt`](https://huggingface.co/jkawamoto/frame-interpolation-pytorch/blob/main/film_net_fp16.pt)

**Other Changes:**
* Removed all other interpolation methods - this fork contains only FILM and RIFE models for a streamlined experience
* Bug fixes

## Nodes
* RIFE VFI (4.0 - 4.9)
* FILM VFI
* Make Interpolation State List
* VFI FloatToInt

## Install
### ComfyUI Manager
Following this guide to install this extension:
https://github.com/ltdrdata/ComfyUI-Manager#how-to-use

### Command-line
#### Windows
Run install.bat

#### Linux
Open your shell app and start venv if it is used for ComfyUI. Run:
```
python install.py
```

## Model Placement
Models can be placed in either of the following locations:

1. **ComfyUI's standard models directory** (recommended):
   - `models/checkpoints/vfi_models/rife/` for RIFE models
   - `models/checkpoints/vfi_models/film/` for FILM models

2. **Legacy location** (fallback):
   - `ComfyUI-Frame-Interpolation/ckpts/rife/` for RIFE models
   - `ComfyUI-Frame-Interpolation/ckpts/film/` for FILM models

The extension will automatically check the ComfyUI models directory first, and fall back to the legacy location if the models are not found there.

## Usage
All VFI nodes can be accessed in **category** `ComfyUI-Frame-Interpolation/VFI` if the installation is successful and require a `IMAGE` containing frames (at least 2).

`clear_cache_after_n_frames` is used to avoid out-of-memory. Decreasing it makes the chance lower but also increases processing time.

It is recommended to use LoadImages (LoadImagesFromDirectory) from [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet/) and [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) along side with this extension.

## Example
### Simple workflow
Download these two images [anime0.png](./demo_frames/anime0.png) and [anime1.png](./demo_frames/anime1.png) and put them into a folder like `E:\test` in this image.
![](./example.png)

## Credit

### RIFE
```bibtex
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

### FILM
[Frame interpolation in PyTorch](https://github.com/dajes/frame-interpolation-pytorch)

```bibtex
@inproceedings{reda2022film,
 title = {FILM: Frame Interpolation for Large Motion},
 author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
 booktitle = {European Conference on Computer Vision (ECCV)},
 year = {2022}
}
```

```bibtex
@misc{film-tf,
  title = {Tensorflow 2 Implementation of "FILM: Frame Interpolation for Large Motion"},
  author = {Fitsum Reda and Janne Kontkanen and Eric Tabellion and Deqing Sun and Caroline Pantofaru and Brian Curless},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/frame-interpolation}}
}
```
