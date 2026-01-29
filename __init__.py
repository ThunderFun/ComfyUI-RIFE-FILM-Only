import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vfi_models.rife import RIFE_VFI
from vfi_models.film import FILM_VFI
from vfi_utils import MakeInterpolationStateList, FloatToInt
    
NODE_CLASS_MAPPINGS = {
    "RIFE VFI": RIFE_VFI,
    "FILM VFI": FILM_VFI,
    "Make Interpolation State List": MakeInterpolationStateList,
    "VFI FloatToInt": FloatToInt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RIFE VFI": "RIFE VFI (recommend rife47 and rife49)"
}
