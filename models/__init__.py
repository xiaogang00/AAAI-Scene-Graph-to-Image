from .model_setup import *

# Generator
from .model_psgim import PSGIM

# Discriminator
from .discriminators import PatchDiscriminator
from .discriminators import PatchDiscriminator2
from .discriminators import AcDiscriminator
from .discriminators import AcCropDiscriminator
from .discriminators import PatchDiscriminator_seg
from .discriminators import PatchDiscriminator_feature
from .discriminators import PatchDiscriminator_seg_feature
from .discriminators import PatchDiscriminator_scene_feature
from .discriminators import PatchDiscriminator_scene_feature2

from .discriminators import MultiscaleDiscriminator
