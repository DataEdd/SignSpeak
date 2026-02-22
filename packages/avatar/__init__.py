"""Avatar package for 3D sign language rendering."""

from .pose_extractor import PoseExtractor, PoseSequence
from .renderer import AvatarRenderer
from .renderer_matplotlib import AvatarMatplotlibRenderer, MatplotlibRenderSettings

__all__ = [
    "PoseExtractor",
    "PoseSequence",
    "AvatarRenderer",
    "AvatarMatplotlibRenderer",
    "MatplotlibRenderSettings",
]

# Try to import SMPL-X renderer (requires smplx, torch)
try:
    from .renderer_smplx import (
        AvatarSMPLXRenderer,
        SMPLXRenderSettings,
        SMPLXSequence,
        MediaPipeToSMPLX,
    )
    __all__.extend([
        "AvatarSMPLXRenderer",
        "SMPLXRenderSettings",
        "SMPLXSequence",
        "MediaPipeToSMPLX",
    ])
except ImportError:
    pass

# Try to import pyrender 3D renderer (optional, has OpenGL requirements)
try:
    from .renderer_3d import Avatar3DRenderer, Render3DSettings
    __all__.extend(["Avatar3DRenderer", "Render3DSettings"])
except ImportError:
    pass
