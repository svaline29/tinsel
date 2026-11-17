"""splat: 3D point-cloud scenes rendered as coloured ASCII. See README.md."""
from .cloud import Cloud, translate, scale, rot_x, rot_y, rot_z
from .render import Scene, Camera, Spot
from .out import play, export, run
from . import shapes, fx
