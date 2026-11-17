#!/usr/bin/env python3
"""Spinning Christmas tree over a reflective floor, with garland, twinkling stars and snow.
   python3 scenes/tree.py                      live in the terminal
   python3 scenes/tree.py --export tree.mp4    1080x1920 video
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from splat import Scene, Camera, shapes, fx, rot_y, run

SPIN = 0.55                                                     # radians / second
scene = Scene(Camera(eye=(0, 1.55, 6.2), target=(0, 0.95, 0), fov=27),
              light=(-0.6, 0.7, 0.55), mirror_y=0.0, glow=0.7)

tree = shapes.pine(n=26000, height=2.3, radius=0.95, tiers=8)

# garland: a spiral that hugs the tree
s = np.linspace(0.06, 0.97, 260)
a = s * 2 * np.pi * 5
r = 0.95 * (1 - s) ** 0.9 + 0.06
spiral = np.c_[r * np.cos(a), 0.28 + s * (2.3 - 0.28), r * np.sin(a)]
out = np.c_[np.cos(a), np.full_like(a, 0.3), np.sin(a)]
garland = fx.light_string(spiral, out, colors=fx.WARM, chase=0.25, blink=4.0, t_on=(0.5, 5.0))

bulbs = fx.lights_on(tree.subset(tree.P[:, 1] > 0.3), n=220)
bulbs.t0 = 1.0 + bulbs.P[:, 1] * 1.8                           # switch on from the bottom up
star = fx.glow_ball((0, 2.36, 0), r=0.09, n=120, color=(1, 0.95, 0.6))

spin = lambda t: rot_y(SPIN * t)
for part in (tree, garland, bulbs, star):
    scene.add(part, xform=spin, reflect=True)

floor, _ = fx.ground(x=(-7, 7), z=(-6, 4.5), row_gap=0.32, dot_gap=0.16,
                     color=(0.30, 0.36, 0.50), glyphs="..::-=", jitter=0.15)
scene.add(floor, cull=False)
flakes, fall = fx.snow(n=500, size=(9, 5, 8), center=(0, 2.5, 0), fall=0.45)
scene.add(flakes, deform=fall, cull=False)
scene.add_backdrop(fx.Stars(n=260, horizon=0.58))

n_lights = len(bulbs) + len(garland) // 4
scene.hud = lambda t: [f"splat . tree.py",
                       f"rotation {np.degrees(SPIN * t) % 360:06.2f}    lights {int((bulbs.t0 <= t).sum() + (garland.t0 <= t).sum() // 4):4d}/{n_lights}",
                       f"points {len(tree) + n_lights + len(star) + len(floor) + len(flakes):,}"]

if __name__ == "__main__":
    run(scene)
