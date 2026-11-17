#!/usr/bin/env python3
"""Any 3D model, wrapped in Christmas lights, spinning in the snow.
   python3 scenes/wrap.py car.obj                       live in the terminal
   python3 scenes/wrap.py car.obj --export car.mp4      1080x1920 video
   options: --color 0.8,0.1,0.1  --turns 12  --bulbs 260  --axis x|y|z  --up y|z  --yaw 90  --palette xmas|warm
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse
import numpy as np
from splat import Scene, Camera, shapes, fx, rot_y, run

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("model")
ap.add_argument("--color", default="0.80,0.10,0.12")
ap.add_argument("--turns", type=float, default=10)
ap.add_argument("--bulbs", type=int, default=240)
ap.add_argument("--axis", default="auto")
ap.add_argument("--up", default="auto")
ap.add_argument("--yaw", type=float, default=0)
ap.add_argument("--palette", default="xmas", choices=["xmas", "warm"])
ap.add_argument("--points", type=int, default=30000)
a, rest = ap.parse_known_args()
sys.argv = [sys.argv[0]] + rest

rng = np.random.default_rng(5)
dense = shapes.load_mesh(a.model, n=150000, up=a.up, yaw=a.yaw, rng=rng)       # dense copy for placing lights
body = dense.subset(rng.choice(len(dense), a.points, replace=False))
paint = np.array([float(x) for x in a.color.split(",")])
body.C = np.clip(paint * (1 + rng.normal(0, 0.06, (len(body), 1))), 0, 1); body.shine[:] = 0.7
messy = shapes.inside_out(dense) > 0.3
top = dense.P[:, 1].max()

lights = fx.wrap_lights(dense, n=a.bulbs, turns=a.turns, axis=a.axis,
                        colors=fx.XMAS if a.palette == "xmas" else fx.WARM, t_on=(0.5, 4.5), chase=0.45, blink=5.0)

SPIN = 0.55
scene = Scene(Camera(eye=(0, top / 2 + 1.0, 6.6), target=(0, top / 2 - 0.25, 0), fov=30),
              light=(-0.6, 0.7, 0.55), ambient=0.28, mirror_y=0.0, glow=0.8)
spin = lambda t: rot_y(SPIN * t)
scene.add(body, xform=spin, reflect=True, two_sided=messy)
scene.add(lights, xform=spin, reflect=True)
floor, _ = fx.ground(x=(-7, 7), z=(-6, 4.5), row_gap=0.32, dot_gap=0.16, color=(0.30, 0.36, 0.50), glyphs="..::-=", jitter=0.15)
scene.add(floor, cull=False)
flakes, fall = fx.snow(n=500, size=(9, 5, 8), center=(0, 2.5, 0), fall=0.45)
scene.add(flakes, deform=fall, cull=False)
scene.add_backdrop(fx.Stars(n=260, horizon=0.58))
name = os.path.basename(a.model)
scene.hud = lambda t: [f"splat . {name}",
                       f"rotation {np.degrees(SPIN * t) % 360:06.2f}    lights {int((lights.t0 <= t).sum() // 4):4d}/{len(lights) // 4}",
                       f"points {len(body) + len(lights) + len(floor) + len(flakes):,}"]

if __name__ == "__main__":
    run(scene)
