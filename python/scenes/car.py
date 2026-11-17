#!/usr/bin/env python3
"""Sports car wrapped in Christmas lights, driving through falling snow at night.
   python3 scenes/car.py                                  live in the terminal
   python3 scenes/car.py --export car.mp4                 1080x1920 vertical video
   python3 scenes/car.py --export car.mp4 --size 320x90   1920x1080 landscape

The car never moves: the ground, snow and trees slide backwards past it at SPEED.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from splat import Scene, Camera, Spot, shapes, fx, translate, rot_z, run

SPEED = 6.0                                                     # world units / second
rng = np.random.default_rng(3)

# camera drifts slowly around the front three-quarter view
eye = lambda t: np.array([4.6 + 1.2 * np.sin(0.25 * t), 1.55 + 0.25 * np.sin(0.4 * t), 5.6 + 0.8 * np.cos(0.25 * t)])
scene = Scene(Camera(eye, target=(0.1, 0.6, 0), fov=40, fov_wide=25), light=(0.45, 0.6, 0.85), ambient=0.3, glow=0.8)

# ---- car
car = shapes.sports_car(n=18000, color=(0.86, 0.07, 0.10))
body, cabin = car["body_loft"], car["cabin_loft"]
strings = []
for loft, angles, x0, x1, n in ((body, (38, 142), -2.05, 2.05, 46),     # along both shoulders
                                (body, (-28, 208), -0.95, 0.95, 20),    # along the sills between the wheels
                                (cabin, (60, 120), -1.55, 0.80, 28)):   # along the roof edges
    for ang in angles:
        path, nrm = loft.rail(ang, n, x0, x1)
        strings.append(fx.light_string(path, nrm, chase=0.6, blink=6.0))
for x, y, z in car["wheel_at"]:                                 # a ring of lights round each wheel arch
    a = np.linspace(0.15, np.pi - 0.15, 12)
    ring = np.c_[x + 0.47 * np.cos(a), y + 0.47 * np.sin(a), np.full(12, z + 0.14 * np.sign(z))]
    strings.append(fx.light_string(ring, np.tile((0, 0, np.sign(z)), (12, 1)), chase=0.6, blink=6.0))
lamps = sum(fx.glow_ball(p, r=0.09, n=60, color=(1.0, 0.97, 0.85)) for p in car["headlights"]) \
      + sum(fx.glow_ball(p, r=0.08, n=50, color=(1.0, 0.08, 0.05)) for p in car["taillights"])

bounce = lambda t: translate(0, 0.012 * np.sin(11 * t) + 0.008 * np.sin(17.3 * t), 0)
scene.add(car["body"] + sum(strings) + lamps, xform=bounce)
for x, y, z in car["wheel_at"]:
    scene.add(car["wheel"], xform=lambda t, x=x, y=y, z=z: translate(x, y, z) @ rot_z(-SPEED * t / car["wheel_r"]))
for x, y, z in car["headlights"]:
    scene.add_spot(Spot((x, y, z), (1, -0.13, 0), angle=24, power=3.0, reach=9))

# ---- world sliding past
snowfield, slide = fx.ground(x=(-13, 13), z=(-14, 7), speed=SPEED, color=(0.38, 0.44, 0.58))
scene.add(snowfield, deform=slide, cull=False)
flakes, fall = fx.snow(n=1800, size=(22, 7, 14), center=(1, 3.4, 0), fall=0.9, drift=(-SPEED, 0, 0))
scene.add(flakes, deform=fall, cull=False)
for i in range(9):                                              # roadside pines
    h = 2.2 + 1.6 * rng.random()
    tree = shapes.pine(n=1300, height=h, radius=0.38 * h, tiers=6, snow=0.7, rng=rng)
    scene.add(tree, xform=fx.scroll_x(SPEED, -15, 15, offset=i * 30 / 9 + rng.random() * 2, z=-4.5 - 8.0 * rng.random()))
for x, y, z in car["wheel_at"][2:]:                             # snow kicked up by the rear tyres
    puff, fly = fx.spray((x - 0.3, 0.05, z), velocity=(-2.6, 1.2, 0), n=110)
    scene.add(puff, deform=fly, cull=False)

scene.add_backdrop(fx.Stars(n=240, horizon=0.42))
scene.hud = lambda t: ["splat . car.py", f"speed {SPEED * 12:5.1f} km/h    distance {SPEED * t:7.1f} m"]

if __name__ == "__main__":
    run(scene)
