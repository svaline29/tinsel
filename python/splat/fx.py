"""Effects: stars, snow, ground, lights, spray. Each returns something you pass to scene.add(...)."""
import numpy as np
from .cloud import Cloud
from .render import SPACE

_rng = np.random.default_rng(11)

XMAS = [(1.0, 0.15, 0.15), (0.2, 1.0, 0.3), (0.25, 0.5, 1.0), (1.0, 0.85, 0.25), (1.0, 0.35, 0.9)]
WARM = [(1.0, 0.82, 0.45)]


class Stars:
    """Twinkling night sky drawn behind everything. scene.add_backdrop(Stars())"""
    def __init__(self, n=220, horizon=0.62, speed=2.2, rng=_rng):
        self.u, self.v = rng.random(n), rng.random(n) ** 1.3 * horizon
        self.base = 0.25 + 0.75 * rng.random(n) ** 2.5
        self.ph, self.sp = rng.random(n) * 6.28, speed * (0.4 + rng.random(n))
        self.tint = 0.8 + 0.2 * rng.random((n, 3))

    def draw(self, t, chars, rgb):
        H, W = chars.shape
        x = (self.u * W).astype(int); y = (self.v * H).astype(int)
        b = self.base * (0.55 + 0.45 * np.sin(self.sp * t + self.ph))
        free = chars[y, x] == SPACE
        g = np.where(b > 0.62, ord("*"), np.where(b > 0.34, ord("+"), ord("."))).astype(np.uint8)
        chars[y[free], x[free]] = g[free]
        rgb[y[free], x[free]] = (self.tint * (0.3 + 0.7 * b[:, None]))[free]


def snow(n=1400, size=(16, 7, 12), center=(0, 3.5, 0), fall=0.8, drift=(0, 0, 0), sway=0.15, rng=_rng):
    """
    Falling snow that loops forever inside a box. Returns (cloud, deform) -> scene.add(cloud, deform=deform, cull=False)
    drift = extra constant velocity. For a car driving at speed v toward +X use drift=(-v, 0, 0).
    """
    size, center, drift = (np.asarray(a, float) for a in (size, center, drift))
    P0 = rng.random((n, 3)) * size
    big = rng.random(n)
    glyph = np.where(big > 0.9, ord("*"), np.where(big > 0.55, ord("+"), ord(".")))
    c = Cloud(P0, None, 0.55 + 0.45 * rng.random((n, 1)) * np.ones(3), glyph=glyph)
    ph, speed = rng.random(n) * 6.28, fall * (0.6 + 0.8 * rng.random(n))
    vel = drift[None, :] + np.c_[np.zeros(n), -speed, np.zeros(n)]

    def deform(t, P):
        Q = P + vel * t
        Q[:, 0] += sway * np.sin(1.3 * t + ph)
        Q[:, 2] += sway * np.cos(1.1 * t + ph)
        return Q % size + center - size / 2
    return c, deform


def ground(y=0.0, x=(-12, 12), z=(-9, 7), row_gap=0.45, dot_gap=0.22, speed=0.0,
           color=(0.36, 0.42, 0.55), glyphs=".:.,", jitter=0.5, rng=_rng):
    """
    A floor made of rows of dots (reads as snow, water or a grid). Returns (cloud, deform).
    speed > 0 scrolls it toward -X, which makes a stationary object look like it is driving toward +X.
    """
    zs = np.arange(z[0], z[1], row_gap)
    xs = np.arange(x[0], x[1], dot_gap)
    X, Z = np.meshgrid(xs, zs)
    X = X + rng.normal(0, dot_gap * jitter, X.shape); Z = Z + rng.normal(0, row_gap * jitter * 0.3, Z.shape)
    m = X.size
    g = np.array([ord(ch) for ch in glyphs])[rng.integers(0, len(glyphs), m)]
    c = Cloud(np.c_[X.ravel(), np.full(m, y), Z.ravel()], None,
              np.asarray(color) * (0.6 + 0.4 * rng.random((m, 1))), glyph=g)
    span = x[1] - x[0]

    def deform(t, P):
        Q = P.copy()
        Q[:, 0] = (P[:, 0] - x[0] - speed * t) % span + x[0]
        return Q
    return c, (deform if speed else None)


def scroll_x(speed, x0, x1, offset=0.0, z=0.0):
    """
    xform for scenery passing by: the object slides toward -X and wraps around between x0 and x1.
    Build the object at the origin; offset = where along the road it starts, z = how far back it sits.
    """
    from .cloud import translate
    return lambda t: translate(((offset - speed * t - x0) % (x1 - x0)) + x0, 0, z)


def lights_on(cloud, n=120, colors=XMAS, lift=0.03, blink=4.0, rng=_rng):
    """Scatter n fairy lights over the surface of any Cloud."""
    i = rng.choice(len(cloud), n, replace=False)
    col = np.asarray(colors)[rng.integers(0, len(colors), n)]
    return Cloud(cloud.P[i] + cloud.N[i] * lift, cloud.N[i], col, emit=1.0,
                 phase=rng.random(n) * 6.28, blink=blink * (0.5 + rng.random(n)))


def light_string(path, normals=None, colors=XMAS, blink=5.0, chase=0.55, t_on=None, size=0.035, dots=4, rng=_rng):
    """
    Bulbs along a path (from shapes.polyline or Loft.rail). chase = phase step between bulbs: gives the running-lights look.
    t_on = (start, end): bulbs switch on one after another between those times.
    size / dots: each bulb is a little cluster of points so it covers more than one character.
    """
    path = np.asarray(path, float); n = len(path)
    k = np.repeat(np.arange(n), dots)
    P = path[k] + rng.normal(0, size, (n * dots, 3))
    col = np.asarray(colors)[k % len(colors)]
    t0 = 0.0 if t_on is None else np.linspace(t_on[0], t_on[1], n)[k]
    return Cloud(P, None if normals is None else np.asarray(normals)[k], col, emit=1.0,
                 phase=-k * chase, blink=blink, t0=t0)


def wrap_lights(cloud, n=220, turns=9, axis="auto", colors=XMAS, lift=0.03, band=0.07, **string_opts):
    """
    Wind one string of lights around ANY shape, like wrapping a tree. Works on meshes, lofts, anything.
    cloud  = a dense Cloud of the shape (30k+ points gives clean results)
    axis   = "x", "y", "z" or "auto" (the shape's longest side)
    turns  = how many times the string goes round
    Returns a light_string Cloud. Extra options (blink, chase, t_on, size, dots) pass through to light_string.
    """
    P = cloud.P
    lo, hi = P.min(0), P.max(0)
    ax = int(np.argmax(hi - lo)) if axis == "auto" else "xyz".index(axis)
    u, v = [k for k in range(3) if k != ax]
    ctr = (lo + hi) / 2
    s = (P[:, ax] - lo[ax]) / (hi[ax] - lo[ax])               # 0..1 along the axis
    ru, rv = P[:, u] - ctr[u], P[:, v] - ctr[v]
    rad = np.hypot(ru, rv)
    f = turns * s - np.arctan2(rv, ru) / (2 * np.pi)
    on = np.abs(f - np.round(f)) < band                       # points lying near the spiral
    idx = np.flatnonzero(on)
    b = np.minimum((s[idx] * n).astype(int), n - 1)           # one bulb per step along the axis...
    order = np.lexsort((-rad[idx], b))                        # ...taking the outermost point in that step
    first = np.r_[True, np.diff(b[order]) > 0]
    pick = idx[order[first]]
    out = np.zeros((len(pick), 3))
    out[:, u], out[:, v] = ru[pick], rv[pick]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return light_string(P[pick] + out * lift, out, colors=colors, **string_opts)


def glow_ball(pos, r=0.06, n=40, color=(1, 1, 0.8), blink=0.0, rng=_rng):
    """A small bright blob: headlight, tail light, tree-top star."""
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return Cloud(np.asarray(pos, float) + v * r * rng.random((n, 1)) ** 0.5, v, color, emit=1.0, blink=blink,
                 phase=rng.random(n) * 6.28)


def spray(origin, velocity=(-2.2, 1.3, 0), spread=0.6, n=160, life=0.55, gravity=4.5,
          color=(0.85, 0.9, 1.0), rng=_rng):
    """Looping particle fountain (snow kicked up by tyres, sparks, exhaust). Returns (cloud, deform)."""
    origin, velocity = np.asarray(origin, float), np.asarray(velocity, float)
    v = velocity + rng.normal(0, spread, (n, 3)) * (1, 0.6, 0.5)
    ph = rng.random(n) * life
    c = Cloud(np.tile(origin, (n, 1)), None, color, glyph=np.where(rng.random(n) > 0.6, ord("*"), ord(".")))

    def deform(t, P):
        a = ((t + ph) % life)[:, None]
        Q = origin + v * a
        Q[:, 1] = np.maximum(Q[:, 1] - 0.5 * gravity * a[:, 0] ** 2, 0.02)
        return Q
    return c, deform
