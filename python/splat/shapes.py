"""Shape builders. Each returns a Cloud (or a dict of Clouds for multi-part models)."""
import numpy as np
from .cloud import Cloud

_rng = np.random.default_rng(7)


def _jitter(color, m, amount=0.08, rng=_rng):
    return np.clip(np.asarray(color, float) + rng.normal(0, amount, (m, 1)) * np.asarray(color, float), 0, 1)


# ------------------------------------------------------------------ generic
def parametric(fn, n, flip=False, rng=_rng, **attrs):
    """Any surface you can write as fn(u, v) -> (x, y, z) with u, v in [0, 1). Normals are automatic."""
    u, v = rng.random(n * 6), rng.random(n * 6)              # oversample, then keep points evenly by area
    e = 1e-4
    P = np.stack(fn(u, v), 1)
    du = (np.stack(fn(u + e, v), 1) - P) / e
    dv = (np.stack(fn(u, v + e), 1) - P) / e
    N = np.cross(du, dv)
    w = np.linalg.norm(N, axis=1) + 1e-12
    i = rng.choice(len(P), n, p=w / w.sum())
    P, N = P[i], N[i]
    return Cloud(P, -N if flip else N, **attrs)


def sphere(n=1500, r=1.0, color=(1, 1, 1), rng=_rng, **attrs):
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    return Cloud(v * r, v, color, **attrs)


def torus(n=2500, R=1.0, r=0.35, color=(1, 1, 1), rng=_rng, **attrs):
    a, b = rng.random(n) * 2 * np.pi, rng.random(n) * 2 * np.pi
    P = np.c_[(R + r * np.cos(b)) * np.cos(a), r * np.sin(b), (R + r * np.cos(b)) * np.sin(a)]
    N = np.c_[np.cos(b) * np.cos(a), np.sin(b), np.cos(b) * np.sin(a)]
    return Cloud(P, N, color, **attrs)


def box(n=2000, size=(1, 1, 1), color=(1, 1, 1), rng=_rng, **attrs):
    sx, sy, sz = size
    areas = np.array([sy * sz, sy * sz, sx * sz, sx * sz, sx * sy, sx * sy])
    face = rng.choice(6, n, p=areas / areas.sum())
    a, b = rng.random(n) - 0.5, rng.random(n) - 0.5
    P, N = np.zeros((n, 3)), np.zeros((n, 3))
    for f in range(6):
        m = face == f; ax = f // 2; sgn = 1 - 2 * (f % 2)
        o = [i for i in range(3) if i != ax]
        P[m, ax] = sgn * 0.5; P[m, o[0]] = a[m]; P[m, o[1]] = b[m]; N[m, ax] = sgn
    return Cloud(P * size, N, color, **attrs)


def lathe(profile, n=3000, color=(1, 1, 1), rng=_rng, **attrs):
    """
    Spin a 2D outline around the vertical axis: trees, rockets, bottles, wheels, planets.
    profile = [(radius, y), ...] listed bottom -> top.
    """
    pr = np.asarray(profile, float)
    a, b = pr[:-1], pr[1:]
    seg = b - a
    ln = np.linalg.norm(seg, axis=1)
    w = ln * (a[:, 0] + b[:, 0]) / 2 + 1e-9                  # area of each band
    i = rng.choice(len(seg), n, p=w / w.sum())
    f = rng.random(n)
    r = a[i, 0] + seg[i, 0] * f
    y = a[i, 1] + seg[i, 1] * f
    th = rng.random(n) * 2 * np.pi
    nr, ny = seg[i, 1], -seg[i, 0]                           # outward normal of the outline
    P = np.c_[r * np.cos(th), y, r * np.sin(th)]
    N = np.c_[nr * np.cos(th), ny, nr * np.sin(th)]
    return Cloud(P, N, color, **attrs)


def polyline(points, n=100):
    """n evenly spaced points along a path. Use for light strings and garlands."""
    p = np.asarray(points, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, p[:, k]) for k in range(3)], 1)


# ------------------------------------------------------------------ nature
def pine(n=6000, height=2.0, radius=0.85, tiers=7, snow=0.0, rng=_rng):
    """Tiered Christmas tree, base at y=0. snow = 0..1 how much white on upward-facing branches."""
    prof = [(0.07, 0.0), (0.07, 0.12 * height)]
    y0 = 0.12 * height
    for k in range(tiers):
        f0, f1 = k / tiers, (k + 1) / tiers
        r_out = radius * (1 - f0) ** 0.9 + 0.03
        r_in = radius * (1 - f1) ** 0.9 * 0.45
        prof += [(r_out, y0 + f0 * (height - y0)), (r_in, y0 + f1 * (height - y0))]
    prof.append((0.0, height))
    c = lathe(prof, n, rng=rng)
    g = rng.random(n)
    c.C = np.c_[0.10 + 0.12 * g, 0.50 + 0.40 * g, 0.12 + 0.12 * rng.random(n)]
    trunk = c.P[:, 1] < y0 - 1e-6
    c.C[trunk] = (0.40, 0.26, 0.14)
    c.shine[:] = 0.15
    if snow > 0:
        white = (c.N[:, 1] > 0.35) & (rng.random(n) < snow) & ~trunk
        c.C[white] = (0.92, 0.95, 1.0)
    return c


# ------------------------------------------------------------------ lofted bodies
class Loft:
    """
    A body described by cross-sections along its length (the X axis): cars, planes, boats, fish.
    stations rows = (x, half_width, y_bottom, y_top, squareness). squareness 2 = ellipse, 4+ = boxy.
    """
    def __init__(self, stations, smooth=7):
        s = np.asarray(stations, float)
        self.x0, self.x1 = s[0, 0], s[-1, 0]
        xs = np.linspace(self.x0, self.x1, 240)
        k = np.ones(smooth) / smooth
        cols = []
        for j in range(1, 5):
            v = np.interp(xs, s[:, 0], s[:, j])
            v = np.convolve(np.pad(v, smooth // 2, mode="edge"), k, mode="valid")
            cols.append(v)
        self.xs, (self.w, self.yb, self.yt, self.e) = xs, cols

    def at(self, x, th):
        w, yb, yt, e = (np.interp(x, self.xs, a) for a in (self.w, self.yb, self.yt, self.e))
        c, s = np.cos(th), np.sin(th)
        z = w * np.sign(c) * np.abs(c) ** (2 / e)
        y = (yt + yb) / 2 + (yt - yb) / 2 * np.sign(s) * np.abs(s) ** (2 / e)
        return np.stack([x * np.ones_like(th), y, z], -1)

    def surface(self, n, color=(1, 1, 1), caps=True, rng=_rng, **attrs):
        x = self.x0 + rng.random(n * 6) * (self.x1 - self.x0)  # oversample, then keep points evenly by area
        th = rng.random(n * 6) * 2 * np.pi
        P = self.at(x, th)
        e = 1e-3
        N = np.cross(self.at(x + e, th) - P, self.at(x, th + e) - P)
        w = np.linalg.norm(N, axis=1) + 1e-12
        i = rng.choice(len(P), n, p=w / w.sum())
        x, th, P, N = x[i], th[i], P[i], N[i]
        mid = np.c_[x, (np.interp(x, self.xs, self.yt) + np.interp(x, self.xs, self.yb)) / 2, np.zeros(n)]
        N[(N * (P - mid)).sum(1) < 0] *= -1                    # always point outward
        out = Cloud(P, N, color, **attrs)
        if caps:
            for xe, sgn in ((self.x0, -1), (self.x1, 1)):
                m = max(n // 25, 8)
                th = rng.random(m) * 2 * np.pi
                edge = self.at(np.full(m, xe), th)
                ctr = edge.mean(0)
                Pc = ctr + (edge - ctr) * np.sqrt(rng.random((m, 1)))
                out = out + Cloud(Pc, np.tile((sgn, 0, 0), (m, 1)), color, **attrs)
        return out

    def rail(self, theta_deg, n=80, x0=None, x1=None, lift=0.03):
        """A path running along the body at a fixed angle around it. Good for strings of lights."""
        x = np.linspace(self.x0 if x0 is None else x0, self.x1 if x1 is None else x1, n)
        th = np.full(n, np.radians(theta_deg))
        P = self.at(x, th)
        mid = np.c_[x, (np.interp(x, self.xs, self.yt) + np.interp(x, self.xs, self.yb)) / 2, np.zeros(n)]
        out = P - mid; out /= np.linalg.norm(out, axis=1, keepdims=True)
        return P + out * lift, out


def wheel(n=1400, r=0.36, width=0.26, rng=_rng):
    """Wheel centred on the origin, axle along Z. Spokes make the spin visible."""
    m = n // 2
    a = rng.random(m) * 2 * np.pi
    tread = Cloud(np.c_[r * np.cos(a), r * np.sin(a), (rng.random(m) - 0.5) * width],
                  np.c_[np.cos(a), np.sin(a), np.zeros(m)], (0.30, 0.30, 0.33))
    parts = [tread]
    for sgn in (-1, 1):
        k = n // 4
        a = rng.random(k) * 2 * np.pi
        rr = r * np.sqrt(rng.random(k))
        spoke = (np.abs(((a * 5 / (2 * np.pi)) % 1) - 0.5) < 0.16) | (rr < 0.28 * r)
        tyre = rr > 0.68 * r
        col = np.where(tyre[:, None], (0.30, 0.30, 0.33),
                       np.where(spoke[:, None], (0.80, 0.82, 0.86), (0.20, 0.20, 0.23)))
        side = Cloud(np.c_[rr * np.cos(a), rr * np.sin(a), np.full(k, sgn * width / 2)],
                     np.tile((0, 0, sgn), (k, 1)), col)
        side.shine[:] = np.where(spoke & ~tyre, 0.7, 0.0)
        parts.append(side)
    return sum(parts)


def sports_car(n=16000, color=(0.86, 0.07, 0.10), rng=_rng):
    """
    Low fastback coupe, nose toward +X, tyres resting on y=0, about 4.4 long.
    Returns dict: body (Cloud), wheel (Cloud), wheel_at (4 positions), wheel_r,
                  body_loft / cabin_loft (for .rail() light strings), headlights, taillights (positions).
    """
    body = Loft([(-2.20, 0.62, 0.40, 0.66, 3.0), (-2.05, 0.86, 0.28, 0.80, 3.5),
                 (-1.50, 0.93, 0.21, 0.85, 3.8), (-0.70, 0.95, 0.19, 0.80, 4.0),
                 (0.10, 0.95, 0.19, 0.76, 4.0), (0.90, 0.94, 0.19, 0.71, 3.6),
                 (1.55, 0.90, 0.21, 0.62, 3.2), (2.00, 0.82, 0.25, 0.51, 2.8),
                 (2.20, 0.60, 0.31, 0.44, 2.5)])
    cabin = Loft([(-1.80, 0.58, 0.70, 0.82, 3.0), (-1.20, 0.72, 0.70, 0.99, 3.0),
                  (-0.50, 0.77, 0.70, 1.15, 3.0), (-0.05, 0.77, 0.70, 1.18, 3.0),
                  (0.40, 0.75, 0.68, 1.04, 3.0), (0.85, 0.70, 0.66, 0.76, 3.0),
                  (1.02, 0.60, 0.64, 0.69, 3.0)], smooth=5)
    wr, wx, wz = 0.36, 1.36, 0.80
    b = body.surface(int(n * 0.68), rng=rng)
    b.C = _jitter(color, len(b), 0.06, rng); b.shine[:] = 0.85
    low = b.P[:, 1] < 0.30                                    # dark sills / splitter
    b.C[low] = (0.22, 0.22, 0.25)
    arch = np.zeros(len(b), bool)                             # cut wheel arches
    for x in (-wx, wx):
        arch |= ((b.P[:, 0] - x) ** 2 + (b.P[:, 1] - wr) ** 2 < (wr + 0.07) ** 2) & (np.abs(b.P[:, 2]) > 0.45)
    b = b.subset(~arch)

    c = cabin.surface(int(n * 0.32), caps=False, rng=rng)
    c = c.subset(c.P[:, 1] > np.interp(c.P[:, 0], body.xs, body.yt) - 0.02)
    roof_y = np.interp(c.P[:, 0], cabin.xs, cabin.yt)
    is_roof = (c.P[:, 1] > roof_y - 0.045) & (c.P[:, 0] > -1.0) & (c.P[:, 0] < 0.35)
    pillar = (np.abs(c.P[:, 0] + 0.35) < 0.05)
    c.C = np.where((is_roof | pillar)[:, None], _jitter(color, len(c), 0.06, rng), (0.16, 0.22, 0.34))
    c.shine[:] = np.where(is_roof | pillar, 0.85, 1.0)

    return dict(body=b + c, wheel=wheel(rng=rng, r=wr), wheel_r=wr,
                wheel_at=[(sx * wx, wr, sz * wz) for sx in (1, -1) for sz in (1, -1)],
                body_loft=body, cabin_loft=cabin,
                headlights=[(2.17, 0.47, 0.55), (2.17, 0.47, -0.55)],
                taillights=[(-2.21, 0.60, 0.48), (-2.21, 0.60, -0.48)])


# ------------------------------------------------------------------ real 3D models
def load_mesh(path, n=8000, size=2.4, up="auto", yaw=0.0, color=(0.8, 0.8, 0.85), rng=_rng, **attrs):
    """
    Turn any .obj or .stl model (binary or text) into a Cloud. Free models: printables.com, sketchfab, NASA 3D.
    The model is centred, sat on y=0, and scaled so its longest side = size.
    up   = "y", "z", or "auto" (.stl files are usually Z-up, .obj usually Y-up)
    yaw  = extra turn about the vertical axis, in degrees (use 90 if the model faces the wrong way)
    """
    is_stl = path.lower().endswith(".stl")
    if is_stl:
        raw = open(path, "rb").read()
        cnt = int(np.frombuffer(raw[80:84], "<u4")[0]) if len(raw) >= 84 else -1
        if len(raw) == 84 + cnt * 50:                         # binary STL
            rec = np.frombuffer(raw[84:], np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
            tri = rec["v"].astype(float)
        else:                                                 # text STL
            v = [[float(x) for x in l.split()[1:4]] for l in raw.decode("ascii", "ignore").splitlines()
                 if l.strip().startswith("vertex")]
            tri = np.asarray(v, float).reshape(-1, 3, 3)
    else:
        V, F = [], []
        for line in open(path, errors="ignore"):
            p = line.split()
            if not p: continue
            if p[0] == "v": V.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                ids = [int(q.split("/")[0]) for q in p[1:]]
                ids = [i - 1 if i > 0 else len(V) + i for i in ids]
                F += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]
        tri = np.asarray(V, float)[np.asarray(F)]
    if up == "z" or (up == "auto" and is_stl):                # Z-up -> Y-up
        tri = tri[..., [0, 2, 1]] * (1, 1, -1)
    if yaw:
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        tri = tri @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(fn, axis=1) / 2
    i = rng.choice(len(tri), n, p=area / area.sum())          # more points on bigger triangles
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1; a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    P = tri[i, 0] + a[:, None] * (tri[i, 1] - tri[i, 0]) + b[:, None] * (tri[i, 2] - tri[i, 0])
    lo, hi = P.min(0), P.max(0)
    P = (P - (lo + hi) / 2) * (size / (hi - lo).max())
    P[:, 1] -= P[:, 1].min()
    return Cloud(P, fn[i], color, **attrs)


def inside_out(cloud):
    """Fraction of normals pointing inward. Above ~0.3 the model has messy faces: add it with two_sided=True."""
    return float(((cloud.N * (cloud.P - cloud.P.mean(0))).sum(1) < 0).mean())
