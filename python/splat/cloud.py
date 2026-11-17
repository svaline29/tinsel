"""Cloud: a batch of 3D points with per-point attributes. Everything in splat is a Cloud."""
import numpy as np


def _per_point(v, m, dtype=float):
    return np.broadcast_to(np.asarray(v, dtype), (m,)).copy()


class Cloud:
    """
    P      (m,3) positions
    N      (m,3) surface normals. Zero normal = "particle": not shaded, never backface-culled.
    C      (m,3) color, 0..1
    emit   0..1  self-illumination (Christmas lights, stars). Ignores scene lighting.
    phase        blink phase offset (radians)
    blink        blink speed (radians/sec). 0 = steady.
    shine  0..1  specular highlight strength (car paint, glass)
    glyph        fixed character for this point ("*", "."), or 0 to pick from the brightness ramp
    t0           time (sec) at which the point appears
    """
    FIELDS = ("P", "N", "C", "emit", "phase", "blink", "shine", "glyph", "t0")

    def __init__(self, P, N=None, C=(1, 1, 1), emit=0.0, phase=0.0, blink=0.0,
                 shine=0.0, glyph=0, t0=0.0):
        P = np.asarray(P, float).reshape(-1, 3)
        m = len(P)
        if N is None:
            N = np.zeros((m, 3))
        else:
            N = np.asarray(N, float).reshape(-1, 3)
            ln = np.linalg.norm(N, axis=1, keepdims=True)
            N = np.divide(N, ln, out=np.zeros_like(N), where=ln > 1e-9)
        if isinstance(glyph, str):
            glyph = ord(glyph)
        elif not np.isscalar(glyph):
            glyph = [ord(g) if isinstance(g, str) else g for g in glyph]
        self.P, self.N = P, N
        self.C = np.broadcast_to(np.asarray(C, float), (m, 3)).copy()
        self.emit = _per_point(emit, m)
        self.phase = _per_point(phase, m)
        self.blink = _per_point(blink, m)
        self.shine = _per_point(shine, m)
        self.glyph = _per_point(glyph, m, int)
        self.t0 = _per_point(t0, m)

    def __len__(self):
        return len(self.P)

    def __add__(self, other):
        out = Cloud.__new__(Cloud)
        for f in self.FIELDS:
            setattr(out, f, np.concatenate([getattr(self, f), getattr(other, f)]))
        return out

    def __radd__(self, other):          # lets sum([...]) work
        return self if other == 0 else other.__add__(self)

    def subset(self, mask):
        out = Cloud.__new__(Cloud)
        for f in self.FIELDS:
            setattr(out, f, getattr(self, f)[mask])
        return out

    def moved(self, M):
        """Return a copy with 4x4 matrix M baked in."""
        out = self.subset(slice(None))
        A = M[:3, :3]
        out.P = self.P @ A.T + M[:3, 3]
        N = self.N @ np.linalg.inv(A)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        out.N = np.divide(N, ln, out=np.zeros_like(N), where=ln > 1e-9)
        return out


# ------------------------------------------------------------ 4x4 transforms
def translate(x=0.0, y=0.0, z=0.0):
    M = np.eye(4); M[:3, 3] = (x, y, z); return M

def scale(sx, sy=None, sz=None):
    sy = sx if sy is None else sy
    sz = sx if sz is None else sz
    return np.diag([sx, sy, sz, 1.0])

def rot_x(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4)
    M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c; return M

def rot_y(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c; return M

def rot_z(a):
    c, s = np.cos(a), np.sin(a); M = np.eye(4)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c; return M
