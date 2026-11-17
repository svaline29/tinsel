"""Scene + renderer: turns Clouds into a grid of (character, color) for time t."""
import numpy as np

SPACE = 32


def _val(v, t):
    return np.asarray(v(t) if callable(v) else v, float)


class Camera:
    """eye / target can be fixed points or functions of time (for orbits, fly-bys, shake)."""
    def __init__(self, eye=(0, 1, 5), target=(0, 0, 0), fov=40.0, fov_wide=None):
        # fov spans the shorter side of the frame. fov_wide (optional) is used instead for landscape frames.
        self.eye, self.target, self.fov, self.fov_wide = eye, target, fov, fov_wide

    @staticmethod
    def orbit(radius=5.0, height=1.0, speed=0.5, target=(0, 0, 0), start=0.0, fov=40.0):
        tg = np.asarray(target, float)
        eye = lambda t: tg + np.array([radius * np.sin(start + speed * t), height,
                                       radius * np.cos(start + speed * t)])
        return Camera(eye, target, fov)


class Spot:
    """A cone of light (headlights, a street lamp). Brightens everything inside it."""
    def __init__(self, pos, direction, angle=22.0, color=(1.0, 0.95, 0.75), power=2.5, reach=8.0):
        self.pos = np.asarray(pos, float)
        d = np.asarray(direction, float); self.dir = d / np.linalg.norm(d)
        self.cos = np.cos(np.radians(angle))
        self.color = np.asarray(color, float)
        self.power, self.reach = power, reach


class Obj:
    def __init__(self, cloud, xform=None, deform=None, cull=True, reflect=False, two_sided=False):
        self.cloud, self.xform, self.deform = cloud, xform, deform
        self.cull, self.reflect, self.two_sided = cull and not two_sided, reflect, two_sided


class Scene:
    """
    camera      Camera
    light       direction the main light comes FROM (world space)
    ambient     floor brightness for surfaces facing away from the light
    ramp        dark -> bright characters
    glow        how much lights bleed color onto their neighbours (0 = off)
    mirror_y    height of a reflective floor (None = no reflection)
    hud         function t -> list of text lines drawn bottom-left
    """
    def __init__(self, camera, light=(-0.5, 0.8, 0.6), ambient=0.18, ramp=" .:-=+*#%@",
                 glow=0.6, mirror_y=None, mirror_dim=0.33, hud=None):
        self.camera = camera
        L = np.asarray(light, float); self.light = L / np.linalg.norm(L)
        self.ambient, self.glow = ambient, glow
        self.ramp = np.frombuffer(ramp.encode(), np.uint8)
        self.mirror_y, self.mirror_dim = mirror_y, mirror_dim
        self.hud = hud
        self.objs, self.backdrops, self.spots = [], [], []

    def add(self, cloud, xform=None, deform=None, cull=True, reflect=False, two_sided=False):
        """
        xform   4x4 matrix, or function t -> 4x4 matrix (rigid motion: spin, drive, bounce)
        deform  function (t, P) -> P'  (per-point motion: falling snow, scrolling ground)
        cull    hide points facing away from the camera (right for solid surfaces)
        reflect mirror this object in the floor at mirror_y
        two_sided  for downloaded models with inside-out faces: light both sides, never cull
        """
        o = Obj(cloud, xform, deform, cull, reflect, two_sided)
        self.objs.append(o)
        return o

    def add_backdrop(self, layer):      # screen-space layer, e.g. fx.Stars
        self.backdrops.append(layer); return layer

    def add_spot(self, spot):
        self.spots.append(spot); return spot

    # ------------------------------------------------------------------ render
    def render(self, t, W, H, cell_aspect=2.0):
        """Returns (chars uint8 [H,W], rgb uint8 [H,W,3]). cell_aspect = cell height / width."""
        P, N, C, EM, PH, BL, SH, GL, CULL, DIM, TWO = ([] for _ in range(11))
        for o in self.objs:
            c = o.cloud
            p = o.deform(t, c.P) if o.deform else c.P
            n = c.N
            M = o.xform(t) if callable(o.xform) else o.xform
            if M is not None:
                A = M[:3, :3]
                p = p @ A.T + M[:3, 3]
                n = n @ A.T
                ln = np.linalg.norm(n, axis=1, keepdims=True)
                n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 1e-9)
            on = c.t0 <= t
            copies = [(p, n, 1.0)]
            if o.reflect and self.mirror_y is not None:
                pm = p * (1, -1, 1) + (0, 2 * self.mirror_y, 0)
                copies.append((pm, n * (1, -1, 1), self.mirror_dim))
            for pp, nn, dim in copies:
                P.append(pp[on]); N.append(nn[on]); C.append(c.C[on]); EM.append(c.emit[on])
                PH.append(c.phase[on]); BL.append(c.blink[on]); SH.append(c.shine[on])
                GL.append(c.glyph[on]); m = int(on.sum())
                CULL.append(np.full(m, o.cull)); DIM.append(np.full(m, dim)); TWO.append(np.full(m, o.two_sided))
        chars = np.full((H, W), SPACE, np.uint8)
        rgb = np.zeros((H, W, 3), float)
        if P:
            P, N, C, EM, PH, BL, SH, GL, CULL, DIM, TWO = (np.concatenate(a) for a in
                                                           (P, N, C, EM, PH, BL, SH, GL, CULL, DIM, TWO))
            self._draw_points(t, W, H, cell_aspect, chars, rgb, P, N, C, EM, PH, BL, SH, GL, CULL, DIM, TWO)
        for layer in self.backdrops:
            layer.draw(t, chars, rgb)
        if self.hud:
            for i, line in enumerate(reversed(self.hud(t))):
                row = H - 1 - i
                if row < 0: break
                b = np.frombuffer(line[:W].encode("ascii", "replace"), np.uint8)
                chars[row, :len(b)] = b
                rgb[row, :len(b)] = (0.42, 0.47, 0.56)
        return chars, (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def _draw_points(self, t, W, H, cell_aspect, chars, rgb, P, N, C, EM, PH, BL, SH, GL, CULL, DIM, TWO):
        eye, target = _val(self.camera.eye, t), _val(self.camera.target, t)
        fwd = target - eye; fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, (0, 1, 0)); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        # --- camera space + perspective projection
        d = P - eye
        zc = d @ fwd
        short = min(H, W / cell_aspect)
        fov = self.camera.fov_wide if (self.camera.fov_wide and W / cell_aspect > H) else self.camera.fov
        F = (short / 2) / np.tan(np.radians(fov) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = np.floor(W / 2 + (d @ right) / zc * F * cell_aspect).astype(np.int64)
            sy = np.floor(H / 2 - (d @ up) / zc * F).astype(np.int64)

        # --- shading
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        V = -d / np.maximum(dist, 1e-9)                       # point -> eye
        lit = (N * N).sum(1) > 0.5
        facing = (N * V).sum(1)
        back = TWO & (facing < 0)                             # two-sided: turn the normal toward the camera
        N = np.where(back[:, None], -N, N); facing = np.abs(facing) * back + facing * ~back
        ndl = np.clip(N @ self.light, 0, 1)
        Hv = self.light + V; Hv /= np.linalg.norm(Hv, axis=1, keepdims=True)
        spec = SH * np.clip((N * Hv).sum(1), 0, 1) ** 28
        lum = np.where(lit, self.ambient + (1 - self.ambient) * ndl, 1.0)
        col = C * (0.3 + 0.7 * lum[:, None]) + spec[:, None]      # keep colour vivid; let the character carry the shading
        bright = lum + spec

        for s in self.spots:                                  # cones of light
            ds = P - s.pos
            r = np.linalg.norm(ds, axis=1)
            ca = (ds @ s.dir) / np.maximum(r, 1e-6)
            k = np.clip((ca - s.cos) / (1 - s.cos), 0, 1) ** 1.5
            k = k * s.power / (1 + (r / s.reach * 3) ** 2)
            col += k[:, None] * s.color * (0.35 + 0.65 * C)
            bright += k

        em = EM > 0                                           # self-lit points
        pulse = np.where(BL > 0, 0.62 + 0.38 * np.sin(BL * t + PH), 1.0) * EM
        col = np.where(em[:, None], C * pulse[:, None], col)
        bright = np.where(em, 0.45 + 0.55 * pulse, bright)

        col *= DIM[:, None]; bright = bright * DIM

        # --- visibility
        ok = (zc > 0.05) & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        ok &= ~(CULL & lit & (facing < -0.08))                # backface cull
        if self.mirror_y is not None:                         # nothing real below the floor
            ok &= (DIM < 1) | (P[:, 1] >= self.mirror_y - 1e-6)
        idx = sy[ok] * W + sx[ok]
        order = np.argsort(zc[ok], kind="stable")             # nearest first
        cell, first = np.unique(idx[order], return_index=True)
        keep = np.flatnonzero(ok)[order[first]]               # z-buffer winners

        nr = len(self.ramp) - 1
        ramp_i = np.clip(np.rint(np.clip(bright[keep], 0, 1) * nr).astype(int), 1, nr)
        g = np.where(GL[keep] > 0, GL[keep], self.ramp[ramp_i]).astype(np.uint8)
        chars.reshape(-1)[cell] = g
        rgb.reshape(-1, 3)[cell] = col[keep]

        # --- glow: lights tint the cells around them
        if self.glow > 0 and em[keep].any():
            src = np.zeros((H, W, 3))
            src.reshape(-1, 3)[cell[em[keep]]] = col[keep][em[keep]]
            pad = np.pad(src, ((1, 1), (3, 3), (0, 0)))
            halo = np.zeros_like(src)
            for dy in (-1, 0, 1):
                for dx in range(-3, 4):
                    if dx == 0 and dy == 0: continue
                    w = 1.0 / (1.5 + dx * dx * 0.8 + dy * dy * 3.0)
                    halo += w * pad[1 + dy:1 + dy + H, 3 + dx:3 + dx + W]
            halo *= self.glow
            is_em = np.zeros(H * W, bool); is_em[cell[em[keep]]] = True
            is_em = is_em.reshape(H, W)
            filled = (chars != SPACE) & ~is_em
            rgb[filled] += halo[filled]
            faint = (chars == SPACE) & (halo.max(2) > 0.16)
            chars[faint] = ord(".")
            rgb[faint] = halo[faint] * 1.6
