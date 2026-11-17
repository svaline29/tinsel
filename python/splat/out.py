"""Output: play live in the terminal, or export .mp4 / .gif / .png."""
import os, shutil, subprocess, sys, time
import numpy as np

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
         "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
         "C:/Windows/Fonts/consolab.ttf", "C:/Windows/Fonts/consola.ttf"]


# ------------------------------------------------------------------ terminal
def to_ansi(chars, rgb):
    q = (rgb >> 3) << 3                                       # quantise so colour codes repeat less
    out = ["\x1b[H"]
    for row in range(chars.shape[0]):
        last, parts = None, []
        line = chars[row].tobytes().decode("ascii")
        for x, ch in enumerate(line):
            if ch != " ":
                key = (q[row, x, 0], q[row, x, 1], q[row, x, 2])
                if key != last:
                    parts.append(f"\x1b[38;2;{key[0]};{key[1]};{key[2]}m"); last = key
            parts.append(ch)
        out.append("".join(parts) + "\x1b[0m\x1b[K\n")
    return "".join(out)


def play(scene, fps=30, seconds=None):
    """Live preview. Resize the terminal freely. Ctrl-C to quit."""
    sys.stdout.write("\x1b[2J\x1b[?25l")
    t0 = time.time()
    try:
        while seconds is None or time.time() - t0 < seconds:
            tick = time.time()
            W, H = shutil.get_terminal_size()
            sys.stdout.write(to_ansi(*scene.render(tick - t0, W, H - 1))[:-1])
            sys.stdout.flush()
            time.sleep(max(0, 1 / fps - (time.time() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")


# ------------------------------------------------------------------ images / video
def _atlas(cw, ch, font):
    from PIL import Image, ImageDraw, ImageFont
    path = font or next((f for f in FONTS if os.path.exists(f)), None)
    f = ImageFont.truetype(path, int(ch * 0.82)) if path else ImageFont.load_default()
    A = np.zeros((128, ch, cw), np.float32)
    for code in range(33, 127):
        im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(im).text((cw / 2, ch / 2), chr(code), 255, font=f, anchor="mm")
        A[code] = np.asarray(im, np.float32) / 255
    return A


def rasterize(chars, rgb, atlas, bg=(7, 9, 20)):
    H, W = chars.shape
    _, ch, cw = atlas.shape
    a = atlas[chars][..., None]                               # H, W, ch, cw, 1
    img = a * rgb[:, :, None, None, :] + (1 - a) * np.asarray(bg, np.float32)
    return img.transpose(0, 2, 1, 3, 4).reshape(H * ch, W * cw, 3).astype(np.uint8)


def export(scene, path, size=(180, 160), seconds=8.0, fps=30, cell=(6, 12), t_start=0.0,
           bg=(7, 9, 20), font=None):
    """
    size  = (columns, rows) of characters.   cell = pixel size of one character.
    Video pixels = size * cell:  (180, 160) x (6, 12) = 1080x1920 vertical (Reels/TikTok/Shorts)
                                 (320, 90)  x (6, 12) = 1920x1080 landscape
    More, smaller characters = more detail. Fewer, bigger characters = chunkier retro look.
    path ending .mp4 (needs ffmpeg), .gif, or .png (single frame at t_start).
    """
    from PIL import Image
    W, H = size; cw, ch = cell
    atlas = _atlas(cw, ch, font)
    frame = lambda t: rasterize(*scene.render(t, W, H, cell_aspect=ch / cw), atlas, bg)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        Image.fromarray(frame(t_start)).save(path); return path
    n = int(seconds * fps)
    if ext == ".gif":
        ims = [Image.fromarray(frame(t_start + i / fps)).quantize(64) for i in range(n)]
        ims[0].save(path, save_all=True, append_images=ims[1:], duration=1000 / fps, loop=0); return path
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found. Install it (brew install ffmpeg / apt install ffmpeg) or export .gif")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W * cw}x{H * ch}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(frame(t_start + i / fps).tobytes())
        if i % fps == 0:
            print(f"\r  rendering {i / fps:.0f}/{seconds:.0f}s", end="", flush=True)
    p.stdin.close(); p.wait(); print(f"\r  wrote {path}            ")
    return path


def run(scene, default_size=(180, 160), default_seconds=8.0):
    """Shared command line for scene scripts: no args = play live; --export out.mp4 = render a file."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", metavar="FILE", help="write .mp4 / .gif / .png instead of playing")
    ap.add_argument("--size", default=f"{default_size[0]}x{default_size[1]}", help="characters, e.g. 180x160 (vertical) or 320x90 (landscape)")
    ap.add_argument("--cell", default="6x12", help="pixels per character")
    ap.add_argument("--seconds", type=float, default=default_seconds)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--at", type=float, default=0.0, help="start time")
    a = ap.parse_args()
    if a.export:
        export(scene, a.export, tuple(map(int, a.size.split("x"))), a.seconds, a.fps,
               tuple(map(int, a.cell.split("x"))), a.at)
    else:
        play(scene, a.fps)
