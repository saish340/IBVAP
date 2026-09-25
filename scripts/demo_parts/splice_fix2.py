"""Splice fix: replace frames [ANPR_START, ANPR_BAD) with plate-window content."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FULL = ROOT / "demo" / "IBVAP_SIH_Demo.mp4"

# current layout (post 4.5s cut): ANPR scene runs 68..96. Bad end-card ~81.5..86.
ANPR_GOOD_END = 81.5   # keep 68..81.5 (plate visible throughout)
ANPR_BAD_END = 86.0    # drop 81.5..86.0 (end-card)

def main():
    import subprocess as _sp
    total = float(_sp.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(FULL)],
        capture_output=True, text=True, check=True).stdout.strip())
    print("total", total, flush=True)
    parts = [(0, ANPR_GOOD_END), (ANPR_BAD_END, total)]
    segs = []
    for i, (a, b) in enumerate(parts):
        seg = FULL.parent / f"_seg{i}.mp4"
        _sp.run(["ffmpeg", "-y", "-v", "error", "-ss", str(a), "-i", str(FULL),
                 "-t", str(b - a), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-crf", "20", "-preset", "fast", str(seg)], check=True)
        segs.append(seg)
    lst = FULL.parent / "_seglist.txt"
    lst.write_text("".join(f"file '{s.name}'\n" for s in segs))
    out = FULL.parent / "_final.mp4"
    _sp.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", str(lst), "-c", "copy", str(out)], check=True)
    out.replace(FULL)
    short = FULL.parent / "IBVAP_SIH_Demo_Short.mp4"
    _sp.run(["ffmpeg", "-y", "-v", "error", "-ss", "8", "-i", str(FULL),
             "-t", "80", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-preset", "fast", str(short)], check=True)
    for s in segs:
        s.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    info = _sp.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=duration,nb_frames",
         "-of", "default=noprint_wrappers=1", str(FULL)],
        capture_output=True, text=True, check=True).stdout
    print(info, flush=True)
    print("short MB:", short.stat().st_size / 1e6, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

