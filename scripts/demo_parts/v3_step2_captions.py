"""Step 2: burn professional captions into trimmed Video 3. Keeps UI pixels intact."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
TRIM = ROOT / "demo" / "_v3_trim.mp4"
FINAL = ROOT / "demo" / "IBVAP_Live_Surveillance_Demo.mp4"

# (start, end, line1, line2) in trimmed timeline seconds
CAPS = [
    (0.0, 9.0, "IBVAP  |  Live AI Surveillance Demo", "Real-time detection + tracking + alerts  |  Team Paridrishti - SIH 2026"),
    (9.0, 20.0, "Human Detection  |  Live Person Tracking", "YOLO + ByteTrack  |  persistent ID with confidence score"),
    (20.0, 33.0, "Degraded Condition Detection  |  Low Visibility", "Pipeline keeps tracking through poor lighting / blur"),
    (33.0, 46.0, "Suspicious Activity  |  Movement Analysis", "System flags abnormal movement, raises threat score"),
    (46.0, 56.0, "Virtual Fence Setup  |  Restricted Zone", "Operator draws the zone  |  ENTER locks it"),
    (56.0, 76.0, "Intrusion Alert  |  Restricted-Zone Violation", "Zone breach turns the box red  |  threat score jumps to 60 (HIGH)"),
    (76.0, 92.0, "Multi-Object Tracking  |  Person + Object", "Second ID appears when the bottle enters the frame"),
    (92.0, 106.0, "Real-Time Alert Logging  |  Evidence + Hash", "Every intrusion, suspicious move and low-visibility frame is logged"),
    (106.0, 118.0, "Continuous Monitoring  |  Threat Escalation", "Risk level rises LOW to HIGH to CRITICAL as events persist"),
    (118.0, 129.4, "IBVAP  |  Always-On Border Surveillance", "Detection + tracking + virtual fence + alerts, verified live"),
]

W, H = 1918, 1078
CAP_H = 108

def esc(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

def drawbox(start, end):
    y = H - 40 - CAP_H
    return (f"drawbox=x=0:y={y}:w={W}:h={CAP_H}:color=black@0.72:t=fill"
            f":enable='between(t,{start:.1f},{end:.1f})'")

def title_filter(text, start, end, y_off, size, bold=True):
    wt = 1 if bold else 0
    return (f"drawtext=text='{esc(text)}':fontfile='C\\:/Windows/Fonts/segoeui.ttf'"
            f":fontsize={size}:fontcolor=white:borderw=1:bordercolor=black@0.8"
            f":x=(w-text_w)/2:y={y_off}:enable='between(t,{start:.1f},{end:.1f})'"
            f"{':shadowx=1:shadowy=1' if wt else ''}")

def sub_filter(text, start, end, y_off):
    return (f"drawtext=text='{esc(text)}':fontfile='C\\:/Windows/Fonts/segoeui.ttf'"
            f":fontsize=24:fontcolor=#FFD98A:borderw=1:bordercolor=black@0.8"
            f":x=(w-text_w)/2:y={y_off}:enable='between(t,{start:.1f},{end:.1f})'")

def fade(start, end):
    parts = []
    if start > 0:
        parts.append(f"fade=t=in:st={start:.1f}:d=0.4")
    parts.append(f"fade=t=out:st={end-0.4:.1f}:d=0.4:enable='between(t,{start:.1f},{end:.1f})'")
    return ",".join(parts)

def main():
    assert TRIM.exists(), f"missing {TRIM} (run v3_step1_trim.py first)"
    filters = []
    for s, e, l1, l2 in CAPS:
        y1, y2 = H - 40 - CAP_H + 16, H - 40 - CAP_H + 58
        filters.append(drawbox(s, e))
        filters.append(title_filter(l1, s, e, y1, 30))
        filters.append(sub_filter(l2, s, e, y2))
    vf = ",".join(filters)
    af = "aresample=44100"
    for s, e in [(s, e) for s, e, _, _ in CAPS]:
        af += f",afade=t=in:st={s:.1f}:d=0.4,afade=t=out:st={e-0.4:.1f}:d=0.4"
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(TRIM),
           "-vf", vf, "-af", af,
           "-c:v", "libx264", "-preset", "medium", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(FINAL)]
    print("[v3] burning captions ...", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[v3] DONE {FINAL} ({FINAL.stat().st_size/1e6:.1f} MB)", flush=True)

if __name__ == "__main__":
    main()
