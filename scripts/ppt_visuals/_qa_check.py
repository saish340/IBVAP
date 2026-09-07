"""Temporary QA: verify the synthetic degradations hit single conditions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_ppt_visuals as g  # noqa: E402

fr = g.load_frames("854621-hd_1920_1080_25fps.mp4", stride=6)
base = g.pick_sharp_bright(fr)
print("base:", g.ConditionMonitor().process_frame(base).condition)
mon = g.ConditionMonitor()
mets = [mon.process_frame(f).raw_metrics for f in fr]
blurs = sorted(m["blur_score"] for m in mets)
print("blur min/med/max:", round(blurs[0], 1), round(blurs[len(blurs) // 2], 1),
      round(blurs[-1], 1))
print("frames blur>=80:", sum(b >= 80 for b in blurs), "/", len(blurs))
print("base metrics:", {k: round(v, 1)
                         for k, v in mon.process_frame(base).raw_metrics.items()})
for name, fn in [("low_light", g.syn_low_light), ("fog", g.syn_fog),
                 ("blur", g.syn_blur), ("noise", g.syn_noise)]:
    out = fn(base)
    rep = g.classify_stable(out)
    m = {k: round(v, 1) for k, v in rep.raw_metrics.items()}
    print(f"{name:9s} -> {rep.condition:22s} sev={rep.severity:.2f} {m}")
