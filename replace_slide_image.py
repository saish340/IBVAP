"""Replace one picture on the SYSTEM VISUAL RESULTS slide, same geometry.

Usage:
    python replace_slide_image.py
    python replace_slide_image.py --list
    python replace_slide_image.py --pic-index 2 --new-image csrnet_density_comparison.jpg

- Opens IBVAP_SIH2026_Idea_Presentation_Paridrishti.pptx
- Finds slide whose title contains "SYSTEM VISUAL RESULTS"
- Records left/top/width/height of the target PICTURE shape
- Deletes only that shape element, inserts new image at identical geometry
- Saves as IBVAP_SIH2026_Idea_Presentation_Paridrishti_updated.pptx
- Touches nothing else.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT = Path(__file__).resolve().parent
PPTX_IN = ROOT / "IBVAP_SIH2026_Idea_Presentation_Paridrishti.pptx"
PPTX_OUT = ROOT / "IBVAP_SIH2026_Idea_Presentation_Paridrishti_updated.pptx"

# ---- CONFIG: edit these two lines for your case ----
TARGET_SLIDE_SUBSTRING = "SYSTEM VISUAL RESULTS"
# Which picture to replace, counting ONLY pictures on that slide (0-based,
# sorted top-to-bottom then left-to-right so row/col maps predictably).
# e.g. for a 3x3 grid: row1 = 0,1,2  row2 = 3,4,5  row3 = 6,7,8
# Row 2 "DEGRADED CCTV" (row=2,col=1) -> index 3
PICTURE_INDEX_TO_REPLACE = 3
NEW_IMAGE_PATH = ROOT / "csrnet_density_comparison.jpg"
# -----------------------------------------------------


def find_slide(prs: Presentation, substring: str):
    matches = []
    for i, slide in enumerate(prs.slides):
        title_text = ""
        for sh in slide.shapes:
            if sh.has_text_frame and sh.text.strip():
                if substring.lower() in sh.text.lower():
                    title_text = " ".join(sh.text.split())[:100]
                    matches.append((i, slide, title_text))
                    break
    return matches


def list_pictures(slide, slide_no: int):
    pics = [s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]
    # sort visually: top first, then left (row-major grid order)
    pics_sorted = sorted(pics, key=lambda s: (s.top, s.left))
    print(f"Slide {slide_no}: {len(pics_sorted)} picture(s) (row-major order):")
    for idx, p in enumerate(pics_sorted):
        print(f"  [{idx}] name={p.name!r} left={p.left} top={p.top} "
              f"width={p.width} height={p.height}")
    return pics_sorted


def replace_picture(slide, target_pic, new_image: Path):
    if not new_image.is_file():
        raise FileNotFoundError(f"New image not found: {new_image}")
    left, top, width, height = target_pic.left, target_pic.top, target_pic.width, target_pic.height
    print(f"Removing {target_pic.name!r} at left={left} top={top} w={width} h={height}")
    # Remove ONLY this shape's XML element
    sp = target_pic._element
    sp.getparent().remove(sp)
    # Insert new image at identical geometry
    slide.shapes.add_picture(str(new_image), left, top, width=width, height=height)
    print(f"Inserted {new_image.name} at identical geometry.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list pictures, don't modify")
    ap.add_argument("--pic-index", type=int, default=None)
    ap.add_argument("--new-image", type=str, default=None)
    ap.add_argument("--slide-title", type=str, default=None)
    args = ap.parse_args()

    slide_sub = args.slide_title or TARGET_SLIDE_SUBSTRING
    pic_index = args.pic_index if args.pic_index is not None else PICTURE_INDEX_TO_REPLACE
    new_image = Path(args.new_image) if args.new_image else NEW_IMAGE_PATH
    if not new_image.is_absolute():
        new_image = ROOT / new_image

    if not PPTX_IN.is_file():
        print(f"ERROR: {PPTX_IN.name} not found in {ROOT}", file=sys.stderr)
        return 1

    prs = Presentation(str(PPTX_IN))
    matches = find_slide(prs, slide_sub)
    if not matches:
        print(f'ERROR: no slide containing "{slide_sub}" found.', file=sys.stderr)
        print("Available title-ish texts:", file=sys.stderr)
        for i, s in enumerate(prs.slides):
            titles = [" ".join(sh.text.split())[:80] for sh in s.shapes
                      if sh.has_text_frame and sh.text.strip()]
            print(f"  slide {i+1}: {titles}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} slides match; using first: slide {matches[0][0]+1}")

    slide_idx, slide, title_text = matches[0]
    print(f'Target slide: #{slide_idx+1} ("{title_text}")')
    pics = list_pictures(slide, slide_idx + 1)

    if args.list:
        return 0
    if not pics:
        print("ERROR: no pictures on target slide.", file=sys.stderr)
        return 1
    if not (0 <= pic_index < len(pics)):
        print(f"ERROR: --pic-index {pic_index} out of range (0..{len(pics)-1})", file=sys.stderr)
        return 1

    replace_picture(slide, pics[pic_index], new_image)
    prs.save(str(PPTX_OUT))
    print(f"Saved: {PPTX_OUT.name} (original untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
