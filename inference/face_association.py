"""Face <-> person-track association and live face-overlay state.

The face pipeline (RetinaFace detection + ArcFace verification) is expensive
and therefore runs on a much slower cadence than person tracking (YOLO +
ByteTrack).  Left alone, the dashboard draws each verified face box at its
*stale* verified position until the next face pass, so the red box visibly
lags behind a moving person.

This module closes that gap WITHOUT a second heavy tracker:

* :func:`associate_faces_with_tracks` links every detected face to the
  ByteTrack person it belongs to (face-centre containment, closest match).
* :class:`FaceOverlayTracker` remembers, per person track, where the face
  sat inside the person box at verification time, and re-derives a live
  face box from the *current* person box on every tracking sweep - so the
  overlay follows movement and size changes between expensive face passes.

Face verification itself is untouched: names, confidences and the verified
bbox always come from the real ArcFace pass; association only adds
``track_id`` context and keeps the box visually attached to its person.
Unassociated faces keep ``track_id=None`` (never an invented ID).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: how long an un-refreshed face may keep being re-drawn from its person
#: track before it is expired (must exceed the face-verification cadence)
DEFAULT_TTL_SECONDS = 8.0


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _contains(person_bbox: Sequence[float], point: Tuple[float, float]) -> bool:
    px1, py1, px2, py2 = (float(v) for v in person_bbox[:4])
    cx, cy = point
    return px1 <= cx <= px2 and py1 <= cy <= py2


def _center_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def associate_faces_with_tracks(
    faces: List[Dict[str, Any]], tracks: Sequence[Dict[str, Any]]
) -> None:
    """Add ``track_id`` to every face dict, in place.

    A face belongs to the person whose ByteTrack bbox contains the face
    centre; when several person boxes contain it, the person whose box
    centre is closest wins.  Each track claims at most one face, so two
    faces are never pinned to the same person while a free track exists.
    A face with no containing person keeps ``track_id=None``.

    Only adds context - existing face keys (``bbox``, ``name``,
    ``confidence``, ...) are never modified.
    """
    if not faces:
        return
    candidates: List[Tuple[float, int, int]] = []  # (distance, face_idx, track_idx)
    for fi, face in enumerate(faces):
        bbox = face.get("bbox")
        if not bbox:
            continue
        center = _bbox_center(bbox)
        for ti, track in enumerate(tracks):
            track_id = track.get("track_id")
            if track_id is None:
                continue
            person_bbox = track.get("bbox")
            if person_bbox and _contains(person_bbox, center):
                candidates.append(
                    (_center_distance(center, _bbox_center(person_bbox)), fi, ti)
                )
    candidates.sort()
    claimed_faces: set = set()
    claimed_tracks: set = set()
    for _, fi, ti in candidates:
        if fi in claimed_faces or ti in claimed_tracks:
            continue
        faces[fi]["track_id"] = tracks[ti].get("track_id")
        claimed_faces.add(fi)
        claimed_tracks.add(ti)
    for face in faces:
        face.setdefault("track_id", None)


class FaceOverlayTracker:
    """Keeps the face overlay attached to its person between face passes.

    For every verified face associated with a person track, the tracker
    stores the face box *relative to that person's bbox* plus the verified
    identity.  :meth:`live_faces` re-derives the face box from the CURRENT
    person bbox on every tracking sweep, so the overlay follows movement
    (walk left/right) and size changes (walk toward the camera) without
    waiting for the next ArcFace pass.

    State expires after ``ttl_seconds`` without a fresh verification, and
    immediately when its person track disappears - a stale box is never
    kept alive indefinitely, and no box is ever invented for a person
    whose face was never verified.  Thread-safe: verification results may
    arrive on a worker thread while the display reads on another.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self._lock = threading.Lock()
        # track_id -> {"rel": (rx1, ry1, rx2, ry2), "face": {...}, "ts": float}
        self._anchored: Dict[Any, Dict[str, Any]] = {}
        # faces verified with no person carrier: last verified box only
        self._loose: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ write
    def update_verified(
        self,
        faces: List[Dict[str, Any]],
        tracks: Sequence[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> None:
        """Record a freshly verified face pass (association happens here).

        Mutates ``faces`` in place (adds ``track_id``) so the caller's
        stored verification result carries person context too.
        """
        associate_faces_with_tracks(faces, tracks)
        now = time.time() if timestamp is None else float(timestamp)
        track_boxes: Dict[Any, Sequence[float]] = {
            t.get("track_id"): t.get("bbox")
            for t in tracks
            if t.get("track_id") is not None and t.get("bbox")
        }
        with self._lock:
            # Drop anchors whose person track is gone (person left frame).
            self._anchored = {
                tid: entry for tid, entry in self._anchored.items()
                if tid in track_boxes
            }
            for face in faces:
                bbox = face.get("bbox")
                if not bbox:
                    continue
                record = {k: v for k, v in face.items()}
                record["ts"] = now
                track_id = face.get("track_id")
                person_bbox = track_boxes.get(track_id)
                if person_bbox is not None:
                    px1, py1, px2, py2 = (float(v) for v in person_bbox[:4])
                    pw = max(px2 - px1, 1e-6)
                    ph = max(py2 - py1, 1e-6)
                    fx1, fy1, fx2, fy2 = (float(v) for v in bbox[:4])
                    self._anchored[track_id] = {
                        "rel": (
                            (fx1 - px1) / pw, (fy1 - py1) / ph,
                            (fx2 - px1) / pw, (fy2 - py1) / ph,
                        ),
                        "face": record,
                        "ts": now,
                    }
                else:
                    # No person carrier: keep the verified box until TTL.
                    self._loose.append(record)
            self._loose = self._loose[-8:]

    # ------------------------------------------------------------------- read
    def live_faces(
        self, tracks: Sequence[Dict[str, Any]], now: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Face boxes for the overlay, re-anchored to the CURRENT tracks.

        Returns one record per known face (same keys as verification plus
        ``interpolated``): anchored faces are re-projected into the current
        person bbox; expired or track-less faces are dropped.
        """
        now = time.time() if now is None else float(now)
        track_boxes: Dict[Any, Sequence[float]] = {
            t.get("track_id"): t.get("bbox")
            for t in tracks
            if t.get("track_id") is not None and t.get("bbox")
        }
        out: List[Dict[str, Any]] = []
        with self._lock:
            # TTL expiry (no stale box kept indefinitely).
            self._anchored = {
                tid: entry for tid, entry in self._anchored.items()
                if (now - entry["ts"]) <= self.ttl_seconds
            }
            # Track disappeared (person left / ID switched) -> drop.
            self._anchored = {
                tid: entry for tid, entry in self._anchored.items()
                if tid in track_boxes
            }
            self._loose = [
                record for record in self._loose
                if (now - record.get("ts", 0.0)) <= self.ttl_seconds
            ]
            for track_id, entry in self._anchored.items():
                person_bbox = track_boxes[track_id]
                px1, py1, px2, py2 = (float(v) for v in person_bbox[:4])
                pw = max(px2 - px1, 1e-6)
                ph = max(py2 - py1, 1e-6)
                rx1, ry1, rx2, ry2 = entry["rel"]
                fx1 = px1 + rx1 * pw
                fy1 = py1 + ry1 * ph
                fx2 = px1 + rx2 * pw
                fy2 = py1 + ry2 * ph
                if fx2 - fx1 < 2 or fy2 - fy1 < 2:
                    continue  # degenerate projection - skip this sweep
                face = {k: v for k, v in entry["face"].items() if k != "ts"}
                face["bbox"] = [
                    int(round(fx1)), int(round(fy1)), int(round(fx2)), int(round(fy2))
                ]
                face["track_id"] = track_id
                face["interpolated"] = True
                out.append(face)
            for record in self._loose:
                face = {k: v for k, v in record.items() if k != "ts"}
                face["interpolated"] = False
                out.append(face)
        return out

