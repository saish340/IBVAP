"""Face verification pipeline (RetinaFace detection + ArcFace embeddings).

Components:

* :class:`WatchlistManager` - persists face embeddings (SQLite) and matches
  new embeddings against the watchlist via cosine similarity (threshold 0.6).
* :class:`FaceVerificationEngine` - ``process_frame(frame)`` detects every
  face with DeepFace's RetinaFace backend, extracts an ArcFace embedding per
  face, matches it against the watchlist, and returns records with bounding
  boxes, names and confidence scores (plus an annotated frame).

Live demo (enroll from a photo, then verify a webcam / video feed):

    python inference/face_verification.py --enroll Alice alice.jpg --source 0
    python inference/face_verification.py --source rtsp://127.0.0.1:8554/cctv --no-window
    python inference/face_verification.py --enroll Bob bob.png --source samples/demo.mp4

Requires: deepface (TensorFlow + retina-face install automatically with it).
Model weights download on first use. Watchlist persists to ``watchlist.db``
(gitignored) unless ``--watchlist`` points elsewhere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

#: cosine similarity above which an identity is accepted
DEFAULT_MATCH_THRESHOLD = 0.6
#: default SQLite file backing :class:`WatchlistManager` (project root)
DEFAULT_DB_PATH = "watchlist.db"


class WatchlistManager:
    """Persistent face watchlist (SQLite) with cosine-similarity matching.

    Embeddings are L2-normalised when stored, so matching is a single dot
    product. Only rows produced by the same embedding model are compared
    (ArcFace vectors are incompatible with other models' spaces).

    Args:
        db_path: SQLite file; created (and tables initialised) on demand.
        threshold: cosine similarity required to accept an identity.
        model_name: embedding model these rows belong to (e.g. ``ArcFace``).
    """

    def __init__(
        self,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        model_name: str = "ArcFace",
    ) -> None:
        self.db_path = str(db_path)
        self.threshold = float(threshold)
        self.model_name = model_name
        parent = Path(self.db_path).parent
        if str(parent) not in ("", ".", str(Path(".").resolve())):
            # SQLite cannot create missing parent directories.
            parent.mkdir(parents=True, exist_ok=True)
        self._query(
            """CREATE TABLE IF NOT EXISTS watchlist (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   name TEXT NOT NULL,
                   model TEXT NOT NULL,
                   embedding BLOB NOT NULL,
                   enrolled_at TEXT NOT NULL
               )"""
        )
        self._query(
            "CREATE INDEX IF NOT EXISTS idx_watchlist_name_model ON watchlist(name, model)"
        )

    # ----------------------------------------------------------- persistence
    def _query(self, sql: str, params: Sequence = (), fetch: Optional[str] = None):
        """Run one statement on its own connection (commit + close)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(sql, params)
            if fetch == "all":
                return cur.fetchall()
            if fetch == "one":
                return cur.fetchone()
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    # ------------------------------------------------------------- enrolment
    def add_embedding(
        self, name: str, embedding: Sequence[float], model_name: Optional[str] = None
    ) -> int:
        """Store one L2-normalised embedding for ``name`` (returns row id).

        Useful for enrolling programmatically without an image; the image
        based flow is :meth:`enroll`.
        """
        vector = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("embedding has zero norm - cannot store it")
        vector = vector / norm
        return int(
            self._query(
                "INSERT INTO watchlist (name, model, embedding, enrolled_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    name,
                    model_name or self.model_name,
                    vector.tobytes(),
                    dt.datetime.now().isoformat(timespec="seconds"),
                ),
            )
        )

    def enroll(
        self, name: str, image: Union[str, Path, np.ndarray], **represent_kwargs: Any
    ) -> Dict[str, Any]:
        """Detect faces in ``image`` and store the most prominent one.

        Uses RetinaFace detection + ArcFace embedding via DeepFace.  If the
        image contains several faces, the largest one is enrolled (so group
        photos do not fail); the result dict reports how many were found.
        """
        from deepface import DeepFace

        kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "detector_backend": "retinaface",
            "enforce_detection": False,
        }
        kwargs.update(represent_kwargs)
        reps = DeepFace.represent(img_path=image, **kwargs)
        reps = [r for r in reps or [] if float(r.get("face_confidence") or 0.0) > 0]
        if not reps:
            raise ValueError(f"no face detected in {image!r} - cannot enroll '{name}'")

        best = max(
            reps, key=lambda r: int(r["facial_area"]["w"]) * int(r["facial_area"]["h"])
        )
        area = best["facial_area"]
        self.add_embedding(name, best["embedding"])
        return {
            "name": name,
            "bbox": [int(area["x"]), int(area["y"]),
                     int(area["x"]) + int(area["w"]), int(area["y"]) + int(area["h"])],
            "face_confidence": round(float(best.get("face_confidence") or 0.0), 4),
            "faces_in_image": len(reps),
        }

    # --------------------------------------------------------------- matching
    def match(self, embedding: Sequence[float]) -> Tuple[str, float]:
        """Cosine-match ``embedding`` against every stored row.

        Returns:
            ``(name, confidence)`` of the best identity when the similarity
            is ``>= threshold``; otherwise ``("UNKNOWN", confidence)``.
            With an empty watchlist, returns ``("UNKNOWN", 0.0)``.
        """
        rows = self._query(
            "SELECT name, embedding FROM watchlist WHERE model = ?",
            (self.model_name,),
            fetch="all",
        )
        if not rows:
            return ("UNKNOWN", 0.0)

        query = np.asarray(embedding, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(query))
        if norm <= 0:
            return ("UNKNOWN", 0.0)
        query = query / norm

        best_name, best_score = "UNKNOWN", -1.0
        for name, blob in rows:
            vector = np.frombuffer(blob, dtype=np.float32)
            score = float(np.dot(query, vector))  # cosine: both normalised
            if score > best_score:
                best_name, best_score = name, score

        if best_score >= self.threshold:
            return (best_name, round(best_score, 4))
        return ("UNKNOWN", round(best_score, 4))

    # ------------------------------------------------------------- management
    def names(self) -> List[str]:
        """Distinct enrolled identities."""
        rows = self._query(
            "SELECT DISTINCT name FROM watchlist ORDER BY name", fetch="all"
        )
        return [row[0] for row in rows]

    def count(self) -> int:
        """Number of stored embeddings for this engine's model."""
        row = self._query(
            "SELECT COUNT(*) FROM watchlist WHERE model = ?",
            (self.model_name,),
            fetch="one",
        )
        return int(row[0])

    def remove(self, name: str) -> int:
        """Delete every embedding stored for ``name`` (returns rows removed)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute("DELETE FROM watchlist WHERE name = ?", (name,))
            conn.commit()
            return int(cur.rowcount)
        finally:
            conn.close()

class FaceVerificationEngine:
    """RetinaFace face detection + ArcFace verification against a watchlist.

    Example:
        engine = FaceVerificationEngine(db_path="watchlist.db")
        engine.enroll("Alice", "alice.jpg")
        results = engine.process_frame(frame)
        # -> [{"bbox": [x1, y1, x2, y2], "name": "Alice",
        #      "confidence": 0.81, "face_confidence": 0.99}, ...]
        annotated = engine.annotated_frame  # boxes + names drawn on a copy
    """

    def __init__(
        self,
        watchlist: Optional[WatchlistManager] = None,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        detector_backend: str = "retinaface",
        model_name: str = "ArcFace",
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        align: bool = True,
        draw: bool = True,
    ) -> None:
        from deepface import DeepFace  # heavy import kept lazy

        self._deepface = DeepFace
        self.detector_backend = detector_backend
        self.model_name = model_name
        self.align = align
        self.draw_enabled = draw
        self.watchlist = watchlist or WatchlistManager(
            db_path=db_path, threshold=threshold, model_name=model_name
        )
        #: last annotated frame from :meth:`process_frame` (None if draw off)
        self.annotated_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ API
    def enroll(self, name: str, image: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Enroll ``name`` from a photo (delegates to the watchlist)."""
        return self.watchlist.enroll(name, image)

    def process_frame(
        self, frame: np.ndarray, draw: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Detect all faces in a BGR frame and verify them against the watchlist.

        Returns:
            One dict per detected face::

                {"bbox": [x1, y1, x2, y2], "name": "Alice" | "UNKNOWN",
                 "confidence": <cosine similarity to the watchlist>,
                 "face_confidence": <RetinaFace detector confidence>}

            The annotated frame (boxes + names; green = recognised, red =
            unknown) is available as ``self.annotated_frame``.
        """
        if draw is None:
            draw = self.draw_enabled

        reps = self._deepface.represent(
            img_path=frame,
            model_name=self.model_name,
            detector_backend=self.detector_backend,
            enforce_detection=False,
            align=self.align,
        )

        results: List[Dict[str, Any]] = []
        for rep in reps or []:
            face_confidence = float(rep.get("face_confidence") or 0.0)
            if face_confidence <= 0:
                # enforce_detection=False can yield a full-image placeholder
                # with zero confidence when nothing was detected - skip it.
                continue
            area = rep["facial_area"]
            embedding = np.asarray(rep["embedding"], dtype=np.float32)
            name, score = self.watchlist.match(embedding)
            results.append(
                {
                    "bbox": [int(area["x"]), int(area["y"]),
                             int(area["x"]) + int(area["w"]),
                             int(area["y"]) + int(area["h"])],
                    "name": name,
                    "confidence": float(score),
                    "face_confidence": round(face_confidence, 4),
                }
            )

        self.annotated_frame = self._annotate(frame, results) if draw else None
        return results

    # ------------------------------------------------------------- internals
    def _annotate(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """Draw boxes + 'name score' labels (green match, red unknown)."""
        scene = frame.copy()  # supervision-style: never mutate caller's frame
        for record in results:
            x1, y1, x2, y2 = (int(v) for v in record["bbox"])
            known = record["name"] != "UNKNOWN"
            color = (0, 200, 0) if known else (0, 0, 220)
            cv2.rectangle(scene, (x1, y1), (x2, y2), color, 2)
            label = f"{record['name']} {record['confidence']:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            text_y = y1 - 6 if y1 - text_h - 8 >= 0 else y2 + text_h + 6
            cv2.rectangle(
                scene, (x1, text_y - text_h - 4), (x1 + text_w + 4, text_y + 4),
                color, -1,
            )
            cv2.putText(
                scene, label, (x1 + 2, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return scene

# ----------------------------------------------------- live demo (__main__)
def _open_capture(source: Union[int, str], attempts: int = 10, delay: float = 1.0):
    """Open a webcam / RTSP stream / video file, retrying while it starts."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = None
    for attempt in range(1, attempts + 1):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok:
                return cap
        cap.release()
        print(f"[verify] opening {source} ({attempt}/{attempts}) ...", flush=True)
        time.sleep(delay)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Face verification demo: RetinaFace detection + ArcFace "
                    "matching against a persisted watchlist.",
    )
    parser.add_argument(
        "--enroll", action="append", nargs=2, metavar=("NAME", "PHOTO"), default=[],
        help="enroll NAME from PHOTO before verifying (repeatable)",
    )
    parser.add_argument(
        "--source", default="0",
        help="webcam index, video file, or RTSP url (default: webcam 0)",
    )
    parser.add_argument("--watchlist", default=DEFAULT_DB_PATH, help="SQLite file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--detector", default="retinaface",
                        help="deepface detector backend (default: retinaface)")
    parser.add_argument("--model", default="ArcFace", help="deepface model name")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run until 'q'/'Esc')")
    parser.add_argument("--no-window", action="store_true",
                        help="headless: print stats only, no GUI")
    parser.add_argument("--connect-attempts", type=int, default=10)
    args = parser.parse_args()

    engine = FaceVerificationEngine(
        db_path=args.watchlist,
        threshold=args.threshold,
        detector_backend=args.detector,
        model_name=args.model,
    )

    for name, photo in args.enroll:
        try:
            info = engine.enroll(name, photo)
            print(
                f"[verify] enrolled '{name}' from {photo}: bbox={info['bbox']} "
                f"(detector conf {info['face_confidence']}, "
                f"{info['faces_in_image']} face(s) in image)"
            )
        except Exception as exc:
            print(
                f"error: enrollment failed for '{name}' ({photo}): {exc}",
                file=sys.stderr,
            )
            return 1
    print(
        f"[verify] watchlist {args.watchlist}: {engine.watchlist.names()} "
        f"({engine.watchlist.count()} embeddings)"
    )

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = _open_capture(source, attempts=args.connect_attempts)
    if cap is None:
        print(f"error: could not open source {args.source}", file=sys.stderr)
        return 1
    print(
        f"[verify] source: {args.source}"
        + ("" if args.no_window else "  -  press 'q' in the window to quit")
    )

    frames = 0
    fps_frames = 0
    fps = 0.0
    fps_last = time.perf_counter()
    last_summary = ""
    failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    print("error: stream stopped delivering frames", file=sys.stderr)
                    return 1
                time.sleep(0.05)
                continue
            failures = 0

            started = time.perf_counter()
            results = engine.process_frame(frame)
            inference_ms = (time.perf_counter() - started) * 1000
            frames += 1
            fps_frames += 1
            now = time.perf_counter()
            if now - fps_last >= 1.0:
                fps = fps_frames / (now - fps_last)
                fps_frames = 0
                fps_last = now

            summary = ", ".join(
                f"{r['name']}({r['confidence']:.2f})" for r in results
            ) or "no faces"
            if args.no_window:
                if frames == 1 or summary != last_summary:
                    print(
                        f"[verify] frame {frames:5d}  {fps:5.1f} fps  "
                        f"{inference_ms:6.0f} ms  | {summary}"
                    )
                    last_summary = summary
            else:
                annotated = (
                    engine.annotated_frame
                    if engine.annotated_frame is not None
                    else frame
                )
                h, w = annotated.shape[:2]
                cv2.putText(
                    annotated,
                    f"{fps:4.1f} fps  {inference_ms:4.0f} ms  faces: {summary}",
                    (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.imshow("IBVAP face verification (RetinaFace + ArcFace)", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            if args.frames and frames >= args.frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[verify] processed {frames} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())



