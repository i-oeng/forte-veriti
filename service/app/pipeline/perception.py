"""Person (silhouette) and object detection with RF-DETR, person tracking with ByteTrack.

No face detection or recognition is done anywhere: people are handled only as body boxes.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .. import config

# COCO category ids as returned by RF-DETR
COCO = {
    1: "person", 27: "backpack", 28: "umbrella", 31: "handbag", 33: "suitcase",
    44: "bottle", 46: "wine glass", 47: "cup", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 67: "dining table", 72: "tv", 73: "laptop", 76: "keyboard", 77: "cell phone", 84: "book",
}
GROUPS = {
    "personal": {"backpack", "handbag", "suitcase", "umbrella"},
    "phone": {"cell phone"},
    "food": {"bottle", "wine glass", "cup", "bowl", "banana", "apple", "sandwich", "orange", "hot dog",
             "pizza", "donut", "cake"},
}
PERSON_MIN_CONF = 0.45
OBJECT_MIN_CONF = 0.35


@dataclass
class Det:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in analysis-frame pixels
    label: str
    conf: float
    track: int | None = None


@dataclass
class FrameDets:
    t: float
    persons: list[Det] = field(default_factory=list)
    objects: list[Det] = field(default_factory=list)


class Detector:
    """Process-wide RF-DETR instance; GPU calls are serialised with a lock and batched."""

    _instance: "Detector | None" = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        import rfdetr

        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[config.DETECTOR]
        kwargs = {"pretrain_weights": config.DETECTOR_WEIGHTS} if config.DETECTOR_WEIGHTS else {}
        self.model = cls(device=config.DETECTOR_DEVICE, **kwargs)
        self.lock = threading.Lock()

    @classmethod
    def get(cls) -> "Detector":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = Detector()
            return cls._instance

    def detect(self, images_bgr: list[np.ndarray], batch: int = 4) -> list[list[Det]]:
        out: list[list[Det]] = []
        for i in range(0, len(images_bgr), batch):
            chunk = [np.ascontiguousarray(img[:, :, ::-1]) for img in images_bgr[i:i + batch]]
            with self.lock:
                preds = self.model.predict(chunk, threshold=min(PERSON_MIN_CONF, OBJECT_MIN_CONF),
                                           include_source_image=False)
            if not isinstance(preds, list):
                preds = [preds]
            for p in preds:
                dets = []
                for box, cid, conf in zip(p.xyxy, p.class_id, p.confidence):
                    label = COCO.get(int(cid))
                    if label is None:
                        continue
                    dets.append(Det(tuple(float(v) for v in box), label, float(conf)))
                out.append(dets)
        return out


def run(frames, fps: float) -> list[FrameDets]:
    """Detect on every sampled frame and attach ByteTrack ids to persons."""
    import supervision as sv
    from trackers import ByteTrackTracker

    images = [f.image for f in frames]
    raw = Detector.get().detect(images)
    tracker = ByteTrackTracker(lost_track_buffer=int(fps * 2), frame_rate=fps, track_activation_threshold=0.5,
                               minimum_consecutive_frames=2, high_conf_det_threshold=0.6)
    result: list[FrameDets] = []
    for frame, dets in zip(frames, raw):
        persons = [d for d in dets if d.label == "person" and d.conf >= PERSON_MIN_CONF]
        objects = [d for d in dets if d.label != "person" and d.conf >= OBJECT_MIN_CONF]
        sv_dets = sv.Detections(xyxy=np.array([p.box for p in persons], dtype=np.float32).reshape(-1, 4),
                                confidence=np.array([p.conf for p in persons], dtype=np.float32),
                                class_id=np.zeros(len(persons), dtype=int))
        tracked = tracker.update(sv_dets, timestamp=frame.t)
        # every raw person box counts; track ids are attached where the tracker confirmed a track
        if len(tracked) and tracked.tracker_id is not None:
            for p in persons:
                ious = [_iou(p.box, tuple(b)) for b in tracked.xyxy]
                k = int(np.argmax(ious))
                if ious[k] > 0.5 and tracked.tracker_id[k] >= 0:
                    p.track = int(tracked.tracker_id[k])
        result.append(FrameDets(frame.t, persons, objects))
    return result


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def group_of(label: str) -> str | None:
    for g, names in GROUPS.items():
        if label in names:
            return g
    return None
