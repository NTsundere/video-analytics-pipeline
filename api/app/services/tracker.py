from dataclasses import dataclass


def iou(a: list[float], b: list[float]) -> float:
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    object_class: str
    bbox: list[float]
    confidence: float
    hits: int = 1
    missed: int = 0
    reported: bool = False


class IouTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15, min_hits: int = 3):
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return self._tracks

    def update(self, detections: list[dict]) -> list[Track]:
        order = sorted(range(len(detections)), key=lambda i: detections[i]["score"], reverse=True)
        assigned: dict[int, Track] = {}
        taken: set[int] = set()
        for index in order:
            detection = detections[index]
            best = None
            best_iou = self._iou_threshold
            for track in self._tracks:
                if track.track_id in taken or track.object_class != detection["class"]:
                    continue
                value = iou(track.bbox, detection["bbox"])
                if value >= best_iou:
                    best = track
                    best_iou = value
            if best is None:
                best = Track(
                    self._next_id,
                    detection["class"],
                    list(detection["bbox"]),
                    detection["score"],
                )
                self._next_id += 1
                self._tracks.append(best)
            else:
                best.bbox = list(detection["bbox"])
                best.confidence = detection["score"]
                best.hits += 1
                best.missed = 0
            taken.add(best.track_id)
            assigned[index] = best
        for track in self._tracks:
            if track.track_id not in taken:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self._max_missed]
        return [assigned[index] for index in range(len(detections))]

    def pop_confirmed(self) -> list[Track]:
        confirmed = [
            track for track in self._tracks if not track.reported and track.hits >= self._min_hits
        ]
        for track in confirmed:
            track.reported = True
        return confirmed
