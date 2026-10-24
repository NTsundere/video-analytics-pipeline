from app.services.tracker import IouTracker, iou


def det(name, x1, y1, x2, y2, score=0.9):
    return {"class": name, "score": score, "bbox": [x1, y1, x2, y2]}


def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_partial_overlap():
    assert round(iou([0, 0, 10, 10], [5, 0, 15, 10]), 3) == 0.333


def test_same_object_keeps_track_id():
    tracker = IouTracker()
    first = tracker.update([det("car", 0, 0, 100, 100)])
    second = tracker.update([det("car", 5, 5, 105, 105)])
    assert first[0].track_id == second[0].track_id


def test_different_classes_get_different_tracks():
    tracker = IouTracker()
    tracks = tracker.update([det("car", 0, 0, 100, 100), det("truck", 0, 0, 100, 100)])
    assert tracks[0].track_id != tracks[1].track_id


def test_two_objects_do_not_swap_ids():
    tracker = IouTracker()
    first = tracker.update([det("car", 0, 0, 50, 50), det("car", 200, 200, 250, 250)])
    second = tracker.update([det("car", 203, 203, 253, 253), det("car", 3, 3, 53, 53)])
    assert second[0].track_id == first[1].track_id
    assert second[1].track_id == first[0].track_id


def test_track_is_confirmed_once_after_min_hits():
    tracker = IouTracker(min_hits=3)
    for _ in range(2):
        tracker.update([det("car", 0, 0, 100, 100)])
        assert tracker.pop_confirmed() == []
    tracker.update([det("car", 0, 0, 100, 100)])
    assert len(tracker.pop_confirmed()) == 1
    tracker.update([det("car", 0, 0, 100, 100)])
    assert tracker.pop_confirmed() == []


def test_track_is_removed_after_max_missed():
    tracker = IouTracker(max_missed=2, min_hits=1)
    tracker.update([det("car", 0, 0, 100, 100)])
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == []


def test_track_survives_short_gap():
    tracker = IouTracker(max_missed=3, min_hits=1)
    first = tracker.update([det("car", 0, 0, 100, 100)])
    tracker.update([])
    tracker.update([])
    again = tracker.update([det("car", 2, 2, 102, 102)])
    assert again[0].track_id == first[0].track_id
