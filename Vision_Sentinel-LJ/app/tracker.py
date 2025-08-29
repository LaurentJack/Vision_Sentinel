from typing import List, Dict, Any
from .utils import iou

class SimpleTracker:
    def __init__(self, max_age_frames=30, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {}
        self.max_age = max_age_frames
        self.iou_thr = iou_threshold
        self.frame_idx = 0

    def update(self, detections: List[Dict[str, Any]], class_name='person'):
        self.frame_idx += 1
        persons = [d for d in detections if d["name"] == class_name]
        assigned = set()
        matches = []
        for tid, t in list(self.tracks.items()):
            best_iou = 0.0
            best_j = -1
            for j, det in enumerate(persons):
                if j in assigned:
                    continue
                score = iou(t["box"], det["box"])
                if score > best_iou:
                    best_iou = score
                    best_j = j
            if best_j >= 0 and best_iou >= self.iou_thr:
                matches.append((tid, best_j))
                assigned.add(best_j)

        for tid, j in matches:
            self.tracks[tid]["box"] = persons[j]["box"]
            self.tracks[tid]["last_update"] = self.frame_idx

        for j, det in enumerate(persons):
            if j not in assigned:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"box": det["box"], "last_update": self.frame_idx}

        to_del = [tid for tid, t in self.tracks.items() if (self.frame_idx - t["last_update"]) > self.max_age]
        for tid in to_del: del self.tracks[tid]

        outputs = []
        for det in detections:
            out = det.copy()
            if det["name"] == class_name:
                best_tid, best_score = None, 0.0
                for tid, t in self.tracks.items():
                    score = iou(t["box"], det["box"])
                    if score > best_score:
                        best_score = score; best_tid = tid
                out["track_id"] = best_tid
            else:
                out["track_id"] = None
            outputs.append(out)
        return outputs
