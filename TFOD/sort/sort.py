import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from kalman_tracker import KalmanBoxTracker
#from scipy.optimize import linear_sum_assignment as linear_assignment

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_id = 0
        self.trackers = []

    def update(self, detections):
        updated_trackers = []
        for track in self.trackers:
            if track.time_since_update < self.max_age:
                track.predict()
                updated_trackers.append(track)
        self.trackers = updated_trackers

        matches, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections)
        for m in matches:
            self.trackers[m[1]].update(detections[m[0]])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)

        tracked_objects = []
        for trk in self.trackers:
            if trk.hits >= self.min_hits or trk.time_since_update <= 1:
                d = trk.get_state()
                tracked_objects.append(np.hstack((d.reshape(1, -1), [[trk.id + 1]])))

        if tracked_objects:
            return np.vstack(tracked_objects)
        else:
            return np.empty((0, 5))  # Retournez un tableau vide avec 5 colonnes si tracked_objects est vide
    def _associate_detections_to_trackers(self, detections):
        iou_matrix = np.zeros((len(detections), len(self.trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self._iou(det, trk.get_state())

        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.column_stack((row_indices, col_indices))

        unmatched_detections = []
        for d, _ in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, _ in enumerate(self.trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    @staticmethod
    def _iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o
