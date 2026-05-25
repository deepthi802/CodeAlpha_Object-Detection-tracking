from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)

    wh = w * h

    o = wh / (
        ((bb_test[2]-bb_test[0]) *
         (bb_test[3]-bb_test[1])) +

        ((bb_gt[2]-bb_gt[0]) *
         (bb_gt[3]-bb_gt[1])) - wh
    )

    return o


class KalmanBoxTracker(object):

    count = 0

    def __init__(self, bbox):

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])

        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

    def convert_bbox_to_z(self, bbox):

        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x = bbox[0] + w/2.
        y = bbox[1] + h/2.

        s = w * h
        r = w / float(h)

        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x):

        w = np.sqrt(x[2] * x[3])
        h = x[2] / w

        return np.array([
            x[0]-w/2.,
            x[1]-h/2.,
            x[0]+w/2.,
            x[1]+h/2.
        ]).reshape((1,4))

    def update(self, bbox):

        self.time_since_update = 0
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):

        self.kf.predict()
        self.time_since_update += 1

        return self.convert_x_to_bbox(self.kf.x)

    def get_state(self):

        return self.convert_x_to_bbox(self.kf.x)


class Sort(object):

    def __init__(self):

        self.trackers = []

    def update(self, detections):

        results = []

        for tracker in self.trackers:
            tracker.predict()

        matched = []

        for det in detections:

            best_iou = 0
            best_tracker = None

            for tracker in self.trackers:

                pred_box = tracker.get_state()[0]

                score = iou(det[:4], pred_box)

                if score > best_iou:
                    best_iou = score
                    best_tracker = tracker

            if best_iou > 0.3:

                best_tracker.update(det[:4])
                matched.append(best_tracker)

                box = best_tracker.get_state()[0]

                results.append([
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                    best_tracker.id
                ])

            else:

                tracker = KalmanBoxTracker(det[:4])

                self.trackers.append(tracker)

                box = tracker.get_state()[0]

                results.append([
                    int(box[0]),
                    int(box[1]),
                    int(box[2]),
                    int(box[3]),
                    tracker.id
                ])

        return results
