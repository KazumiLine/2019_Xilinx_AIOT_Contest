from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import time


def is_invertible(a):
    return np.linalg.det(a) != 0


# y = coef_*x + intercept_
# normalize vector (1,-coef_)
def model2line(model):
    # ax + by = c
    norm1 = (model.coef_[0] ** 2 + 1) ** 0.5
    a = -model.coef_[0] / norm1
    b = 1.0 / norm1
    c = model.intercept_ / norm1
    return a, b, c


def get_x_interval(a, b, c):

    frame_shape = (240, 352)
    w = frame_shape[0]
    h = frame_shape[1]

    # slove Av = B
    A = np.array([[a, b], [0, 1]])
    B = np.array([c, 0])
    if is_invertible(A):
        x1, _ = np.linalg.solve(A, B)
    else:
        x1 = 0

    A = np.array([[a, b], [0, 1]])
    B = np.array([c, h])
    if is_invertible(A):
        x2, _ = np.linalg.solve(A, B)
    else:
        x2 = w

    return max(0, min(x1, x2)), min(w, max(x1, x2))


def line_point_dst(l, pt):
    a, b, c = l
    x, y = pt
    return abs(a * x + b * y - c) / (a ** 2 + b ** 2) ** 0.5


def avg_integral_dst(l1, x_lower, x_upper, l2):

    if abs(x_lower - x_upper) < 1e-3:
        return 0.0

    a1, b1, c1 = l1
    a2, b2, c2 = l2

    A = np.array([[a1, b1], [a2, b2]])
    B = np.array([c1, c2])
    if is_invertible(A):
        pt_x, _ = np.linalg.solve(A, B)
    else:
        pt_x = 0

    """
    integral abs(ax+by-c)/sqrt(a^2+b^2)
    (x,y) =  (x1, (c1-a1x1)/b1)
    l1 as variable
    l2 as target
    """
    integral_fnc = (
        lambda x: (-a1 * b2 / 2 / b1 + a2 / 2) * x ** 2 + (b2 * c1 / b1 - c2) * x
    )

    area = 0

    if x_lower < pt_x and pt_x < x_upper:
        area += abs(integral_fnc(pt_x) - integral_fnc(x_lower))
        area += abs(integral_fnc(x_upper) - integral_fnc(pt_x))
    else:
        area += abs(integral_fnc(x_upper) - integral_fnc(x_lower))
    print(area, x_upper, x_lower)
    return area / abs(x_upper - x_lower)


def avg_dst(l1, l2):
    a1, b1, c1, x_lower1, x_upper1 = l1
    a2, b2, c2, x_lower2, x_upper2 = l2
    return (
        avg_integral_dst((a1, b1, c1), x_lower1, x_upper1, (a2, b2, c2))
        + avg_integral_dst((a2, b2, c2), x_lower2, x_upper2, (a1, b1, c1))
    ) / 2


from sklearn.metrics import pairwise_distances


def custom_affinity(X):
    return pairwise_distances(X, metric=avg_dst)


class DataCollector:
    def __init__(self) -> None:
        self.tracker_models = []
        self.cluster_model = None
        self.result = None
        self.min_samples = 20
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlim(0, 240)
        self.ax.set_ylim(0, 350)
        self.ax.invert_yaxis()

    def start_train(self):
        print("Start training...")
        self.cluster_model = AgglomerativeClustering(
            n_clusters=None,
            affinity=custom_affinity,
            linkage="average",
            distance_threshold=50,
        )

        self.cluster_model.fit(np.array(self.tracker_models, dtype=float))

        self.result = []
        for label in range(max(self.cluster_model.labels_) + 1):
            vectors = np.array(self.tracker_models)[self.cluster_model.labels_ == label]
            self.result.append(vectors[:, :3].mean(0))

        print(f"n_cluster:{max(self.cluster_model.labels_)+1}")
        print(f"train result:{self.result}")

        # self.figure, self.ax = plt.subplots(figsize=(8, 6))
        # self.ax.invert_yaxis()

        for (a, b, c) in self.result:
            print(a, b, c)
            lower_x, upper_x = get_x_interval(a, b, c)
            print(lower_x, upper_x)
            x = list(range(int(lower_x), int(upper_x)))
            y = [(c - a * i) / b for i in x]
            self.ax.plot(x, y)
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()
            time.sleep(0.1)

        print("Done")

    def update_tracker_info(self, tracker):
        if tracker.duration > 1.0:
            x_start = tracker.record_bbox[0][0][0]
            x_end = tracker.record_bbox[len(tracker.record_bbox) - 1][0][0]

            print(x_start, x_end)
            self.tracker_models.append(
                (*model2line(tracker.model), min(x_start, x_end), max(x_start, x_end))
            )
            print(np.array(self.tracker_models, dtype=float).shape)
            # a, b, c = model2line(tracker.model)
            # x = list(range(0, 230))
            # y = [i * tracker.model.coef_.T + tracker.model.intercept_ for i in x]
            # self.ax.plot(x, y)
            # self.figure.canvas.draw()
            # self.figure.canvas.flush_events()

        else:
            return

        if self.cluster_model == None and len(self.tracker_models) > self.min_samples:
            self.start_train()
        # if self.cluster_model != None:
        #     dst_matrix = [
        #         (idx, avg_dst(vec, model2line(tracker.model)))
        #         for idx, vec in enumerate(self.result)
        #     ]
        #     res, _ = min(dst_matrix, key=lambda x: x[1])
        #     print(f"id:{tracker.id},group:{res}")


data_collector = DataCollector()

if __name__ == "__main__":
    pass
