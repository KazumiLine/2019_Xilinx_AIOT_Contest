import cv2
import numpy as np
import math
from tracker import processLiveFeed
# from data_collector import *
import time
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

xv2 = None
try:
    import pynq_cv.overlays.xv2CarTracking as xv2
    from pynq import Xlnk
    Xlnk.set_allocator_library("/usr/local/lib/python3.6/dist-packages/pynq_cv/overlays/xv2CarTracking.so")
    mem_manager = Xlnk()
    print("xilinx hardware acceleration")
except:
    print("improt hardware design error.")



def center_cnt(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return x + w // 2, y + h // 2


def crop_obj(img, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    mask_cnt = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask_cnt, [cnt], -1, (255,), -1)
    obj_img = cv2.bitwise_and(img, img, mask=mask_cnt)[x : x + w + 1, y : y + h + 1]
    return obj_img


def IOU_bbox(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0])  # x
    iou_y = max(bbox_ai[1], bbox_gt[1])  # y
    iou_w = min(bbox_ai[2] + bbox_ai[0], bbox_gt[2] + bbox_gt[0]) - iou_x  # w
    iou_w = max(iou_w, 0)

    iou_h = min(bbox_ai[3] + bbox_ai[1], bbox_gt[3] + bbox_gt[1]) - iou_y  # h
    iou_h = max(iou_h, 0)

    iou_area = iou_w * iou_h
    all_area = bbox_ai[2] * bbox_ai[3] + bbox_gt[2] * bbox_gt[3] - iou_area

    return max(iou_area / all_area, 0)


def IOU_cnt(cnt1, cnt2):
    bbox1 = cv2.boundingRect(cnt1)
    bbox2 = cv2.boundingRect(cnt2)
    return IOU_bbox(bbox1, bbox2)


def bbox2center(bbox):
    return (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3])


def center2bbox(bbox):
    return (bbox[0] - bbox[2] // 2, bbox[1] - bbox[3] // 2, bbox[2], bbox[3])


class Tracker:
    def __init__(self, id, frame, obj_cnt, min_iou=0.2) -> None:

        self.id = id

        self.pre_frame = frame
        self.cur_frame = frame
        self.pre_bbox = bbox2center(cv2.boundingRect(obj_cnt))
        self.cur_bbox = bbox2center(cv2.boundingRect(obj_cnt))
        self.obj_img = crop_obj(frame, obj_cnt)
        self.obj_cnt = obj_cnt

        self.record_bbox = []

        self.min_iou = min_iou
        self.max_dst = 70

        self.created_time = time.time()
        self.removed_time = time.time()
        self.duration = 0

        self.model = LinearRegression()
        self.predict = np.zeros((1, 0))
        self.position = 0

    def on_remove(self):

        self.model.fit(
            np.array([x[0][0] for x in self.record_bbox]).reshape(-1, 1),
            [x[0][1] for x in self.record_bbox],
        )
        self.predict = self.model.predict(
            np.array([x[0][0] for x in self.record_bbox]).reshape(-1, 1)
        )

        # global data_collector
        # data_collector.update_tracker_info(self)
        print(f"tracker removed id:{self.id} duration:{self.duration}")

        return [x[0][0] for x in self.record_bbox], self.predict

    def record(self, frame, contours) -> None:
        self.removed_time = time.time()
        self.duration = self.removed_time - self.created_time
        self.record_bbox.append((self.cur_bbox, time.time()))

    def matching_orb(self, frame, contours) -> bool:

        self.record(frame, contours)

        if len(contours) == 0:
            print(f"empty contours, id:{self.id}")
            return False

        try:
            self.pre_bbox, self.cur_bbox = processLiveFeed(
                self.pre_frame, self.cur_frame, self.pre_bbox, self.cur_bbox
            )
        except:
            print(f"processLiveFeed failed, id:{self.id}")
            return False

        self.pre_frame = self.cur_frame
        self.cur_frame = frame

        # contours_IOUs = [(cnt, IOU_bbox(cv2.boundingRect(
        #     cnt), center2bbox(self.cur_bbox))) for cnt in contours]
        # best_cnt, best_iou = max(contours_IOUs, key=lambda cnt_iou: cnt_iou[1])

        # if best_iou > self.min_iou:
        #     self.obj_cnt = best_cnt
        #     return True
        # else:
        #     print(f"IOU too small iou:{best_iou}, id:{self.id}")
        #     return False

        contours_DSTs = [
            (cnt, math.dist(center_cnt(cnt), center_cnt(self.obj_cnt)))
            for cnt in contours
        ]
        best_cnt, best_dst = min(contours_DSTs, key=lambda cnt_dst: cnt_dst[1])
        if best_dst < self.max_dst:
            self.obj_cnt = best_cnt
            self.cur_bbox = bbox2center(cv2.boundingRect(best_cnt))
            return True
        else:
            print(f"dst too large dst:{best_dst}, id:{self.id}")
            return False


class TrackerManager:
    def __init__(self, frame_shape, min_bbox=np.array([0.07, 0.07])) -> None:
        self.backgroundObject = cv2.createBackgroundSubtractorKNN(
            dist2Threshold=400.0, detectShadows=False, history=10000
        )
        self.trackers = []
        self.kernel = None
        self.counter = 0
        w, h = frame_shape[:2] * min_bbox
        self.min_area = w * h

        self.max_IOU = 0.001
        self.min_IOU = 0.7
        # plt.ion()
        # plt.title("Dynamic Plot of sinx", fontsize=25)
        # plt.xlabel("X", fontsize=18)
        # plt.ylabel("sinX", fontsize=18)
        # self.figure, self.ax = plt.subplots(figsize=(8, 6))

    def get_free_id(self) -> int:
        self.counter += 1
        return self.counter

    def new_frame(self, frame) -> None:
        # Pre-Prossing
        start_t = time.time()

        fgmask = self.backgroundObject.apply(frame)

        # print(f"backgroundObject {time.time()-start_t}")
        start_t = time.time()

        initialMask = fgmask.copy()

        if xv2 != None:
            xfmem = mem_manager.cma_array(fgmask.shape, np.uint8)
            xfbuf = mem_manager.cma_array(fgmask.shape, np.uint8)
            xfmem[:] = fgmask[:]
            xv2.threshold(xfmem, 254, 255, xv2.THRESH_BINARY, dst=xfbuf)
            xv2.erode(xfbuf, self.kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, dst=xfmem)
            xv2.dilate(xfmem, self.kernel, iterations=3, borderType=cv2.BORDER_CONSTANT, dst=xfbuf)
            fgmask[:] = xfbuf[:]
        else:
            _, fgmask = cv2.threshold(fgmask, 254, 255, cv2.THRESH_BINARY)
            fgmask = cv2.erode(fgmask, self.kernel, iterations=1)
            fgmask = cv2.dilate(fgmask, self.kernel, iterations=3)

        masked_frame = cv2.bitwise_and(frame, frame, mask=fgmask)

        # print(f"Pre-Processing {time.time()-start_t}")
        start_t = time.time()

        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # print(f"findContours {time.time()-start_t}")
        start_t = time.time()

        # update & remove trackers
        new_trackers = []
        for tracker in self.trackers:
            if (
                tracker.matching_orb(masked_frame, contours)
                and (tracker.cur_bbox[2] * tracker.cur_bbox[3]) > 500
            ):
                new_trackers.append(tracker)
            else:
                x, y = tracker.on_remove()
                # if tracker.duration > 2:
                #     self.ax.plot(x, y)
                #     self.figure.canvas.draw()
                #     self.figure.canvas.flush_events()
                #     time.sleep(0.1)
        self.trackers = new_trackers

        for cnt in contours:

            if cv2.contourArea(cnt) < self.min_area:
                continue

            for tracker in self.trackers:
                if IOU_cnt(tracker.obj_cnt, cnt) > self.max_IOU:
                    break
            else:
                new_tracker = Tracker(self.get_free_id(), masked_frame, cnt)
                self.trackers.append(new_tracker)

        # print(f"update tracker {time.time()-start_t}")
        start_t = time.time()

        # for DEBUG
        # bg_img = self.backgroundObject.getBackgroundImage()
        # cv2.imshow("Background", bg_img)
        # cv2.imshow("initial Mask", initialMask)
        # cv2.imshow("Clean Mask", fgmask)
        # cv2.imwrite('background.jpg', bg_img)

