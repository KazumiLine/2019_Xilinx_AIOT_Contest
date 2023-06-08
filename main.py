import time
import cv2
import colorsys
from numpy import record
from tracker_manager import *
from mjpg_stream import *
# from data_collector import *
from background import *

def main():
    # video = VideoStream("https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=1031")
    # video = VideoStream('https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=10002')
    video = cv2.VideoCapture('data/demo_raw.avi')
    frameSize = (352, 240)
    tracker_manager = TrackerManager(frameSize)
    print("initialize background")
    initialCount = 0
    while initialCount < 200:
        ret, img = video.read()
        if not ret:
            continue
        tracker_manager.new_frame(img)
        cv2.imshow(f"{img.shape}", img)
        initialCount += 1

    backdec = BackgroundDetector(tracker_manager.backgroundObject.getBackgroundImage())
    print("done")

    while True:

        ret, img = video.read()
        org = img.copy()
        if not ret:
            break

        # t_start = time.time()
        orimg = img.copy()

        tracker_manager.new_frame(img)
        img = backdec.visualize_lines(img)

        for t in tracker_manager.trackers:

            x, y, w, h = center2bbox(t.cur_bbox)
            t.position = backdec.getClassify((x+w/2, y+h))
            cv2.rectangle(img, (x, y), (x + w, y + h), np.array(colorsys.hsv_to_rgb(t.created_time%0.1*10, 1, 1))*255, 2)
            cv2.putText(
                img,
                f"L{int(t.position)}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # cv2.putText(
        #     img,
        #     f"FPS: {1/(t_end - t_start):4.2f}",
        #     (10, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 255, 0),
        #     1,
        #     cv2.LINE_AA,
        # )

        cv2.imshow(f"{img.shape}", img)
        cv2.imshow(f"org", org)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord("w"):
            while not cv2.waitKey(1) & 0xFF == ord("w"):
                pass
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
