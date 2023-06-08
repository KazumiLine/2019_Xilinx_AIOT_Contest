import cv2
import urllib.request
import numpy as np
import ssl
from threading import Thread, Lock
import time


def merge(master, addition):
    first = addition[0]
    n = max(len(master) - len(addition), 1)  # (1)
    while 1:
        try:
            n = master.index(first, n)  # (2)
        except ValueError:
            return master + addition

        if master[-n:] == addition[:n]:
            return master + addition[n:]
        n += 1


class VideoStream:
    def __init__(self, url, force_retry=True, buffer_size=1024 * 50) -> None:

        self.released = False
        self.url = url
        self.force_retry = force_retry
        self.buffer_size = buffer_size

        self.bytes = b""
        self.frame = None
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.mutex = Lock()
        Thread(target=self.downloader).start()
        time.sleep(1)

    def downloader(self):
        stream = urllib.request.urlopen(self.url, context=self.ctx)
        while not self.released:
            try:
                self.bytes += stream.read(4096)
            except:
                stream = urllib.request.urlopen(self.url, context=self.ctx)
                continue
            a = self.bytes.find(b"\xff\xd8")
            b = self.bytes.find(b"\xff\xd9")
            if a != -1 and b != -1 and a < b:
                jpg = self.bytes[a : b + 2]
                self.bytes = self.bytes[b + 2 :]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                self.frame = img

    def get_file(self):
        stream = urllib.request.urlopen(self.url, context=self.ctx)
        buffer = b""
        while len(buffer) < self.buffer_size:
            try:
                buffer += stream.read(1024)
            except:
                return
        self.mutex.acquire()
        self.bytes = merge(self.bytes, buffer)
        self.mutex.release()

    def read(self):
        if type(self.frame) == type(None):
            return False, None
        else:
            return  True, self.frame.copy()

    def release(self):
        self.released = True


if __name__ == "__main__":

    video = VideoStream("https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=10002")

    while True:
        ret, frame = video.read()
        cv2.imshow(f"{frame.shape}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()