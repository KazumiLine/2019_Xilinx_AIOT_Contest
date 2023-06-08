import numpy as np 
import cv2, math
from scipy.spatial import distance as dist

class BackgroundDetector:
    def __init__(self, backgroundImg):
        self.backgroundImg = backgroundImg
        canny = self.do_canny()
        segment = self.do_segment(canny)
        hough = cv2.HoughLinesP(segment, 2, np.pi/180, 4, minLineLength=100, maxLineGap=5)
        self.lines = hough[:,0, :]
        self.lines = self.deleteRepeatLines()
        self.lines = self.calculate_lines(backgroundImg)
        self.lines = sorted(self.lines, key = lambda s: s[0])
        self.lines = [self.lines[0], self.lines[-1]]
        segment = cv2.fillPoly(canny, np.array([[[0, self.backgroundImg.shape[0]], [0, 0], [self.backgroundImg.shape[1], 0], [self.backgroundImg.shape[1], self.backgroundImg.shape[0]], self.lines[1][:2], self.lines[1][2:], self.lines[0][2:], self.lines[0][:2]]]), (0, 0, 0))
        segment = self.do_segment(segment)
        hough = cv2.HoughLinesP(segment, 1, np.pi/180, 8, minLineLength=80, maxLineGap=4)
        self.lines = hough[:,0, :]
        self.lines = self.deleteRepeatLines()
        self.lines = self.calculate_lines(backgroundImg)
        # self.lines = sorted(self.lines, key = lambda s: s[0])
        # self.transM, self.maxWidth, self.maxHeight = self.get_four_point()

    def do_canny(self):
        gray = cv2.cvtColor(self.backgroundImg,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        canny = cv2.Canny(blur,50,150)
        return canny

    def do_segment(self, frame):
        frame = cv2.dilate(frame, None, iterations=2)
        frame = cv2.erode(frame, None, iterations=1)
        orign = cv2.cvtColor(self.backgroundImg, cv2.COLOR_RGB2GRAY)
        _, orign = cv2.threshold(orign, 210, 255, cv2.THRESH_BINARY)
        orign = cv2.dilate(orign, None, iterations=1)
        segment = cv2.bitwise_and(frame, frame, mask=orign) 
        segment[:45,:] = 0
        # segment = cv2.erode(segment, None, iterations=1)
        return segment

    def calculate_lines(self, frame):
        output = []
        for x1,y1,x2,y2 in self.lines:
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            slope = parameters[0] #斜率 
            if slope == 0:
                slope += 0.0001
            y_intercept = parameters[1] #截距
            output.append((slope,y_intercept))
        output = [self.calculate_coordinate(frame, parameters=line) for line in output]
        return np.array(output)

    def calculate_coordinate(self, frame,parameters):
        slope, y_intercept = parameters
        y1 = frame.shape[0]
        y2 = int(y1-150)
        x1 = int((y1-y_intercept)/slope)
        x2 = int((y2-y_intercept)/slope)
        return np.array([x1,y1,x2,y2])

    def visualize_lines(self, img):
        lines_visualize = np.zeros_like(img)
        if self.lines is not None:
            index = 0
            for x1,y1,x2,y2 in self.lines.copy():
                # if x2 < 0 or y2 < 0 or x1>img.shape[0]*2 or x2>img.shape[0]*2 or y1>img.shape[1]*2 or y2>img.shape[1]*2:
                #     self.lines = np.delete(self.lines, index, 0)
                #     continue
                try:
                    cv2.line(lines_visualize,(x1,y1),(x2,y2),(0,0,255),5)
                except:
                    self.lines = np.delete(self.lines, index, 0)
                    index -= 1
                index += 1
        lines_visualize = cv2.addWeighted(img,0.6,lines_visualize,1,0.1)
        return lines_visualize

    def getClassify(self, corr):
        res = ''
        for x1, y1, x2, y2 in self.lines:
            parameters = np.polyfit((x1,x2), (y1,y2), 1)
            res+=str(int(parameters[0] * corr[0] + parameters[1] - corr[1] >= 0))
        if res != "":
            res = int(res, 2)+1 / float(2^len(self.lines))
        else:
            res = 0
        return res

    def deleteRepeatLines(self):
        res = []
        for x1, y1, x2, y2 in self.lines:
            param1 = np.polyfit((x1,x2), (y1,y2), 1)
            for x11, y11, x12, y12 in res:
                param2 = np.polyfit((x11,x12), (y11,y12), 1)
                if abs(math.atan(param1[0]) - math.atan(param2[0])) < 0.5 and abs(param1[1]-param2[1]) < 50:
                    break
                elif abs(x12 - x1) < 5 and abs(y12 - y1):
                    break
                elif abs(x2 - x11) < 5 and abs(y2 - y11):
                    break
            else:
                res.append((x1, y1, x2, y2))
        return res

    def order_points(self, pts):
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]
        return np.array([tl, tr, br, bl], dtype="float32")


    def four_point_transform(self, img):
        warped = cv2.warpPerspective(img, self.transM, (self.maxWidth, self.maxHeight))
        return warped

    def two_point_transform(self, pt):
        dstx, dsty, dstr = self.transM.dot(np.array([pt[0], pt[1], 1]))
        return (dstx/dstr, dsty/dstr)

    def get_four_point(self, theda=60):
        sortLines = sorted(self.lines, key = lambda s: s[0])
        line_1 = sortLines[0]
        line_2 = sortLines[-1]
        if len(sortLines) > 1:
            line_3 = sortLines[1]
            self.roadWidth = abs(line_1[0] - line_3[0])
            print(self.roadWidth)
        rect = self.order_points(np.array([line_1[:2], line_1[2:], line_2[:2], line_2[2:]]))
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        minWidth = min(int(widthA), int(widthB))
        h = (bl[1] - tl[1]) / math.cos(theda / 180 * math.pi)
        maxHeight = (
            maxWidth * h * math.log(minWidth / maxWidth, math.e) / (minWidth - maxWidth)
        )
        maxHeight = int(maxHeight)

        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        return M, maxWidth, maxHeight


if __name__ == "__main__":
    frame = cv2.imread("background.jpg")
    back = BackgroundDetector(frame)
    output = back.visualize_lines(back.backgroundImg)
    print(back.lines)
    # back.testlines()
    while True:
        cv2.imshow("output", output)
        if cv2.waitKey(10)&0xff == ord('q'):
            break
    # while True:
    #     cv2.imshow("Road", back.four_point_transform(back.backgroundImg))
    #     if cv2.waitKey(10)&0xff == ord('q'):
    #         break

    cv2.destroyAllWindows()
