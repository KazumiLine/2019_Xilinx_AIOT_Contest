from PIL import ImageGrab
import win32gui
import cv2
import numpy as np

hwnd = win32gui.FindWindow(None, 'Road')
l, t, r, b = win32gui.GetWindowRect(hwnd)
out = cv2.VideoWriter('test2.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (r-l-20,  b-t-40))
while True:
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    l += 10
    r -= 10
    t += 30
    b -= 10
    img = ImageGrab.grab(bbox=(l, t, r, b))
    cv2.imshow("A", np.array(img))
    out.write(np.array(img))
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break

cv2.destroyAllWindows()
