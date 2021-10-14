import cv2
import time

start = time.time()


for i in range(1000):
    bg_img = cv2.imread("./orii.bmp")

end = time.time()
print(end-start)
start = time.time()


for i in range(1000):
    bg_img = cv2.imread("./img.png")

end = time.time()
print(end-start)


start = time.time()


for i in range(1000):
    bg_img = cv2.imread("./orii.png")

end = time.time()
print(end-start)


cv2.imwrite('./test.png', bg_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
bg_2_img = cv2.imread("./test.png")

print((bg_img - bg_2_img).any())
