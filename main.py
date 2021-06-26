import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
segmentor = SelfiSegmentation()
fpsReader = cvzone.FPS()

listImg = os.listdir("images")
print(listImg)
imglist = []
for imgPath in listImg:
    img = cv2.imread(f'images/{imgPath}')
    imglist.append(img)
print(len(imglist))

indexImg = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img,imglist[indexImg],threshold=0.8)

    imgStacked = cvzone.stackImages([img,imgOut],2,1)
    _, imgStacked = fpsReader.update(imgStacked,color=(0,0,233))
    print(indexImg)
    cv2.imshow("image", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if indexImg>0:
            indexImg -=1
    elif key == ord('d'):
        if indexImg<len(imglist)-1:
            indexImg +=1
    elif key == ord('q'):
        break