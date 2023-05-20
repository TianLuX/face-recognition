import cv2
import dlib
import imutils
from imutils.face_utils import rect_to_bb
from faceAligner import FaceAligner
import os


# 初始化检测器对象
detector = dlib.get_frontal_face_detector()
# 实例化我们的面部标志预测器，使用shape_predictor_68_face_landmarks.dat
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

# 人脸居中处理
def align_once(img_path):
    img = cv2.imread(img_path)
    img = imutils.resize(img, width=800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    # 一张图片多个人脸
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(img[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(img, gray, rect)
        # cv2.imshow("Original", faceOrig)
        # cv2.imshow("Aligned", faceAligned)
        # cv2.waitKey(0)
        return faceAligned

#依次处理lfw文件夹中每张人脸，存储到lfw_align文件夹同样目录下
for filepath,dirnames,filenames in os.walk(r'E:\ai-code\deep-learning\face-recognition\lfw'):
    for filename in filenames:
        path1 = str(filepath) + str('\\') + str(filename)
        faceAligned = align_once(path1)
        path2 = str(filepath).replace('lfw','lfw_align') + str('\\') + str(filename)
        if faceAligned is not None:
            cv2.imwrite(path2, faceAligned)

