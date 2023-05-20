#将人脸旋转至水平，并居中放大
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
from imutils.face_utils.helpers import shape_to_np
import numpy as np
import cv2 as cv


class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.30, 0.30),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # 面部标志性预测器模型
        self.predictor = predictor
        # 指定所需的输出左眼位置
        self.desiredLeftEye = desiredLeftEye
        # 人脸宽度
        self.desiredFaceWidth = desiredFaceWidth
        # 人脸高度值
        self.desiredFaceHeight = desiredFaceHeight
        # 图片为正方形
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # image : RGB 输入图像
        # gray ：灰度输入图像
        # rect ：由 dlib 的 HOG 人脸检测器生成的边界框矩形
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        # 找到左右眼区域
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # 计算每只眼睛的质心
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

        # 计算左右眼之间角度
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # 计算右眼坐标，得到新结果的图像比例
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # 仿射变换准备步骤：找到眼睛之间的中点以及计算旋转矩阵并更新其平移分量
        eyesCenter = (
        int((leftEyeCenter[0] + rightEyeCenter[0]) // 2), int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))
        # 用于旋转和缩放面部的旋转矩阵
        # eyeCenter ：眼睛之间的中点是围绕面部旋转的点
        # angle：将面部旋转到的角度，确保眼睛位于同一水平线上
        # scale ：放大或缩小图像的百分比，确保图像缩放到所需的大小
        M = cv.getRotationMatrix2D(eyesCenter, angle, scale)

        # 更新矩阵的平移分量，使人脸在仿射变换后仍然在图像中
        # x 方向的平移:面宽的一半
        tX = self.desiredFaceWidth * 0.5
        # y 方向的平移:将所需的面部高度乘以所需的左眼y值
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # 应用仿射变换来对齐人脸
        # M ：平移、旋转和缩放矩阵
        # (self.desiredFaceWidth, self.desiredFaceHeight)输出面所需的宽度和高度
        output = cv.warpAffine(image, M, (self.desiredFaceWidth, self.desiredFaceHeight), flags=cv.INTER_CUBIC)

        return output
