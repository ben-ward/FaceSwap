import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

print "Press T to draw the keypoints and the 3D model"
print "Press R to start recording to a video file"

#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
predictor_path = "../shape_predictor_68_face_landmarks.dat"
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 380

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("../candide.npz")

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None
# cap.set(3,640)
# cap.set(4,360)
# cap.set(3,1280)
# cap.set(4,720)
# cap.set(11, 99)
# cap.set(5, 3)
# cap.set(15, 0.1)
cameraImg = cap.read()[1]


import time
duration = 10
timeout = time.time() + duration

images = ["1.jpg","2.jpg","3.jpg","4.jpg","6.jpg","7.jpg","8.jpg","9.png"]

count = 0
renderer = FaceRendering.FaceRenderer(cameraImg, mesh)
while count < len(images):
    image_name = "../data/" + images[count]
    textureImg = cv2.imread(image_name)
    textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
    renderer.useImage(cameraImg, textureImg, textureCoords)
    count += 1

    cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_NORMAL)

    while True:
        if time.time() > timeout:
            timeout = time.time() + duration
            # renderedImg = renderer.destroy
            if count >= len(images):
                count = 0
            break


        cameraImg = cap.read()[1]
        # cameraImg = cv2.resize(cameraImg, (1024, 640))
        shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

        if shapes2D is not None:
            for shape2D in shapes2D:
                #3D model parameter initialization
                modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

                #3D model parameter optimization
                modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

                #rendering the model to an image
                shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
                renderedImg = renderer.render(shape3D)

                #blending of the rendered face with the image
                mask = np.copy(renderedImg[:, :, 0])
                renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
                cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)


                #drawing of the mesh and keypoints
                if drawOverlay:
                    drawPoints(cameraImg, shape2D.T)
                    drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

        if writer is not None:
            writer.write(cameraImg)

        cameraImg = cv2.resize(cameraImg, (1280, 720))
        cv2.imshow("window", cameraImg)
        # cv2.resizeWindow('window', 1400,1400)
        # cv2.imshow('image', cameraImg)
        key = cv2.waitKey(1)

        if key == ord('z'):
            if count == 1:
                count = len(images) - 1
            else:
                count -= 2
            break
        if key == ord('x'):
            if count >= len(images):
                count = 0
            break
        if key == 27:
            break
        if key == ord('t'):
            drawOverlay = not drawOverlay
        if key == ord('r'):
            if writer is None:
                print "Starting video writer"
                writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

                if writer.isOpened():
                    print "Writer succesfully opened"
                else:
                    writer = None
                    print "Writer opening failed"
            else:
                print "Stopping video writer"
                writer.release()
                writer = None
