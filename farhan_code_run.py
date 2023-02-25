import cv2
import math
import numpy as np
def decodeBoundingBoxes(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if (score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0], sinA * w + offset[1])
            center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
            detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

boxlist=[]
net = cv2.dnn.readNet("frozen_east_text_detection.pb")   #This is the model we get after extraction
frame = cv2.imread("ZoomedOut.png")
inpWidth = inpHeight = 1920 # A default dimension
# Preparing a blob to pass the image through the neural network
# Subtracting mean values used while training the model.
image_blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

output_layer = []
output_layer.append("feature_fusion/Conv_7/Sigmoid")
output_layer.append("feature_fusion/concat_3")

net.setInput(image_blob)
output = net.forward(output_layer)
scores = output[0]
geometry = output[1]



confThreshold = 0.5
nmsThreshold = 0.3
[boxes, confidences] = decodeBoundingBoxes(scores, geometry, confThreshold) #was previously named decode()
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)

height_ = frame.shape[0]
width_ = frame.shape[1]
rW = width_ / float(inpWidth)
rH = height_ / float(inpHeight)

for i in indices:
    # get 4 corners of the rotated rect
    vertices = cv2.boxPoints(boxes[i[0]])#i[0]
    # scale the bounding box coordinates based on the respective ratios
    for j in range(4):
        vertices[j][0] *= rW
        vertices[j][1] *= rH
    square=[]
    for j in range(4):
        p1 = (vertices[j][0], vertices[j][1])
        p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
        points=[p1,p2]
        square.append(points)
    boxlist.append([
    min([square[0][0][0],square[0][1][0],square[1][0][0],square[3][1][0]]),
    min([square[0][1][1],square[1][0][1],square[1][1][1],square[2][0][1]]),
    max([square[1][1][0],square[2][0][0],square[2][1][0],square[3][0][0]]),
    max([square[0][0][1],square[2][1][1],square[3][0][1],square[3][1][1]])
    ])
boxlist=np.array(boxlist)
# To save the image:
# cv2.imwrite("maggi_boxed.jpg", frame)
# frame = cv2.imread("ZoomedOut.png")
sorted=boxlist[boxlist[:,1].argsort()]
# print(boxlist[0,:])
j=0
def convert(list):
    return (*list, )
for i in sorted:
    cv2.line(frame, convert(list(map(int,i[0:2]))), convert(list(map(int,i[2:]))), (0, 255, 0), 3)
    # print(i[0:2],i[2:])
    j+=1
cv2.imwrite("maggi_boxed.jpg", frame)