import cv2 as cv
import numpy as np
import datetime

# Load Yolo Algorithm
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Use GPU instead (like 5 fps increase LOL -- but hey better than nothing) 
# the build tutorial for opencv https://www.youtube.com/watch?v=YsmhKar8oOc 
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# loads the text in the coco.names file into an array
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# you can print what you get
# getLayerNames(): Get the name of all layers of the network.
# getUnconnectedOutLayers(): Get the index of the output layers.
layer_names = net.getLayerNames()
print(layer_names)


output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

for i in net.getUnconnectedOutLayers():
    print(layer_names[i[0] - 1])

print(print(layer_names[i[0] - 1]))


# random.uniform(low=0.0, high=1.0, size=None (int or tuples))
colors = np.random.uniform(0, 255, size = (len(classes), 3))
# Load Webcam Video in 480P
cap = cv.VideoCapture(0)

fps_start = datetime.datetime.now()

fps = 0
total_frames = 0


while(True):
    _, frame =  cap.read()
    
    # code below gets the frames per second
    total_frames = total_frames + 1

    fps_end = datetime.datetime.now()

    time_diff = fps_end - fps_start

    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)
    
    # 2 decimal points e.g 33.33
    fps_text = "FPS: {:.2f}".format(fps)


    # code below is for implementing YOLO algorithm    
    height, width, channels = frame.shape

    # Detecting Objects
    blob = cv.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # show each blob, each blob represents each channel of RGB, where we said True we invert it from BGR to RGB ^


    # we need blobs for You Only Look Once algorithm
    # once we have the blob the images are ready to be processed by the YOLO algorithm
    
    # shows 3 grayscaled images that represent each channel of the RGB color spectrum
    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv.imshow(str(n), img_blob)


    net.setInput(blob)
    outputs = net.forward(output_layers)
    # print(outputs)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                # top left coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # number_objects_detected = len(boxes)
    # take only these indexes, we set a threshhold that will remove extra boxes or noise in the image. If there is more than 1 box it's most likely just 1 object 
    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # the indexes that are printed represent only the indexes that we want to display
    # print(indexes)
    font = cv.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        if i in indexes:
            
            label = classes[class_ids[i]]
            
            # display confidence %
            confidence = "{:.2f}%".format(confidences[i] * 100)
          
            color = colors[i]
            cv.rectangle(frame, (x,y), (x + w, y  + h), (0, 255, 0), 2)
            
            # makes the text get put into a white box 
            (label_width, label_height), baseline = cv.getTextSize(f"{label} {confidence}", font, 2, 1)
            cv.rectangle(frame, (x,y), (x + label_width, y + label_height + 10), (255, 255, 255), cv.FILLED)
            cv.putText(frame, f"{label} {confidence}", (x, y + label_height + 5), font, 2, (0,0,0), 2)
            
            # print(label)

    cv.putText(frame, fps_text, (5, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)

    cv.imshow("frame", frame)
    # press q to quit
    if cv.waitKey(20) & 0XFF == ord('q'):
            break
# When everything is done, release 
cap.release()
cv.destroyAllWindows()