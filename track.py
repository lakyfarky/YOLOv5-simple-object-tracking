import yolov5
import cv2
from sort import *
import torch
BLUE   = (255,178,50)
tracker = Sort()
# load pretrained model
model = yolov5.load('weights\yolov5s.pt', device="cpu")
#model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights\yolov5s.pt')  
model.float()
model.eval()
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# # perform inference
# results = model(img)

# # inference with larger input size
# results = model(img, size=1280)

# # inference with test time augmentation
# results = model(img, augment=True)


capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    if frame is None:
        print("End of stream")
        break

    results = model(frame)
    detections = results.pred[0].numpy()
    tracks = tracker.update(detections)
    for j in range (len(tracks.tolist())):
        coords = tracks.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        name_idx = int(coords[4])
        name = f'ID {str(name_idx)}'
        color = BLUE
        cv2.rectangle(frame,(x1, x2), (y1, y2), color, 2)
        cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 2)
        cv2.imshow('Output', frame)
        cv2.waitKey(5)
    
    # # parse results
    # predictions = results.pred[0]
    # boxes = predictions[:, :4] # x1, y1, x2, y2
    # scores = predictions[:, 4]
    # categories = predictions[:, 5]

    # # show detection bounding boxes on image
    # #results.show()
    # #output.write(img)
    # cv2.imshow('Output', frame)
    # cv2.waitKey(5)

    if cv2.waitKey(1) > -1:
        print("finished by user")
        break