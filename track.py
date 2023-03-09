import yolov5
import cv2
from sort import *

def load_classes(classes_path="config\coco_classes.txt"):
        classesFile = classes_path
        classes = None
        with open(classesFile, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        return classes

def check_crossed(line, center):
    start, end = line["start"], line["end"]
    buffer = 20
    rect_start = (min(start[0], end[0]) - buffer, min(start[1], end[1]) - buffer)
    rect_end = (max(start[0], end[0]) + buffer, max(start[1], end[1]) + buffer)
    if rect_start[0] <= center[0] <= rect_end[0] and rect_start[1] <= center[1] <= rect_end[1]:
        return True
    return False
# Set up colors and font
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 128, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def text(frame, label="null", position=(0,0), font=cv2.FONT_HERSHEY_COMPLEX, scale=0.5, color=BLUE, thickness=2):
    cv2.putText(frame, label, (position), font, scale, color, thickness)  


# Set up crossing lines
lines = [
    {"start": (150, 400), "end": (500, 400), "color": RED, "name": "Line 1"},
    {"start": (550, 400), "end": (700, 400), "color": RED, "name": "Line 2"},
    {"start": (890, 500), "end": (1150, 540), "color": RED, "name": "Line 3"}
    ]
crossed = {line["name"]:set() for line in lines}


     
classes = load_classes()
tracker = Sort()
# load pretrained model
model = yolov5.load('weights\yolov5s.pt', device=0)
#model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights\yolov5s.pt')  
model.float()
model.eval()
# set model parameters
model.conf = 0.35  # NMS confidence threshold
model.iou = 0.75  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# # perform inference
# results = model(img)

# # inference with larger input size
# results = model(img, size=1280)

# # inference with test time augmentation
# results = model(img, augment=True)
# parse results
    # predictions = results.pred[0]
    # boxes = predictions[:, :4] # x1, y1, x2, y2
    # scores = predictions[:, 4]
    # categories = predictions[:, 5]

capture = cv2.VideoCapture(r"data\road_traffic.mp4")
output = cv2.VideoWriter(f'data/vehicles_counting.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (int(capture.get(3)), int(capture.get(4))))


while True:
    _, frame = capture.read()
    if frame is None:
        print("End of stream")
        break
    
    results = model(frame)
    detections = results.pred[0].to("cpu").numpy()
    tracks = tracker.update(detections)
    # detections[:, -1] = detections[:, -1][::-1]
    # class_ids = detections[:, -1].reshape(-1, 1)
    # clss = class_ids.squeeze()
    # # Concatenate class ids with the rest of the columns in `tracks`
    # tracks = np.hstack((tracks, class_ids))
    	
    for j in range (len(tracks.tolist())):
        coords = tracks.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        center = ((x1 + x2) // 2,(y1 + y2) // 2)

        # class_id = int(clss[j])
        # class_name = classes[class_id]
        name_idx = int(coords[4])
        id = f'ID {str(name_idx)}'
        color = BLUE

        cv2.rectangle(frame,(x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (center), radius=5, color=(0, 0, 255), thickness=-1)
        text(frame, id, (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
        # cv2.putText(frame, class_name, (x1+50, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

        for line in lines:
             
            start, end, color, name = line.values()
            cv2.line(frame, start, end, color, 3)
            check = check_crossed(line, center)
            if check: 
                crossed[name].add(id)
                
        labels = [f'{name}: {len(crossed[name])}' for name in ['Line 1','Line 2','Line 3']]
        text(frame, labels[0], (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, BLACK, 2)
        text(frame, labels[1], (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, BLACK, 2)
        text(frame, labels[2], (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.7, BLACK, 2)
        cv2.imshow('Output', frame)

        # # Check if vehicle has crossed any line
        # for line in enumerate(lines):
        #     start, end, color, name = line.values()
        #     cv2.line(frame, start, end, color, 3)

        #     # Calculate distance between center of circle and line
        #     distance = abs((end[1] - start[1]) * center_x - (end[0] - start[0]) * center_y + end[0] * start[1] - end[1] * start[0]) / np.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)

        #     # Check if distance is less than a threshold
        #     if distance < 30:
        #         # Add line to set of crossed lines for this vehicle
        #         vehicle_lines[name_idx].add(name)

        #     cv2.putText(frame, len(vehicle_lines[name_idx]), (x1+50, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

                
    #output.write(frame)
    

    # # show detection bounding boxes on image
    # #results.show()
    output.write(frame)
    # cv2.imwrite("data/img.png", frame)
    # break

    if cv2.waitKey(1) == ord('q'):
        print("finished by user")
        break