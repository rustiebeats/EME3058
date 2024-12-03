import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import threading
import ctypes
import inspect
from time import sleep
from jetbot import Robot, Camera, bgr8_to_jpeg, ObjectDetector
from deep_sort_realtime.deepsort_tracker import DeepSort

robot = Robot()
camera = Camera.instance(width=300, height=300)

model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('../collision_avoidance/best_model.pth'))
collision_model = collision_model.to(torch.device('cuda'))

# https://github.com/levan92/deep_sort_realtime
# Initialize DEEPSORT tracker
tracker = DeepSort(max_age=30, nn_budget=100, nms_max_overlap=1.0, max_iou_distance=0.7, embedder="torchreid")

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

target_person_id = None

def preprocess(camera_value):
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    return x.unsqueeze(0).to(torch.device('cuda'))

def detection_center(bbox):
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return center_x, center_y

def execute(image):
    global robot, tracker, target_person_id
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])

    if prob_blocked > 0.7:
        robot.left(0.5) # ??
        return

    detections = model(image)
    detection_bboxes, detection_scores, detection_classes = [], [], []
    for det in detections[0]:
        bbox = det['bbox']
        detection_bboxes.append([
            int(bbox[0] * 300), int(bbox[1] * 300), int(bbox[2] * 300), int(bbox[3] * 300)
        ])
        detection_scores.append(det['confidence'])
        detection_classes.append(det['label'])

    tracks = tracker.update_tracks(detection_bboxes, detection_scores, image, detection_classes)
    
    track_classes = {}  # Track ID -> class label
    for track, cls in zip(tracks, detection_classes):
        if track.is_confirmed():
            track_classes[track.track_id] = cls
    
    if target_person_id is None:
        for track in tracks:
            if track.is_confirmed() and track.track_id in track_classes:
                if track_classes[track.track_id] == 0:  # '0' is the COCO label for person
                    target_person_id = track.track_id
                    print(f"Target person ID: {target_person_id}")
                    break

    target_found = False
    for track in tracks:
        if track.track_id == target_person_id and track.is_confirmed():
            target_found = True
            bbox = track.to_ltrb()
            center_x, _ = detection_center(bbox)
            robot.set_motors(
                float(0.4 + 0.8 * center_x),
                float(0.4 - 0.8 * center_x)
            )
            break

def stop_thread(thread):
    tid = ctypes.c_long(thread.ident)
    if not inspect.isclass(SystemExit):
        exctype = SystemExit
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("Thread termination failed")

def detection_loop():
    global camera
    while True:
        image = camera.value
        execute(image)

thread1 = threading.Thread(target=detection_loop)
thread1.daemon = True
thread1.start()

try:
    while True:
        pass
except KeyboardInterrupt:
    stop_thread(thread1)
    camera.unobserve_all()
    robot.stop()
    print("Program terminated.")
