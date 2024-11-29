# Object Follow-Avoid with DEEPSORT Tracking
# ------------------------------------------
# @Copyright (C): 2010-2019, Shenzhen Yahboom Tech
# @Author: Malloy.Yuan
# @Date: 2019-07-17 10:10:02
# @LastEditors: Malloy.Yuan
# @LastEditTime: 2019-09-17 17:54:19

import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
import threading
import ctypes
import inspect
from jetbot import Robot, Camera, bgr8_to_jpeg
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize robot and camera
robot = Robot()
camera = Camera.instance(width=300, height=300)

# Load pre-trained collision avoidance model
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('../collision_avoidance/best_model.pth'))
collision_model = collision_model.to(torch.device('cuda'))

# Initialize DEEPSORT tracker
tracker = DeepSort(max_age=30, nn_budget=100, nms_max_overlap=1.0, max_iou_distance=0.7)

# Preprocessing for collision detection
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    return x.unsqueeze(0).to(torch.device('cuda'))

# Detect the center of a bounding box
def detection_center(bbox):
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return center_x, center_y

# Detection loop
def execute(image):
    global robot, tracker
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])

    # Collision handling
    if prob_blocked > 0.7:
        robot.left(0.5)
        return

    # Detect objects using model
    detections = model(image)
    detection_bboxes, detection_scores, detection_classes = [], [], []
    for det in detections[0]:
        bbox = det['bbox']
        detection_bboxes.append([
            int(bbox[0] * 300), int(bbox[1] * 300), int(bbox[2] * 300), int(bbox[3] * 300)
        ])
        detection_scores.append(det['confidence'])
        detection_classes.append(det['label'])

    # Update DEEPSORT tracker
    tracks = tracker.update_tracks(detection_bboxes, detection_scores, image, detection_classes)

    # Draw tracks and control robot
    if tracks:
        target = tracks[0]  # Use the first valid track
        if target.is_confirmed():
            bbox = target.to_ltrb()
            center_x, _ = detection_center(bbox)
            robot.set_motors(
                float(0.4 + 0.8 * center_x),
                float(0.4 - 0.8 * center_x)
            )
        else:
            robot.forward(0.4)
    else:
        robot.forward(0.4)

# Multithreading support
def stop_thread(thread):
    tid = ctypes.c_long(thread.ident)
    if not inspect.isclass(SystemExit):
        exctype = SystemExit
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("Thread termination failed")

# Continuous execution loop
def detection_loop():
    global camera
    while True:
        image = camera.value
        execute(image)

# Start detection thread
thread1 = threading.Thread(target=detection_loop)
thread1.daemon = True
thread1.start()

# Cleanup on exit
try:
    while True:
        pass
except KeyboardInterrupt:
    stop_thread(thread1)
    camera.unobserve_all()
    robot.stop()
    print("Program terminated.")
