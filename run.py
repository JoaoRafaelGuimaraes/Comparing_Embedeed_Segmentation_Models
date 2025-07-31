import os
import time
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2 as cv
import numpy as np


model_path = 'models/land-seg.engine'
model = YOLO(model_path)


images_path = Path('dataset/landslide_dataset_nicholas/images')
all_images = sorted(images_path.glob("*.jpg"))
selected_images = all_images[:10]
print(f'Total images found: {len(all_images)}')


def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def load_ground_truth_boxes(label_path, img_width, img_height):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 3 or len(parts) % 2 == 0:
                continue  

            coords = list(map(float, parts[1:]))  
            x_coords = coords[::2]
            y_coords = coords[1::2]

           
            x_min = min(x_coords) * img_width
            y_min = min(y_coords) * img_height
            x_max = max(x_coords) * img_width
            y_max = max(y_coords) * img_height

            boxes.append([x_min, y_min, x_max, y_max])
    return boxes


inference_times = []
total_iou = 0
count = 0


results = model(selected_images[0])
img = results[0].plot()
cv.imwrite('outputs/out1.jpg', img)

start = time.time()
for i, img_path in enumerate(selected_images[1:]):
    img = Image.open(img_path)
    width, height = img.size
    results = model(img)
    
    inference_times.append(results[0].speed['inference'])

    
    label_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt')
    gt_boxes = load_ground_truth_boxes(label_path, width, height)
    
    tensor = results[0].boxes.xyxy
    print(tensor.device)
    pred_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else [] # Transfere para a CPU para usar numpy -> Pode ser melhorado para usar torch direto

    for gt_box in gt_boxes:
        best_iou = 0
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box)
            if iou > best_iou:
                best_iou = iou
        total_iou += best_iou
        count += 1

end = time.time()
duration = end - start

# Averages
avg_time_real = (duration / (len(selected_images) - 1)) * 1000
avg_time_model = sum(inference_times) / len(inference_times)
mean_iou = total_iou / count if count > 0 else 0

# Report
print(f'\nNúmero de Amostras n = {len(selected_images)}')
print(f"Tempo total: {duration:.2f} segundos")
print(f"✅ Média de tempo REAL por inferência: {avg_time_real:.4f} ms")
print(f"✅ Média de tempo APENAS DA INFERÊNCIA: {avg_time_model:.4f} ms")
print(f" Média de IoU sobre {count} bounding boxes: {mean_iou:.4f}")
