import os
import time
import random
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import cv2 as cv


model_path = 'models/land-seg.pt'


images_path = Path('dataset/landslide_dataset_nicholas/images')

all_images = sorted(images_path.glob("*.jpg"))
# selected_images = random.sample(all_images, min(100, len(all_images)))
selected_images = all_images[:100]
print(f'Tamanho = {len(all_images)}')

model = YOLO(model_path)


inference_times = []

results = model(selected_images[0]) 
img = results[0].plot()
cv.imwrite('outputs/out1.jpg', img)
start = time.time()
for i, img_path in enumerate(selected_images[1:]):
    img = Image.open(img_path)
    results = model(img)
    inference_times.append(results[0].speed['inference'])
end = time.time()
duration = end - start
print(f'Duration = {duration} segundos')
# Estatísticas
avg_time_medido = (sum(inference_times))/len(inference_times)
avg_time = (duration/(len(selected_images)-1))*1000
print(f"\n✅ Média de tempo REAL por inferência (n={len(selected_images)-1}): {avg_time:.4f} ms\n")
print(f"\n✅ Média de tempo APENAS DA INFERENCIA (n={len(selected_images)-1}): {avg_time_medido:.4f} ms")
