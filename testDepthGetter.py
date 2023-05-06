
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import requests
import torch
import numpy as np
import cv2

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
pixel_values = feature_extractor(image, return_tensors="pt").pixel_values

with torch.no_grad():
  outputs = model(pixel_values)
  predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
              ).squeeze()
output = prediction.cpu().numpy()
formatted = (output * 255 / np.max(output)).astype('uint8')
depth = Image.fromarray(formatted)

while(True):
    cv2.imshow("depth",formatted)
    cv2.waitKey(1)
