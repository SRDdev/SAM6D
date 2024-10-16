import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import json
import os
from ultralytics import SAM

class InstanceSegmentation:
    def __init__(self, model_path="sam2_b.pt", device="cuda"):
        print(f"Initializing SAM2 model with checkpoint: {model_path}")
        self.device = device
        self.model = SAM(model_path)
        self.model.to(device=self.device)
        print("SAM2 model initialized and moved to device.")

    def set_image(self, image_path):
        print(f"Loading image from path: {image_path}")
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print("Image loaded.")

    def show_mask(self, mask, ax=None, random_color=False):
        print("Showing mask.")
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Convert the mask image to 8-bit format for saving
        mask_image_8bit = (mask_image[:, :, :3] * 255).astype(np.uint8)
        
        # Save the image using cv2.imwrite
        cv2.imwrite("Data/Example/outputs/sam6d_results/mask.png", mask_image_8bit)
        print("Mask saved to Data/Example/outputs/sam6d_results/mask.png")

    @staticmethod
    def mask_to_rle(binary_mask):
        print("Converting mask to RLE.")
        rle = {"counts": [], "size": list(binary_mask.shape)}
        counts = rle.get("counts")

        last_elem = 0
        running_length = 0

        for i, elem in enumerate(binary_mask.ravel(order="F")):
            if elem == last_elem:
                pass
            else:
                counts.append(running_length)
                running_length = 0
                last_elem = elem
            running_length += 1

        counts.append(running_length)

        return rle

    @staticmethod
    def create_detection_json(scene_id, image_id, category_id, bbox, scores, time, masks, image_size):
        print("Creating detection JSON.")
        detections = []
        for i in range(len(masks)):
            encoded_mask = InstanceSegmentation.mask_to_rle(masks[i])
            detection = {
                "scene_id": scene_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": float(scores[i]),
                "time": time,
                "segmentation": {
                    "counts": encoded_mask["counts"],
                    "size": image_size
                }
            }
            detections.append(detection)
        return json.dumps(detections, indent=4)

    def infer(self, x, y, output_path="Data/Example/outputs/sam6d_results/"):
        print(f"Running inference with point ({x}, {y}).")
        results = self.model(self.image_path, points=[x, y], labels=[1])
        
        # Extract mask from results
        mask = results[0].masks.data[0].cpu().numpy()
        
        self.show_mask(mask)

        # Example input
        scene_id = 0
        image_id = 0
        category_id = 1
        bbox = results[0].boxes.xyxy[0].tolist()  # Get bounding box from results
        scores = results[0].boxes.conf.tolist()  # Get confidence scores from results
        time = 0.0
        image_size = [self.image.shape[0], self.image.shape[1]]  # [height, width]

        # Creating JSON
        detection_json = self.create_detection_json(scene_id, image_id, category_id, bbox, scores, time, [mask], image_size)

        json_output_path = f"{output_path}/detection_ism.json"
        with open(json_output_path, "w") as json_file:
            json_file.write(detection_json)
        print(f"Detection JSON saved to {json_output_path}")

# Example usage
if __name__ == "__main__":
    instance_segmentation = InstanceSegmentation(model_path="sam2_b.pt")
    instance_segmentation.set_image('/home/sai/SAM-6D/SAM-6D/aligned_color_click2.png')
    instance_segmentation.infer(x=223,y=366)