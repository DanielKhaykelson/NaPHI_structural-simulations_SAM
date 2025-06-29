# sam_image_utils.py (REWRITTEN)
import numpy as np
import cv2
from tqdm import tqdm
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage import exposure
import torch
import math

class SamMaskProcessor:
    def __init__(self, mask_generator, device="cuda"):
        self.mask_generator = mask_generator
        self.device = device
        print(f"[INFO] SAM Mask Processor using device: {self.device}")

    def preprocess_image(self, image):
        image_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        blurred_image = gaussian(image_3ch, sigma=4, preserve_range=False)
        image_rescaled = exposure.rescale_intensity(blurred_image, in_range=(0, 0.6))
        rescaled_image = rescale(image_rescaled, 1/2, channel_axis=2)
        return rescaled_image

    def make_mask(self, rescaled_image):
        masks = self.mask_generator.generate(rescaled_image)
        sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        return sorted_masks

    def get_masks_list(self, image_stack):
        masks_list = []
        for image in tqdm(image_stack, desc="Processing images"):
            rescaled_image = self.preprocess_image(image)
            sorted_masks = self.make_mask(rescaled_image)
            masks_list.append(sorted_masks)
        return np.array(masks_list, dtype=object)

    def filter_masks(self, masks, area_range=(50, 1000), aspect_ratio_threshold=1.2,
                     min_length=20, min_distance=30, max_distance=130, min_r2=0.5):
        filtered = []

        for mask in masks[1:10]:  # Skip the largest one
            area = mask['area']
            if not (area_range[0] <= area <= area_range[1]):
                continue

            width, height = mask['bbox'][2], mask['bbox'][3]
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < aspect_ratio_threshold or max(width, height) < min_length:
                continue

            img = mask['segmentation'].astype(int)
            y, x = np.nonzero(img)
            if len(x) < 2 or len(y) < 2:
                continue

            try:
                line_fit = np.polyfit(x, y, 1)
                y_pred = np.poly1d(line_fit)(x)
            except np.linalg.LinAlgError:
                continue

            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res / ss_tot #if ss_tot != 0 else 0
            if r2 < min_r2:
                continue

            x_min_px, x_max_px = np.min(x), np.max(x)
            mid_x = (x_min_px + x_max_px) / 2
            mid_y = np.poly1d(line_fit)(mid_x)
            xc, yc = img.shape[1] / 2, img.shape[0] / 2

            delta_x = mid_x - xc
            delta_y = img.shape[1] - mid_y - yc  # ← your convention
            distance = np.sqrt(delta_x**2 + delta_y**2)
            # Use bounding box center like in line_angles_mask
            # x_min, y_min, width, height = mask['bbox']
            # x_mid = x_min + width / 2
            # y_mid = y_min + height / 2
            # p = [x_mid, y_mid]
            # q = [img.shape[0]/2, img.shape[1]/2]
            # distance = math.dist(p, q)
            if not (min_distance <= distance <= max_distance):
                continue

            angle = math.degrees(math.atan2(delta_y, delta_x)) % 180

            # Save metadata
            mask['angle'] = angle
            mask['distance'] = distance
            mask['midpoint'] = (mid_x, mid_y)
            filtered.append(mask)

        return filtered

    def load_prz_data(self, path):
        sig = np.load(path, allow_pickle=True)
        img_stack = sig['data']
        img_stack = np.reshape(img_stack, (img_stack.shape[0] * img_stack.shape[1], img_stack.shape[2], img_stack.shape[3]))
        return img_stack
