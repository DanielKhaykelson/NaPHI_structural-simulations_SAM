
import numpy as np
import cv2
import torch
from skimage.filters import gaussian
from skimage.transform import rescale
from skimage import exposure
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class SamMaskProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="./sam_vit_h_4b8939.pth", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)

    def load_prz_data(self, path):
        sig = np.load(path, allow_pickle=True)
        img_stack = sig['data']
        img_stack = np.reshape(img_stack, (img_stack.shape[0] * img_stack.shape[1], img_stack.shape[2], img_stack.shape[3]))
        return img_stack

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

    def filter_masks(self, masks, area_range=(50, 1000), aspect_ratio_threshold=1.2, min_length=20):
        filtered = []
        for line_mask in masks[1:10]:
            area = line_mask['area']
            if area < area_range[0] or area > area_range[1]:
                continue
            width, height = line_mask['bbox'][2], line_mask['bbox'][3]
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio < aspect_ratio_threshold:
                continue
            if max(width, height) < min_length:
                continue
            filtered.append(line_mask)
        return filtered
