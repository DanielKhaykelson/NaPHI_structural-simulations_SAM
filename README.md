# NaPHI_structural-simulations_SAM
This repo contains the code used in the NaPHI structural elucidation paper. The first part is the structural analysis and abTEM simulation of the wave-like deformation. The second part is the Segment Anything analysis of the 4DSTEM dataset.

## SAM
Meta’s Segment Anything (SAM)2 was used off-the-shelf without fine-tuning, employing the ViT-Huge model. Raw 4D-STEM data were uploaded as a .prz file, and each raw image was extracted without metadata. Images were originally grayscale and were transformed to a 3-channel representation to work with SAM via OpenCV3, which also automatically rescaled image intensity. A blurring step was performed to reduce noise, followed by image resizing, both using the scikit-image library. Each image was analyzed via SAM. Specific masks of interest were selected by filtering based on shape and position parameters of the mask boxes, using criteria such as:
•	Mask area range, to exclude very small or very large masks;
•	Aspect ratio threshold, to select elongated features by requiring masks to exceed a minimum aspect ratio;
•	Minimum line length, to reject small blobs and prioritize extended line features;
•	Minimum and maximum distance from the image center, to remove central beam artifacts and distant noise;
•	Minimum linear fit R², to ensure masks correspond to well-defined linear features.
An example of image outputs at each processing step is detailed in Figure S3. All calculations were performed on a single NVIDIA RTX4080 GPU. Two false negatives were observed during analysis: (1) overly strong blurring merged line features with central-beam noise into a single large disk-like mask, and (2) insufficient blurring caused line features to fragment into small patches, leading to missed detections during sorting. Suggestions for addressing these false negatives are included in the code but were not applied in this work, as they affect only the size—not the nature—of the line domains. 


## Structural Simulations using abTEM
