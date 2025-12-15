# NaPHI_structural-simulations_SAM
Official Segment Anything Model (SAM) workflow for 4D-STEM diffraction analysis and abTEM structural simulations of NaPHI. Code for the paper: "Elucidating Structural Disorder in a Polymeric Layered Material".

## 🚀 Usage & Prerequisites

This workflow relies on Meta's **Segment Anything Model (SAM)**. To run the analysis notebook (`SAM/SAM_analysis.ipynb`), you must first set up the model.

### 1. Download the Model Checkpoint
1.  Download the **ViT-H SAM model** (`sam_vit_h_4b8939.pth`) from the official repository:  
    👉 [https://github.com/facebookresearch/segment-anything#model-checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)
2.  Place the downloaded `.pth` file inside the `SAM/` folder of this repository (or update the path in the notebook accordingly).

### 2. Running the Analysis
The notebook demonstrates how to:
* Load raw 4D-STEM data (currently configured for `.prz` files).
* Convert 1-channel diffraction patterns into 3-channel inputs for SAM.
* Apply zero-shot segmentation to identify diffuse scattering and diffraction spots.
* Filter masks based on geometric parameters.

**Note:** The code uses the `vit_h` (huge) model by default for maximum accuracy, which requires a GPU with significant VRAM (tested on RTX 4080).

---

## 🧠 Methodology: SAM for Diffraction

This repository applies Meta’s Segment Anything (SAM) **off-the-shelf** (without fine-tuning) to segment complex diffraction features in 4D-STEM datasets.

**Data Processing Pipeline:**
1.  **Data Loading:** Raw 4D-STEM data is loaded from `.prz` files. Each diffraction pattern is extracted as a raw image without metadata.
2.  **Preprocessing:**
    * **Channel Conversion:** Grayscale diffraction patterns are converted to a 3-channel (RGB) representation to match SAM's input requirements via OpenCV (which also handles intensity rescaling).
    * **Noise Reduction:** A blurring step is applied to reduce high-frequency noise, followed by resizing (using `scikit-image`).
3.  **Segmentation:** Each preprocessed diffraction pattern is passed through the SAM `ViT-Huge` model to generate candidate masks.

**Filtering Strategy:**
To isolate relevant structural features (e.g., streaks, diffuse lines) from the background, masks are filtered based on geometric properties:
* **Area:** Excludes masks that are too small (noise) or too large (background).
* **Aspect Ratio:** Selects for elongated features (diffuse streaks).
* **Linearity (R²):** Ensures masks correspond to well-defined linear features.
* **Position:** Removes central beam artifacts and distant noise based on distance from the image center.
* **Line Length:** Prioritizes extended features over small blobs.

**Limitations & False Negatives:**
Two primary types of false negatives were observed:
1.  **Over-blurring:** Merged distinct line features with central-beam noise into a single large disk-like mask.
2.  **Under-blurring:** Caused line features to fragment into small patches, leading to missed detections during sorting.

*Suggestions for addressing these false negatives are included in the code comments.*


## ⚛️ Structural Simulations using abTEM

The structural modeling and diffraction simulations were performed using the following open-source Python packages:

* **Structure Generation:** [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/)
* **Diffraction Simulation:** [abTEM](https://github.com/abTEM/abTEM)

**Workflow:**
1.  **Model Construction:** A monolayer of NaPHI was extracted using ASE. Geometric wave-like modifications were applied to the atomic coordinates to mimic the structural disorder observed experimentally.
2.  **Simulation:** The modified structures were used as input for `abTEM` to generate simulated electron diffraction patterns for comparison with the experimental 4D-STEM data.

---

## 📜 Citation

If you use this code or workflow in your research, please cite the following paper:

> **Elucidating Structural Disorder in a Polymeric Layered Material: The Case of Sodium Poly(heptazine imide) Photocatalyst** > Daniel Khaykelson, Gabriel A. A. Diab, Sidney R. Cohen, Tamar Kashti, Tatyana Bendikov, Iddo Pinkas, Ivo F. Teixeira, Nadezda V. Tarakina, Lothar Houben, and Boris Rybtchinski.  
> *Nano Letters* **2025**, 25, 49, 17230–17236  
> DOI: [10.1021/acs.nanolett.5c04946](https://pubs.acs.org/doi/10.1021/acs.nanolett.5c04946)

### BibTeX
```bibtex
@article{Khaykelson2025,
  title = {Elucidating Structural Disorder in a Polymeric Layered Material: The Case of Sodium Poly(heptazine imide) Photocatalyst},
  author = {Khaykelson, Daniel and Diab, Gabriel A. A. and Cohen, Sidney R. and Kashti, Tamar and Bendikov, Tatyana and Pinkas, Iddo and Teixeira, Ivo F. and Tarakina, Nadezda V. and Houben, Lothar and Rybtchinski, Boris},
  journal = {Nano Letters},
  year = {2025},
  volume = {25},
  number = {49},
  pages = {17230--17236},
  doi = {10.1021/acs.nanolett.5c04946},
  url = {[https://pubs.acs.org/doi/10.1021/acs.nanolett.5c04946](https://pubs.acs.org/doi/10.1021/acs.nanolett.5c04946)}
}
