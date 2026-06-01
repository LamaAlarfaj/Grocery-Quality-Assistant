# 🍎 Grocery Quality Assistant

> An automated crop quality assessment system combining **YOLOv8 transfer learning** with classical image processing techniques to detect, classify, and grade 32 types of fruits and vegetables.


## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Implementation Environment](#implementation-environment)
- [Getting Started](#getting-started)
- [Quality Grading System](#quality-grading-system)
- [Future Work](#future-work)


## Overview

Manual quality inspection of fruits and vegetables is subjective, time-consuming, and inconsistent. This project presents a **Grocery Quality Assistant** — an end-to-end pipeline that automatically identifies crop types and evaluates their quality into three grades: **Excellent**, **Acceptable**, and **Poor**.

The system combines a fine-tuned **YOLOv8l-cls** classification model with multiple image processing functions (color uniformity, surface defect detection, texture analysis, and sprouting detection) to deliver reliable, generalizable quality assessment across diverse crop types.


## Problem Statement

- Manual produce inspection is subjective and inconsistent.
- Most existing research focuses only on **binary ripeness classification** (ripe vs. stale).
- Multi-class ripeness studies are limited to a small number of crop types and require large labeled datasets.
- Image-processing approaches are typically designed for a **single crop type**, limiting generalization.
- There is a clear need for a **general, automated, and scalable** crop quality assessment system.


## Proposed Solution

- Fine-tune a **YOLOv8l-cls** model on a custom dataset covering **32 crop classes**.
- Apply post-classification **image processing techniques** to evaluate quality dimensions:
  - Color uniformity
  - Surface defect detection
  - Texture analysis
  - Sprouting detection
- Use a **rule-based grading system** to produce a final quality label.
- Reduce dependence on large, manually labeled ripeness datasets.
- Improve generalization across diverse fruits and vegetables.


## Dataset

The dataset was constructed specifically for this project and consists of **10,800 images** covering **32 crop classes**.

| Property | Details |
|---|---|
| Total Images | 10,800 |
| Number of Classes | 32 |
| Preprocessing | Resizing, augmentation, train/validation split |
| Background Removal | Applied using `rembg` |

### Crop Classes

The dataset includes the following 32 fruit and vegetable classes:

<!-- Replace with your actual class list -->
`Apple` · `Banana` · `Carrot` · `Cucumber` · `Grape` · `Kiwi` · `Lemon` · `Mango` · `Orange` · `Peach` · `Pear` · `Pineapple` · `Pomegranate` · `Potato` · `Strawberry` · `Tomato` · *(and 16 more)*

> 📂 The dataset folder is included in this repository. Each class has its own subdirectory.


## Methodology

```
Input Image
     │
     ▼
┌─────────────────────┐
│  Background Removal  │  ← rembg
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Crop Classification │  ← Fine-tuned YOLOv8l-cls
└─────────────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│         Image Processing Pipeline        │
│  ┌──────────────┐  ┌───────────────────┐ │
│  │ Color        │  │ Surface Defect    │ │
│  │ Uniformity   │  │ Detection         │ │
│  └──────────────┘  └───────────────────┘ │
│  ┌──────────────┐  ┌───────────────────┐ │
│  │ Texture      │  │ Sprouting         │ │
│  │ Analysis     │  │ Detection         │ │
│  └──────────────┘  └───────────────────┘ │
└──────────────────────────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Rule-Based Grading  │
│  Excellent / Acceptable / Poor
└─────────────────────┘
```

### Steps

1. **Data Collection & Preprocessing** : A total of 10,800 images across 32 crop classes were resized, augmented, and split into train/validation sets.
2. **Background Removal** : `rembg` library used to isolate the crop and improve feature extraction accuracy.
3. **Transfer Learning** : Pre-trained YOLOv8l-cls fine-tuned for 80 epochs on the custom dataset.
4. **Image Processing** : Four specialized functions applied after classification to assess quality features.
5. **Quality Grading** : A rule-based system aggregates the results into a final grade.


## Results

| Metric | Score |
|---|---|
| **Top-1 Accuracy** | **98.24%** |
| **Precision** | ~98% |
| **Recall** | ~98% |
| **F1-Score** | ~98% |
| **Training Duration** | 0.503 hours (80 epochs) |
| **Training Hardware** | NVIDIA A100 (Google Colab Pro) |

- Performance was **consistent across all 32 crop classes**.
- Results demonstrate the effectiveness of combining **transfer learning** with **image processing** for quality assessment.


## Repository Structure

```
📦 grocery-quality-assistant/
├── 📓 grocery_quality_assistant.ipynb   # Main pipeline notebook
├── 📁 dataset/                          # Crop image dataset (32 classes)
│   ├── apple/
│   ├── banana/
│   └── ...
├── 📁 model/                            # Trained YOLOv8 model weights
│   └── best.pt
└── 📄 README.md
```


## Implementation Environment

### Training Environment
| Component | Details |
|---|---|
| Platform | Google Colaboratory (Colab Pro) |
| GPU | NVIDIA A100 |
| Training Time | 0.503 hours / 80 epochs |
| Environment Manager | Miniconda / Anaconda |

### Inference Environment
| Component | Details |
|---|---|
| Platform | Local Machine — Windows 11 Home 64-bit |
| CPU | 12th Gen Intel Core i7-1260U @ 1.1 GHz |
| RAM | 16 GB |
| IDE | Visual Studio Code |

### Key Libraries

| Library | Purpose |
|---|---|
| `ultralytics` | YOLOv8 model training and inference |
| `opencv-python` | Image preprocessing and feature extraction |
| `scikit-image` | Color space manipulation, filtering, morphology |
| `rembg` | Automatic background removal |
| `Python` | Core programming language |


## Getting Started

### Prerequisites

```bash
# Create and activate a conda environment
conda create -n grocery-quality python=3.10
conda activate grocery-quality

# Install required packages
pip install ultralytics opencv-python scikit-image rembg
```

### Running the Pipeline

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/grocery-quality-assistant.git
   cd grocery-quality-assistant
   ```

2. **Open the notebook**
   ```bash
   jupyter notebook grocery_quality_assistant.ipynb
   ```

3. **Run all cells** — The notebook will:
   - Load the trained YOLOv8 model from `model/best.pt`
   - Remove the background from the input image
   - Classify the crop type
   - Run the image processing functions
   - Output the final quality grade


## Quality Grading System

The system evaluates each crop across four dimensions and combines the scores using a rule-based grading function:

| Grade | Description |
|---|---|
| 🟢 **Excellent** | Uniform color, no visible defects, smooth texture, no sprouting |
| 🟡 **Acceptable** | Minor color variation or slight surface marks; overall still fresh |
| 🔴 **Poor** | Significant discoloration, defects, rough texture, or sprouting detected |


## Future Work

- Extend the dataset to include more crop varieties and ripeness-labeled images.
- Integrate a real-time video stream for live quality inspection.
- Deploy as a mobile or web application for consumer and retail use.
- Explore additional image processing features (e.g., weight estimation, size grading).
- Compare with other architectures (e.g., EfficientNet, Vision Transformers).


## 📄 License

This project was developed as a graduation project at **King Saud University**, College of Computer and Information Sciences.


*Built using YOLOv8 + OpenCV + Python*
