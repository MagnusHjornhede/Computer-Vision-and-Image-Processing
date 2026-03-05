# Computer Vision Algorithms

Implementation of classical computer vision and image processing algorithms using **Python and OpenCV**.

This repository collects a set of small experiments and case studies covering fundamental techniques used in many vision pipelines.

<br>

## Implemented Techniques

- Edge detection
- Image filtering
- Thresholding
- Feature extraction
- Image enhancement

<br>

## Technologies

Python  
OpenCV  
NumPy

<br>

## Goal

The goal of this repository is to explore and implement core computer vision algorithms that form the foundation of many modern vision systems.

Each module focuses on a specific topic and demonstrates how classical techniques can be applied to real images.

<br>

## Case Studies

Each case study is organized in its own folder with code, visual outputs, and short documentation.

<br>

### 1. Image Processing Fundamentals  
📂 Folder: [CaseStudy1_ImageProcessing](https://github.com/MagnusHjornhede/Computer-Vision-and-Image-Processing/tree/main/CaseStudy1_ImageProcessing)

Focus: improving and analyzing image quality.

Techniques explored:

- Grayscale conversion and histogram equalization
- Image smoothing, sharpening, and denoising
- Color space transformations (RGB ↔ HSV)
- Visual comparison of filter effects

These operations form the preprocessing stage of many vision pipelines.

<br>

### 2. Feature Detection and Tracking  
📂 Folder: [CaseStudy2_FeatureDetection](https://github.com/MagnusHjornhede/Computer-Vision-and-Image-Processing/tree/main/CaseStudy2_FeatureDetection)

Focus: identifying and tracking meaningful image structures.

Techniques implemented:

- Harris corner detection
- Edge-based feature extraction
- Template matching
- Frame-to-frame feature tracking

This module explores how local patterns and geometric structures can be detected and tracked across images.

<br>

### 3. Image Segmentation  
📂 Folder: [CaseStudy3_Segmentation](https://github.com/MagnusHjornhede/Computer-Vision-and-Image-Processing/tree/main/CaseStudy3_Segmentation)

Focus: separating images into meaningful regions.

Methods explored:

- Global and adaptive thresholding (including Otsu)
- Region growing and contour detection
- K-means clustering
- Watershed segmentation

Segmentation is a key step in moving from pixel-level processing to object-level understanding.

<br>

### 4. Stereo Vision and Depth Estimation  
📂 Folder: [CaseStudy4_StereoVision](https://github.com/MagnusHjornhede/Computer-Vision-and-Image-Processing/tree/main/CaseStudy4_StereoVision)

Focus: estimating depth from stereo image pairs.

Implemented components:

- Stereo calibration and rectification
- Disparity map computation
- Depth estimation and visualization

This section demonstrates how camera geometry can be used to reconstruct 3D structure from 2D images.
