# Depth Perception and 3D Reconstruction from Stereo Images

 Magnus H
**Topics:** stereo calibration, rectification, disparity, 3D reconstruction, pedestrian detection  
**Tools:** MATLAB (Stereo Camera Calibrator, ACF detector)

---

## Objective
Implement a complete **depth perception pipeline** using stereo images — covering calibration, rectification, disparity mapping, 3D reconstruction, and pedestrian detection.

---

## 1) Camera Calibration

### Intrinsic Parameters
Each camera’s intrinsics define focal length, principal point, and distortion coefficients.

\[
K = 
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\]

- **Camera 1**
  \[
  K_1 = 
  \begin{bmatrix}
  718.856 & 0 & 607.1928 \\
  0 & 718.856 & 185.2157 \\
  0 & 0 & 1
  \end{bmatrix}
  \]
- **Camera 2**
  \[
  K_2 =
  \begin{bmatrix}
  718.856 & 0 & 607.1928 \\
  0 & 718.856 & 185.2157 \\
  0 & 0 & 1
  \end{bmatrix}
  \]

### Extrinsic Parameters
Define relative orientation and translation between the two cameras.

\[
R =
\begin{bmatrix}
1.0000 & 0.0024 & 0.0042 \\
-0.0025 & 0.9999 & 0.0137 \\
-0.0041 & -0.0137 & 0.9999
\end{bmatrix},
\quad
T = [-109.6614,\; 0.5196,\; 0.5960]^T\ \text{mm}
\]

### Fundamental & Essential Matrices

**Fundamental Matrix (F):**
\[
F =
\begin{bmatrix}
-3.2184\times10^{-9} & -2.8043\times10^{-6} & 0.0018 \\
6.5546\times10^{-7} & -7.0014\times10^{-6} & 0.2383 \\
-7.0464\times10^{-4} & -0.2335 & -4.0983
\end{bmatrix}
\]

**Essential Matrix (E):**
\[
E =
\begin{bmatrix}
-6.9304\times10^{-4} & -0.6031 & 0.5114 \\
0.1410 & -1.5038 & 109.6526 \\
-0.2504 & -109.6520 & -1.5063
\end{bmatrix}
\]

---

## 2) Rectification and Disparity Map Calculation

### Image Rectification
Rectification aligns epipolar lines horizontally, simplifying correspondence search.

> Used MATLAB’s `rectifyStereoImages` to warp both images into a common plane.

### Disparity Map
Calculated using **Semi-Global Matching (SGM)** via MATLAB’s `disparitySGM`:

```matlab
disparityMap = disparitySGM(frameLeftGray, frameRightGray, ...
                            'DisparityRange', [0 64]);
```

- Disparity range optimized for expected depth.
- Each pixel value encodes horizontal shift between the two cameras.

---

## 3) 3D Reconstruction

Converted disparity to real-world 3D coordinates \((X, Y, Z)\) using MATLAB’s `reconstructScene`.

```matlab
points3D = reconstructScene(disparityMap, stereoParams.ReprojectionMatrix) / 1000;
```

- Depth \( Z \propto 1 / \text{disparity} \)  
- Scaled from **millimeters → meters**  
- Generated a dense 3D point cloud.

**Visuals**

![Disparity map](./images/disparity_map.png)  
![Reconstructed 3D points](./images/3d_reconstruction.png)

---

## 4) Pedestrian Detection — ACF (Aggregated Channel Features)

### Method
Used MATLAB’s **`peopleDetectorACF`**, which:
- Extracts gradient and color-based channel features.
- Uses boosted decision trees for classification.
- Is robust to **illumination**, **pose**, and **background** variation.

### Implementation

```matlab
peopleDetector = peopleDetectorACF();
bboxes = detect(peopleDetector, frameLeftGray);
annotatedFrame = insertObjectAnnotation(frameLeftRect, 'rectangle', bboxes, "Person");
```

**Visual Example**

![Pedestrian Detection](./images/pedestrian_detection.png)

---

## 🧭 Discussion & Takeaways

- **Calibration**: Accurate intrinsics/extrinsics are vital — small errors propagate to disparity and depth.  
- **Rectification**: Ensures efficient disparity computation; misalignment leads to slanted epipolar lines.  
- **Disparity (SGM)**: Produces smoother depth maps than block matching; sensitive to texture and occlusion.  
- **3D Reconstruction**: Depth precision decreases quadratically with distance — design baseline accordingly.  
- **ACF Detection**: Lightweight and effective; can be combined with depth to filter false positives.

---

## 🧩 Integrated Understanding

| Stage | Purpose | Key Insight |
|-------|----------|-------------|
| Calibration | Recover camera intrinsics/extrinsics | Foundation for metric reconstruction |
| Rectification | Align stereo pair epipolar geometry | Simplifies disparity search |
| Disparity (SGM) | Estimate pixel shifts | Inverse proportionality to depth |
| 3D Reconstruction | Map disparities to world coordinates | Enables metric scene interpretation |
| ACF Detection | Human/object localization | Merges 2D detection with 3D awareness |

---

This case study explains how depth perception from stereo images works by combining geometric calibration, rectification, disparity computation, and 3D reconstruction into one functional pipeline.

The key idea is that two cameras placed side by side observe the same scene from slightly different viewpoints. By knowing each camera’s intrinsic parameters (focal length, principal point, distortion) and extrinsic parameters (relative rotation and translation), every point seen in both images can be triangulated in 3D space. Calibration provides the mathematical relationship needed for this triangulation to produce real-world distances instead of arbitrary pixel offsets.

Rectification is used to align the two image planes so that corresponding pixels lie on the same horizontal line. This greatly simplifies the search for correspondences — the system only needs to look along the x-axis rather than across the entire image.

The disparity map measures how much each pixel shifts between the left and right views. Nearby objects have large disparities (bigger shifts), while distant objects have smaller ones. Using this relationship, disparity can be directly converted to depth through simple geometry. The resulting 3D reconstruction provides metric information that can be used for navigation, obstacle avoidance, or mapping.

