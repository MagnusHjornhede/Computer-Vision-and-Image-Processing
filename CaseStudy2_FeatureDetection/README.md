# 🔎 Lab 2 — Image Feature Detection and Matching (Reworked)

**Author:** Magnus Hjornhede  
**Topics:** Harris corners (custom vs OpenCV), parameter tuning, SIFT keypoints & matching, ORB stereo matching, robotics dataset performance  
**Tools:** Python (Colab + OpenCV)

---

## 1️⃣ Task 1 — Harris Corner Detector (Custom vs OpenCV)

Implemented a **custom Harris Corner Detector** from scratch and compared it to OpenCV’s built-in function to explore **corner response, rotation/scale sensitivity, and parameter influence**.

### Harris Corner Algorithm
1. **Convert to grayscale**  
2. **Compute gradients** using Sobel filters (\(x,y\))  
3. **Smooth gradients** via Gaussian blur  
4. **Compute Harris response** (determinant / trace of second-moment matrix)  
5. **Apply non-maximum suppression + thresholding** to retain strong corners  

### Quantitative Results
| Variant | Corners Detected |
|:--|:--:|
| Custom Harris – Original | 13 |
| Custom Harris – Rotated | 13 |
| Custom Harris – Scaled 0.5× | 12 |
| Custom Harris – Scaled 2× | 14 |
| OpenCV Harris – Original | 272 |
| OpenCV Harris – Rotated | 305 |
| OpenCV Harris – Scaled 0.5× | 269 |
| OpenCV Harris – Scaled 2× | 367 |

### Visuals
![Custom Harris – original](./images/custom_harris_original.png)  
![Custom Harris – rotated](./images/custom_harris_rotated.png)  
![OpenCV Harris – original](./images/opencv_harris_original.png)  
![OpenCV Harris – rotated](./images/opencv_harris_rotated.png)

### Interpretation
- **Rotation robustness:** Custom Harris was stable (13 → 13); OpenCV detected extra weak responses (272 → 305).  
- **Scale sensitivity:** Upscaling creates more pixel gradients → more corners.  
- **Parameter effects:** Low *k* ≈ 0.04 = more sensitive; low threshold = noisier; Gaussian blur = less noise.  

---

## 2️⃣ Task 2 — SIFT Detection and Matching

Applied **SIFT** (Scale-Invariant Feature Transform) to test its **robustness to scale and rotation** and compared it against Harris + ORB descriptors.

### SIFT Results
![SIFT – left/right](./images/sift_matches.png)  
![SIFT – Notre Dame](./images/sift_matches_notre_dame.png)

### Harris → Keypoints + ORB Descriptors
![Harris+ORB – left/right](./images/harris_matches_left_right.png)  
![Harris+ORB – Notre Dame](./images/harris_matches_notre_dame.png)

### Observations
- **SIFT:** high density of keypoints, robust under rotation and scale changes, some redundancy expected.  
- **Harris (+ ORB):** lightweight and fast but detects fewer points and misses complex textures.  
- **Trade-off:** SIFT = accuracy and invariance; Harris + ORB = efficiency for real-time systems.  

---

## 3️⃣ Task 3 — Feature Matching in a Robotic Scenario

Applied feature matching to two stereo datasets using **ORB + Brute-Force matcher**:

- **Custom dataset** (Boston Dynamics Spot robot)  
- **PennCOSYVIO** dataset  

### Performance Metrics
| Dataset | FPS | Avg Matches / Pair | Failed Pairs |
|:--|:--:|:--:|:--:|
| Custom (Spot) | 112.1 | 11.0 | 0 |
| PennCOSYVIO | 53.3 | 231.0 | 0 |

### Examples (PennCOSYVIO)
![Pair 10 Penn](./images/pair_10_matches_penn.png)  
![Pair 15 Penn](./images/pair_15_matches_penn.png)  
![Pair 20 Penn](./images/pair_20_matches_penn.png)

### Examples (Custom Dataset)
![Pair 10 Custom](./images/pair_10_matches_custom.png)  
![Pair 15 Custom](./images/pair_15_matches_custom.png)  
![Pair 20 Custom](./images/pair_20_matches_custom.png)

### Analysis
- **Custom dataset:** few distinct features → low match count despite high FPS (speed > quality).  
- **PennCOSYVIO:** dense textures and good overlap → hundreds of matches.  
- **Insight:** texture and stereo geometry matter more than raw speed for stable matching.  

---

## 🧭 Discussion & Takeaways

### 🔹 Corner Detectors
- Custom Harris is stable and precise; OpenCV detects dense responses needing post-filtering.  
- Multi-scale pyramids help normalize scale differences in real scenes.

### 🔹 Descriptor and Matcher Comparisons
- **SIFT** offers robust, repeatable matches across illumination and viewpoint changes.  
- **Harris + ORB** is faster but less stable for complex imagery.  
- SIFT is ideal for mapping / SLAM; ORB fits embedded real-time applications.

### 🔹 Robotics Dataset Insights
- Feature density depends on scene texture and camera baseline.  
- High FPS alone is not enough; **feature quality and geometric consistency** are crucial.  
- Combine ORB or SIFT with **RANSAC + epipolar filtering** for reliable pose estimation.

---

## 🧩 Integrated Understanding

| Stage | Purpose | Broader Role |
|:--|:--|:--|
| **Harris Corner Detection** | Identify repeatable interest points | Foundation for feature descriptors |
| **SIFT Descriptors** | Capture scale/rotation-invariant signatures | Enable cross-view matching |
| **ORB + Brute Force** | Efficient real-time matching | Supports stereo vision & odometry |
| **Evaluation (FPS + Match Count)** | Quantify speed vs robustness | Balance accuracy and efficiency in robot vision |

Together these steps show how modern vision pipelines progress from low-level corner math to high-level 3D understanding.  
They form the practical foundation for **stereo vision, structure-from-motion, and robotic localization** tasks that follow in later labs.
