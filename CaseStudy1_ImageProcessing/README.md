# Image Processing (Barbara, Flower, Clock) - (reworked old work)
 
 Magnus H
**Topics:** image loading & typing, grayscale conversion, resizing, brightness & histograms, noise modeling (SNR→variance), denoising (average vs. median), edge detection (Sobel/LoG/Canny), multi-scale template matching (NCC).  
**Tools:** MATLAB, Python (OpenCV), Peter Corke Machine Vision Toolbox

---

## 1) Getting Familiar with Images & Processing Tools

### 1.1 Loading *Barbara* three ways
Start by verifing consistency across loaders and dtypes:

- From `.mat` (MATLAB)
- From `.png` using MATLAB Image Processing Toolbox (IPT)
- From `.png` using Peter Corke’s Machine Vision Toolbox (MVT)

**Outputs**

![Barbara from .mat](./images/barbara_from_mat.png)  
*Loaded from `.mat`.*

![Barbara via MATLAB IPT](./images/barbara_from_png_MATLAB.png)  
*Loaded from `.png` via MATLAB IPT.*

![Barbara via Peter Corke MVT](./images/barbara_from_png_PeterCorke.png)  
*Loaded from `.png` via Peter Corke’s MVT.*

**Notes**
- After explicit dtype normalization, pixel values are consistent across loaders.
- Be explicit with `uint8↔double` casts when mixing libraries.

### 1.2 Grayscale, size, dtype
Converted to grayscale, inspected shape/channels/dtype; then created half/quarter versions.

![Grayscale 256×256](./images/grayscale_barbara.png)  
*Grayscale Barbara (256×256).*

![Half size](./images/gray_half_barbara.png)  
*Half size.*

![Quarter size](./images/gray_quarter_barbara.png)  
*Quarter size.*

### 1.3 Brightness adjustment & histograms
Brightness and histogram behavior (linear/gamma) were compared.

![Brighter image](./images/brighter_gray_barbara.png)

| Before | After |
|---|---|
| ![Original hist](./images/original_histogram.png) | ![Brightened hist](./images/brightened_histogram.png) |

**Observations**
- Linear brightness shifts translate the histogram; gamma < 1 lifts mid-tones with less highlight clipping.
- Always check value range (0–1 vs 0–255) before histogramming.

---

## 2) Image Restoration — Noise Modeling & Filtering

### 2.1 Setup: SNR→noise variance
Gaussian noise added with **SNR = 20 dB**. Noise variance derived from signal variance:

\[
\sigma^2_{\text{noise}}=\frac{\sigma^2_{\text{signal}}}{10^{\text{SNR}/10}}
\]

Here, \(\sigma^2_{\text{signal}}=0.0516 \Rightarrow \sigma^2_{\text{noise}}\approx 5.1635\times 10^{-4}\).

**Filter heuristics**
- **20 dB (moderate noise)**: Average \(5\times5\); Median \(3\times3\).  
- Lower SNR → larger kernels (more smoothing); higher SNR → smaller kernels (detail preservation).

### 2.2 Gaussian noise → Average vs. Median

| Noisy (20 dB) | Average \(5\times5\) | Median \(3\times3\) |
|---|---|---|
| ![Noisy Gauss](./images/flower_noisy_gauss.png) | ![Avg](./images/flower_filtered_avg_gauss.png) | ![Median](./images/flower_filtered_median_gauss.png) |

**Takeaways**
- Average reduces Gaussian “grain” well but blurs edges.
- Median preserves edges better but is not optimal statistically for Gaussian noise.

### 2.3 Salt & Pepper (20%) → Average vs. Median

| Noisy (20%) | Average \(5\times5\) | Median \(3\times3\) |
|---|---|---|
| ![Noisy S&P](./images/flower_noisy_saltpepper.png) | ![Avg S&P](./images/flower_filtered_avg_saltpepper.png) | ![Median S&P](./images/flower_filtered_median_saltpepper.png) |

**Takeaways**
- Median dominates for impulse noise; Average tends to smear pepper/salt.

### 2.4 Combined noise → Sequential filters

| Combined Noisy | Avg → Median |
|---|---|
| ![Combined](./images/flower_noisy_combined.png) | ![Sequential](./images/flower_filtered_combined.png) |

**Strategy**
- First reduce Gaussian variance (Average), then remove outliers (Median).  
- Order matters: Median→Average can re-blur protected edges.

---

## 3) Edge Detection — Sobel, LoG, Canny

**Input**  
![Clock original](./images/clock_original.png)

**Outputs**

| Sobel | LoG | Canny |
|---|---|---|
| ![Sobel](./images/clock_sobel_edges.png) | ![LoG](./images/clock_log_edges.png) | ![Canny](./images/clock_canny_edges.png) |

**Binary thresholding example (Sobel)**  
![Sobel binary](./images/clock_sobel_binary.png)

**Notes**
- **Sobel**: simple gradients; noise-sensitive; threshold tuning needed.  
- **LoG**: smoothing + Laplacian; good zero-crossings; may double-respond at edges.  
- **Canny**: non-max suppression + hysteresis; best balance of recall/precision with adaptive thresholds.

---

## 4) Template Matching — Multi-Scale NCC

**Pipeline**
1. Grayscale + **Canny** (image & template)  
2. **Morphological closing** (fills edge gaps)  
3. **Multi-scale** pyramid search with **Normalized Cross-Correlation (NCC)**  
4. **Argmax NCC** → location + scale

**Result**

![Template detected](./images/task4_result.png)  
*Best NCC ≈ 0.67; rectangle marks detected location.*


- Edge-based preprocessing reduces lighting sensitivity.  
- A light Gaussian blur can stabilize NCC maps.  
- For multiple targets, apply non-max suppression on the NCC heatmap.

## 🧭 Discussion & Takeaways
- **SNR math → filter sizing:**  
  By computing the noise variance directly from a given SNR (for example, 20 dB), filtering became a **quantitative design decision** rather than trial-and-error.  
  Knowing the signal-to-noise ratio allows you to select kernel sizes that balance noise suppression and detail retention — large kernels for lower SNR (noisier images) and smaller kernels for higher SNR (cleaner images).

- **Median vs Average → match the filter to the noise type:**  
  Gaussian noise (continuous fluctuations) responds best to an **average filter**, while impulse noise (salt & pepper) is better removed by a **median filter** that preserves edges.  
  When both noise types were present, a **sequential combination** (average → median) leveraged both approaches to achieve the cleanest restoration.

- **Canny for robust edge detection:**  
  Compared to Sobel and Laplacian-of-Gaussian, **Canny** consistently produced **thin, stable, and connected** edges thanks to its Gaussian pre-filtering, non-maximum suppression, and hysteresis thresholds.  
  This makes it a strong default choice for **segmentation, contour extraction, or feature matching** stages later in the pipeline.

- **NCC + edges + multi-scale search:**  
  Normalized Cross-Correlation (NCC) is most reliable when comparing **edge-based representations** rather than raw intensities.  
  Edge preprocessing removes brightness dependency, while a **multi-scale pyramid search** ensures templates can be matched even when size or viewpoint changes.  
  The result is a correlation method that’s far more **robust to illumination and scale variation**, giving accurate and repeatable template localization.

---

### 🔗 Integrated Understanding

All four insights connect into a single low-level vision sequence:

| Processing Step | Purpose | Broader Role |
|-----------------|----------|--------------|
| **SNR-based filtering** | Quantify and control noise reduction | Foundation for reliable preprocessing |
| **Filter type selection** | Adapt to different noise distributions | Preserve image structure and edge integrity |
| **Edge detection (Canny)** | Extract stable, continuous boundaries | Provide clean input for segmentation or feature matching |
| **Template matching (NCC)** | Recognize geometric patterns across scales | Early stage toward object detection and localization |

Together they form a **progressive vision pipeline:**  
raw pixels → denoised → edges → pattern recognition.  

These relationships explain why the techniques from Lab 1 build the foundation for later modules such as **feature detection, segmentation, and stereo vision**, where the quality of early preprocessing determines the reliability of all higher-level results.
