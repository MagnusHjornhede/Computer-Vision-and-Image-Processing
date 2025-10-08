#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-scale template matching (NCC on edge images)

Magnus H
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Hardcoded config (edit here)
# ----------------------------
IMAGE_NAME = "original_image.jpg"
TEMPLATE_NAME = "template.png"

SCALE_FACTOR = 0.85      # template downscale per level
MAX_LEVELS = 18
MIN_SIZE = 30            # stop when min(template side) < this
CANNY_SIGMA = 0.33       # auto-canny sigma
MANUAL_CANNY = None      # e.g., (100, 250) to override auto; or None
CLOSING_KERNEL = 3       # morphological closing kernel size
CLOSING_ITERS = 1
EARLY_STOP = 0.92        # stop early if NCC exceeds this

# ----------------------------


def auto_canny(gray, sigma=0.33):
    """Compute Canny thresholds from image median (robust to brightness)."""
    med = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(gray, lower, upper), (lower, upper)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    # Paths: use this task folder + images/
    task_dir = Path(__file__).resolve().parent
    img_dir = task_dir / "images"
    ensure_dir(img_dir)

    image_path = img_dir / IMAGE_NAME
    template_path = img_dir / TEMPLATE_NAME

    # Load
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if template is None:
        raise FileNotFoundError(f"Could not read template: {template_path}")

    # Grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Edges: manual or auto
    if MANUAL_CANNY is not None:
        tl, tu = MANUAL_CANNY
        image_edges = cv2.Canny(image_gray, tl, tu)
        template_edges = cv2.Canny(template_gray, tl, tu)
        canny_note = f"manual Canny [{tl},{tu}]"
    else:
        image_edges, (il, iu) = auto_canny(image_gray, sigma=CANNY_SIGMA)
        template_edges, (tl, tu) = auto_canny(template_gray, sigma=CANNY_SIGMA)
        canny_note = f"auto Canny σ={CANNY_SIGMA:.2f} img[{il},{iu}] tpl[{tl},{tu}]"

    # Closing to connect gaps
    kernel = np.ones((CLOSING_KERNEL, CLOSING_KERNEL), np.uint8)
    img_closed = cv2.morphologyEx(image_edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSING_ITERS)
    tpl_closed = cv2.morphologyEx(template_edges, cv2.MORPH_CLOSE, kernel, iterations=CLOSING_ITERS)

    # Save debug
    cv2.imwrite(str(img_dir / "task4_image_edges.png"), img_closed)
    cv2.imwrite(str(img_dir / "task4_template_edges.png"), tpl_closed)

    # Multi-scale search (downscale template)
    best_score = -1.0
    best_top_left = None
    best_wh = None

    h_img, w_img = img_closed.shape[:2]
    tmpl = tpl_closed.copy()

    for _ in range(MAX_LEVELS):
        th, tw = tmpl.shape[:2]
        if th < MIN_SIZE or tw < MIN_SIZE:
            break
        if th <= h_img and tw <= w_img:
            res = cv2.matchTemplate(img_closed, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_top_left = max_loc
                best_wh = (tw, th)

            if best_score >= EARLY_STOP:
                break

        new_w = max(int(tw * SCALE_FACTOR), 1)
        new_h = max(int(th * SCALE_FACTOR), 1)
        if new_w < MIN_SIZE or new_h < MIN_SIZE:
            break
        tmpl = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Draw result
    vis = image.copy()
    if best_top_left and best_wh:
        x, y = best_top_left
        w, h = best_wh
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        title = f"NCC={best_score:.2f} | {canny_note}"
    else:
        title = f"No match | {canny_note}"

    out_path = img_dir / "final_best_matched_image_pyramid_optimized.jpg"
    cv2.imwrite(str(out_path), vis)

    # Show
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"✅ Done. Best score: {best_score:.4f}")
    print(f"   Result saved: {out_path}")
    print("   Debug edges: task4_image_edges.png, task4_template_edges.png")


if __name__ == "__main__":
    main()
