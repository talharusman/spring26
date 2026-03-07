"""
Template Matching: Convolution vs. Correlation
===============================================
How to run:
    python template_matching.py

Requirements:
    pip install numpy matplotlib opencv-python

Input :  shelf.jpg, template.jpg  (must be in the same directory)
Output:  Displays and saves the following images:
         - match_convolution.jpg   (result using convolution-based matching)
         - match_correlation.jpg   (result using correlation-based matching)
         - template_matching_comparison.png  (side-by-side summary)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time


# ──────────────────────────────────────────────────────────────────────
# Helper: 2-D convolution (from scratch with NumPy only)
# ──────────────────────────────────────────────────────────────────────
def convolve2d(image, kernel):
    
    img = image.astype(np.float64)
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    # i) Zero-padding
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)

    # ii) Flip kernel 180°
    flipped_kernel = kernel[::-1, ::-1]

    # iii) Weighted sum at each pixel
    H, W = img.shape
    output = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * flipped_kernel)

    return output


# ──────────────────────────────────────────────────────────────────────
# Helper: 2-D correlation (from scratch with NumPy only)
# ──────────────────────────────────────────────────────────────────────
def correlate2d(image, kernel):
   
    img = image.astype(np.float64)
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    # Zero-padding
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant', constant_values=0)

    # NO flip – this is correlation, not convolution
    H, W = img.shape
    output = np.zeros((H, W), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * kernel)

    return output


# ──────────────────────────────────────────────────────────────────────
# Template matching via convolution
# ──────────────────────────────────────────────────────────────────────
def template_match_convolution(image_gray, template_gray):
    
    # Subtract means to prevent bias toward brighter regions
    img_zero = image_gray.astype(np.float64) - np.mean(image_gray)
    tmpl_zero = template_gray.astype(np.float64) - np.mean(template_gray)

    
    # (convolve2d internally flips again → net effect = original orientation)
    flipped_tmpl = tmpl_zero[::-1, ::-1]

    response = convolve2d(img_zero, flipped_tmpl)
    return response


# ──────────────────────────────────────────────────────────────────────
# Template matching via correlation
# ──────────────────────────────────────────────────────────────────────
def template_match_correlation(image_gray, template_gray):
   
    # Subtract means
    img_zero = image_gray.astype(np.float64) - np.mean(image_gray)
    tmpl_zero = template_gray.astype(np.float64) - np.mean(template_gray)

    response = correlate2d(img_zero, tmpl_zero)
    return response


# ──────────────────────────────────────────────────────────────────────
# Draw a bounding box at the best-match location
# ──────────────────────────────────────────────────────────────────────
def draw_match(image_rgb, response_map, template_shape, colour, label):
   
    tH, tW = template_shape[:2]
    peak_idx = np.unravel_index(np.argmax(response_map), response_map.shape)
    cy, cx = peak_idx    # centre of match

    top_left = (max(cx - tW // 2, 0), max(cy - tH // 2, 0))
    bottom_right = (min(cx + tW // 2, image_rgb.shape[1]),
                    min(cy + tH // 2, image_rgb.shape[0]))

    result = image_rgb.copy()
    cv2.rectangle(result, top_left, bottom_right, colour, 3)
    cv2.putText(result, label, (top_left[0], max(top_left[1] - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return result, (cy, cx)


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def main():
    # Load images
    shelf_bgr = cv2.imread("shelf.jpg")
    template_bgr = cv2.imread("template.jpg")
    if shelf_bgr is None:
        raise FileNotFoundError("shelf.jpg not found in the current directory.")
    if template_bgr is None:
        raise FileNotFoundError("template.jpg not found in the current directory.")

    shelf_rgb = cv2.cvtColor(shelf_bgr, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)
    shelf_gray = cv2.cvtColor(shelf_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    print(f"Shelf   image size : {shelf_gray.shape}")
    print(f"Template image size: {template_gray.shape}")

    # ── Convolution-based template matching 
    print("\n[1] Running template matching via CONVOLUTION ...")
    t0 = time.time()
    resp_conv = template_match_convolution(shelf_gray, template_gray)
    time_conv = time.time() - t0
    print(f"    Time elapsed: {time_conv:.2f} s")

    result_conv, loc_conv = draw_match(shelf_rgb, resp_conv,
                                       template_gray.shape,
                                       (255, 0, 0), "Convolution")
    cv2.imwrite("match_convolution.jpg",
                cv2.cvtColor(result_conv, cv2.COLOR_RGB2BGR))
    print(f"    Peak location (row, col): {loc_conv}")
    print("    Saved → match_convolution.jpg")

    # ── Correlation-based template matching 
    print("\n[2] Running template matching via CORRELATION ...")
    t0 = time.time()
    resp_corr = template_match_correlation(shelf_gray, template_gray)
    time_corr = time.time() - t0
    print(f"    Time elapsed: {time_corr:.2f} s")

    result_corr, loc_corr = draw_match(shelf_rgb, resp_corr,
                                       template_gray.shape,
                                       (0, 255, 0), "Correlation")
    cv2.imwrite("match_correlation.jpg",
                cv2.cvtColor(result_corr, cv2.COLOR_RGB2BGR))
    print(f"    Peak location (row, col): {loc_corr}")
    print("    Saved → match_correlation.jpg")

    # ── Visualise results side-by-side 
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(shelf_rgb);          axes[0, 0].set_title("Shelf (original)")
    axes[0, 1].imshow(template_rgb);       axes[0, 1].set_title("Template")
    axes[0, 2].axis("off")

    # Normalise response maps for display
    def norm_map(m):
        m = m - m.min()
        if m.max() != 0:
            m = m / m.max()
        return m

    axes[1, 0].imshow(norm_map(resp_conv), cmap='hot')
    axes[1, 0].set_title(f"Convolution response\n(time: {time_conv:.2f}s)")

    axes[1, 1].imshow(norm_map(resp_corr), cmap='hot')
    axes[1, 1].set_title(f"Correlation response\n(time: {time_corr:.2f}s)")

    # Combined overlay
    combined = shelf_rgb.copy()
    tH, tW = template_gray.shape
    # Draw convolution box (red)
    cv2.rectangle(combined,
                  (max(loc_conv[1] - tW // 2, 0), max(loc_conv[0] - tH // 2, 0)),
                  (min(loc_conv[1] + tW // 2, combined.shape[1]),
                   min(loc_conv[0] + tH // 2, combined.shape[0])),
                  (255, 0, 0), 3)
    # Draw correlation box (green)
    cv2.rectangle(combined,
                  (max(loc_corr[1] - tW // 2, 0), max(loc_corr[0] - tH // 2, 0)),
                  (min(loc_corr[1] + tW // 2, combined.shape[1]),
                   min(loc_corr[0] + tH // 2, combined.shape[0])),
                  (0, 255, 0), 3)
    axes[1, 2].imshow(combined)
    axes[1, 2].set_title("Combined (Red=Conv, Green=Corr)")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("template_matching_comparison.png", dpi=150)
    plt.show()

    # ── Discussion (printed to console) 
    print("\n" + "=" * 65)
    print("ANALYSIS & DISCUSSION")
    print("=" * 65)

    print("""
(e) Which method more accurately locates the product on the shelf?
    ---------------------------------------------------------------
    Correlation-based template matching typically locates the product
    more accurately. Because correlation directly slides the template
    across the image WITHOUT flipping, the orientation of the template
    and the target region in the shelf image match naturally.
    Convolution flips the kernel (template) by 180°. Even though we
    pre-flip the template before calling convolve2d (to cancel out the
    internal flip), any slight asymmetry or implementation nuance can
    affect the result. In practice, both should give similar peak
    locations when the pre-flip is handled correctly, but correlation
    is the theoretically correct operation for template matching.

(f) Which method is more efficient in terms of computation?
    --------------------------------------------------------
    Both methods have the same computational complexity: O(N * M) where
    N is the number of pixels in the image and M is the number of pixels
    in the template. However, correlation is marginally more efficient
    in implementation because it does NOT require the extra step of
    flipping the kernel. This saves one array reversal operation.

(g) Why may one method be better suited for template matching?
    -----------------------------------------------------------
    Correlation is better suited for template matching because:
    1. Template matching is fundamentally a SIMILARITY measure –
       we are looking for the region in the image that looks most
       like the template. Correlation directly computes this similarity.
    2. Convolution is designed for applying FILTERS (edge detection,
       blurring, etc.) where the kernel represents a mathematical
       operation, not a visual pattern. The 180° flip is essential
       for the mathematical properties of convolution (associativity,
       commutativity) but is unnecessary and counter-intuitive for
       pattern matching.
    3. Correlation avoids the extra pre-flip step, making the code
       simpler, less error-prone, and marginally faster.

(h) See the saved images (match_convolution.jpg, match_correlation.jpg,
    and template_matching_comparison.png) for visual results showing
    the detected product location on the shelf using both methods.
""")


if __name__ == "__main__":
    main()
