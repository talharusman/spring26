"""
Image Processing: Gaussian Noise, Convolution, Denoising, and Sharpening
=========================================================================
How to run:
    python image_processing.py

Requirements:
    pip install numpy matplotlib opencv-python

Input :  dog.jpg  (must be in the same directory)
Output:  Displays and saves the following images:
         - noisy_dog.jpg          (Step 1 – Gaussian noise added)
         - sobel_convolved.jpg    (Step 2 – Sobel edge-detection kernel)
         - denoised_dog.jpg       (Step 3 – 7×7 Gaussian filter denoising)
         - sharpened_dog.jpg      (Step 4 – Sharpening filter)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Helper: 2-D convolution (from scratch with NumPy only)
# ──────────────────────────────────────────────────────────────────────
def convolve2d(image, kernel):
   
    if image.ndim == 3:
        channels = []
        for c in range(image.shape[2]):
            channels.append(convolve2d(image[:, :, c], kernel))
        return np.stack(channels, axis=-1)

    img = image.astype(np.float64)
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    # i) Zero-padding
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # ii) Flip kernel horizontally and vertically (180° rotation)
    flipped_kernel = kernel[::-1, ::-1]

    # iii) Weighted sum at each pixel
    H, W = img.shape
    output = np.zeros_like(img, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            region = padded[i:i + kH, j:j + kW]
            output[i, j] = np.sum(region * flipped_kernel)

    return output


# ──────────────────────────────────────────────────────────────────────
# Helper: Build a 2-D Gaussian kernel
# ──────────────────────────────────────────────────────────────────────
def gaussian_kernel(size, sigma):
    
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()          # normalise so weights sum to 1


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────
def main():
   
    dog_bgr = cv2.imread("dog.jpg")
    if dog_bgr is None:
        raise FileNotFoundError("dog.jpg not found in the current directory.")
    dog_rgb = cv2.cvtColor(dog_bgr, cv2.COLOR_BGR2RGB)
    dog = dog_rgb.astype(np.float64)

    # ── Step 1: Add Gaussian noise (mean=0, std=15) ──────────────────
    print("[Step 1] Adding Gaussian noise (mean=0, std=15) ...")
    noise = np.random.normal(loc=0, scale=15, size=dog.shape)
    noisy_dog = dog + noise
    noisy_dog = np.clip(noisy_dog, 0, 255).astype(np.uint8)
    cv2.imwrite("noisy_dog.jpg", cv2.cvtColor(noisy_dog, cv2.COLOR_RGB2BGR))
    print("         Saved → noisy_dog.jpg")

    # ── Step 2: Convolution with the Sobel-like kernel ───────────────
    print("[Step 2] Applying Sobel convolution kernel ...")
    sobel_kernel = np.array([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=np.float64)
    convolved = convolve2d(dog_rgb, sobel_kernel)
    # Normalise to 0-255 for visualisation
    conv_vis = convolved - convolved.min()
    if conv_vis.max() != 0:
        conv_vis = (conv_vis / conv_vis.max() * 255).astype(np.uint8)
    else:
        conv_vis = conv_vis.astype(np.uint8)
    cv2.imwrite("sobel_convolved.jpg", cv2.cvtColor(conv_vis, cv2.COLOR_RGB2BGR))
    print("         Saved → sobel_convolved.jpg")

    # ── Step 3: Denoise with a 7×7 Gaussian filter (σ = 1.0) ────────
    print("[Step 3] Denoising with 7×7 Gaussian filter (σ=1.0) ...")
    gauss_kern = gaussian_kernel(size=7, sigma=1.0)
    denoised = convolve2d(noisy_dog, gauss_kern)
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    cv2.imwrite("denoised_dog.jpg", cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
    print("         Saved → denoised_dog.jpg")

    # ── Step 4: Sharpen the image ────────────────────────────────────
    print("[Step 4] Sharpening the image ...")
    sharpening_kernel = np.array([
        [1,  4,   6,   4,  1],
        [4,  16,  24,  16, 4],
        [6,  24, -476, 24, 6],
        [4,  16,  24,  16, 4],
        [1,  4,   6,   4,  1]
    ]) * -1.0 / 256.0

    sharpened = convolve2d(dog_rgb, sharpening_kernel)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    cv2.imwrite("sharpened_dog.jpg", cv2.cvtColor(sharpened, cv2.COLOR_RGB2BGR))
    print("         Saved → sharpened_dog.jpg")

    # ── Display all results
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    titles = ["Original", "Noisy (σ=15)", "Sobel Convolution",
              "Denoised (7×7 Gauss)", "Sharpened", ""]
    images = [dog_rgb, noisy_dog, conv_vis, denoised, sharpened, None]

    for ax, title, img in zip(axes.flat, titles, images):
        if img is not None:
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("image_processing_results.png", dpi=150)
    plt.show()
    print("\nAll steps completed. Summary figure → image_processing_results.png")


if __name__ == "__main__":
    main()
