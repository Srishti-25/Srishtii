# Srishtii
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('Humayuns-Tomb.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title('Original Image')
plt.axis('off')
img_denoised = cv2.medianBlur(img_rgb, 3)
print("Image denoised and stored in 'img_denoised' variable.")
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_denoised)
ax[1].set_title('Denoised Image (Median Blur)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
edges = cv2.Canny(img_gray, 100, 200)
print("Canny edge detection applied and edges stored in 'edges' variable.")
plt.figure(figsize=(8, 4))
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')
plt.show()
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
print("Image converted to grayscale and stored in 'img_gray' variable.")
plt.figure(figsize=(8, 4))
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()
img_equalized = cv2.equalizeHist(img_gray)
print("Histogram equalization applied and image stored in 'img_equalized' variable.")
plt.figure(figsize=(8, 4))
plt.imshow(img_equalized, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')
plt.show()
kernel_sharpening = np.array([[-1,-1,-1],
                              [-1,9,-1],
                              [-1,-1,-1]])

img_sharpened = cv2.filter2D(img_rgb, -1, kernel_sharpening)
print("Image sharpening filter applied and sharpened image stored in 'img_sharpened' variable.")
plt.figure(figsize=(8, 4))
plt.imshow(img_sharpened)
plt.title('Sharpened Image')
plt.axis('off')
plt.show()
img_denoised_nl = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)
print("Non-local Means Denoising applied and stored in 'img_denoised_nl' variable.")
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_denoised_nl)
ax[1].set_title('Denoised Image (Non-local Means)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
img_lab = cv2.cvtColor(img_denoised_nl, cv2.COLOR_RGB2LAB)
l_channel, a_channel, b_channel = cv2.split(img_lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl_l_channel = clahe.apply(l_channel)

img_merged_lab = cv2.merge([cl_l_channel, a_channel, b_channel])
img_contrast_enhanced_denoised = cv2.cvtColor(img_merged_lab, cv2.COLOR_LAB2RGB)

print("Contrast enhancement applied to denoised image and stored in 'img_contrast_enhanced_denoised' variable.")
fig, ax = plt.subplots(1, 3, figsize=(20, 7))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_denoised_nl)
ax[1].set_title('Non-local Means Denoised')
ax[1].axis('off')

ax[2].imshow(img_contrast_enhanced_denoised)
ax[2].set_title('Enhanced (Denoised + Contrast)')
ax[2].axis('off')

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].imshow(img_denoised)
ax[0].set_title('Median Blur Denoised Image')
ax[0].axis('off')

ax[1].imshow(img_denoised_nl)
ax[1].set_title('Non-local Means Denoised Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()

img_denoised_gaussian = cv2.GaussianBlur(img_rgb, (5, 5), 0)
print("Gaussian Blur applied and stored in 'img_denoised_gaussian' variable.")
fig, ax = plt.subplots(1, 4, figsize=(25, 6))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_denoised)
ax[1].set_title('Median Blur Denoised')
ax[1].axis('off')

ax[2].imshow(img_denoised_nl)
ax[2].set_title('Non-local Means Denoised')
ax[2].axis('off')

ax[3].imshow(img_denoised_gaussian)
ax[3].set_title('Gaussian Blur Denoised')
ax[3].axis('off')

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 3, figsize=(20, 7))

ax[0].imshow(img_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img_denoised_nl)
ax[1].set_title('Non-local Means Denoised')
ax[1].axis('off')

ax[2].imshow(img_denoised_gaussian)
ax[2].set_title('Gaussian Blur Denoised')
ax[2].axis('off')

plt.tight_layout()
plt.show()
