```python
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load the image
image_path = "/content/제목 없음4.png"  # Path to the uploaded image
image = Image.open(image_path)

# Convert the image to OpenCV format
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Step 1: Detect QR codes and create a mask to completely ignore their areas
qr_decoder = cv2.QRCodeDetector()

# Detect and decode QR codes in the original image
retval, decoded_info, points, _ = qr_decoder.detectAndDecodeMulti(image_cv)

# Create a mask to ignore QR code areas
mask = np.ones(image_cv.shape[:2], dtype=np.uint8) * 255  # Initialize a white mask

# Check if any QR codes are detected
qr_contours = []
if retval and points is not None:
    for i in range(len(points)):
        pts = points[i].reshape(-1, 2).astype(int)
        qr_contours.append(pts)  # Store QR code contours for further analysis
        # Strongly mask QR code areas on the mask to prevent them from being detected as part of the box contour
        cv2.fillPoly(mask, [pts], 0)  # Fill QR code area with black to fully mask it

# Step 2: Detect box contours (without QR code influence)
# Convert to grayscale
gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

# Apply the mask to the grayscale image to ignore QR code areas
masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

# Apply edge detection on the masked image
edges = cv2.Canny(masked_gray, 50, 150)

# Find contours in the masked image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area to exclude small noise (set to 0 for full detection)
min_area = 0
box_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Draw contours on a copy of the original image for visualization
contour_image = image_cv.copy()
cv2.drawContours(contour_image, box_contours, -1, (0, 255, 0), 2)  # Green for box contour

# Find the largest contour which is assumed to be the box
if box_contours:
    box_contour = max(box_contours, key=cv2.contourArea)

    # Get bounding rectangle of the box contour
    box_rect = cv2.boundingRect(box_contour)

    # Check if QR codes are inside the box contour and print URLs accordingly
    if retval and points is not None:
        for i, qr_pts in enumerate(qr_contours):
            # Get bounding rectangle of the QR code
            qr_rect = cv2.boundingRect(qr_pts)

            # Check if QR bounding box is fully inside the box bounding box
            inside = (qr_rect[0] >= box_rect[0] and
                      qr_rect[1] >= box_rect[1] and
                      qr_rect[0] + qr_rect[2] <= box_rect[0] + box_rect[2] and
                      qr_rect[1] + qr_rect[3] <= box_rect[1] + box_rect[3])

            # Determine if QR code is inside or outside based on bounding box check
            if inside:
                print(f"QR Code Detected Outside Box: {decoded_info[i]}")
            else:
                print(f"QR Code Detected Outside Box: {decoded_info[i]}")

# Display the original and contour image side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
axes[1].set_title("Detected Box Contours")
axes[1].axis("off")

plt.tight_layout()
plt.show()

```

    QR Code Detected Outside Box: https://me-qr.com/trAzWwOj



    
![png](2024-10-03-Box_QR_Detection_files/2024-10-03-Box_QR_Detection_0_1.png)
    

