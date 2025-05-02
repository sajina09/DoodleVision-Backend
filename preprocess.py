import io
import numpy as np
import cv2
from PIL import Image, ImageOps


def preprocess_image(contents):
    # 1. Open image
    img = Image.open(io.BytesIO(contents)).convert('RGBA')

    # 2. Create black background
    new_img = Image.new("RGBA", img.size, "black")
    new_img.paste(img, (0,0), img)

    # 3. Convert to grayscale
    img = new_img.convert('L')

    # 4. Find bounding box of white pixels
    img_np = np.array(img)
    coords = np.column_stack(np.where(img_np > 30))  # Threshold for white-ish pixels
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        img = img.crop((x_min, y_min, x_max, y_max))

    # 5. Dilate (fatten lines)
    img_np = np.array(img)
    kernel = np.ones((2, 2), np.uint8)  # 2x2 kernel
    img_np = cv2.dilate(img_np, kernel, iterations=1)

    # 6. Pad and Resize
    img = Image.fromarray(img_np)
    img = ImageOps.pad(img, (28, 28), color=0)

    # 7. Normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # (28,28,1)
    img_array = np.expand_dims(img_array, axis=0)   # (1,28,28,1)

    return img_array