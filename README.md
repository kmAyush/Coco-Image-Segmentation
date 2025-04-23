

# Part 1 : COCO Multi-Class Segmentation Mask Generator

This script processes COCO images and annotations to generate multi-class masks with pixel values representing `category_id`.

## Features
- Supports 3000 training and 600 validation images.
- Creates grayscale and multi-class mask using standard Coco annotation.
- Output: grayscale `.png` masks with pixel = category ID.

## Usage
```bash
pip install -r requirements.txt
python inference.py
```
