import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import UNET
import subprocess

subprocess.run(["pip", "install", "gdown"])
# https://drive.google.com/file/d/1CMpyaER9Sw0ZhsCguyc2P9Kr302n7bbt/view?usp=sharing
subprocess.run(["gdown", "https://drive.google.com/uc?id=1CMpyaER9Sw0ZhsCguyc2P9Kr302n7bbt"])

img_path = "tiny_coco/train2017/000000086408.jpg"
col_mask_path = "colored_masks_train2017/000000086408_mask_colored.png"
model_path = "model_final.pth"
input_size = (256, 256)
num_classes = 91
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
model = UNET(out_channel=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Load Image ---
orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(orig_img, input_size)

mask_img = cv2.imread(col_mask_path)
mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

img_tensor = torch.tensor(resized_img / 255.0, dtype=torch.float).permute(2, 0, 1).unsqueeze(0).to(device)

# --- Predict ---
with torch.no_grad():
    pred = model(img_tensor)
    pred = F.interpolate(pred, size=orig_img.shape[:2], mode="bilinear", align_corners=False)
    pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

# --- Colorize Output ---
def colorize_mask(mask):
    from matplotlib import cm
    unique_ids = sorted(np.unique(mask))
    colormap = (cm.get_cmap("tab20c", 256)(np.arange(256))[:, :3] * 255).astype(np.uint8)
    id_to_color = {cid: colormap[i % len(colormap)] for i, cid in enumerate(unique_ids)}

    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid in unique_ids:
        color_mask[mask == cid] = id_to_color[cid]
    return color_mask

print("unique classes in mask  ",np.unique(pred_mask))
print("pred_mask shape ",pred_mask.shape)
colored_pred = colorize_mask(pred_mask)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Your original mask (2D numpy array of shape [427, 640], values from 0 to 90)
mask = np.array(pred_mask)  # replace with your mask variable if different

# Create a color map: 91 distinct colors for 91 classes
def get_colormap(num_classes=91):
    np.random.seed(42)  # to keep colors consistent
    return np.random.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)

colormap = get_colormap()

# Map each class index in the mask to an RGB color
rgb_mask = colormap[mask]  # shape will be (427, 640, 3)

# If you want to visualize
# plt.imshow(rgb_mask)
# plt.axis('off')
# plt.title("RGB Mask Visualization")
# plt.show()
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(orig_img); plt.title("Training Image")
plt.subplot(1, 3, 2); plt.imshow(mask_img); plt.title("Actual Mask")
plt.subplot(1, 3, 3); plt.imshow(rgb_mask); plt.title("Predicted Mask")
plt.show()
# Or save it as an image
# Image.fromarray(rgb_mask).save("colored_mask.png")
