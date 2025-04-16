# Final Report: Segmentation Model Training on COCO Subset

## Model Architecture
- **U-Net with ResNet34 encoder**
- Trained with CrossEntropyLoss and Adam optimizer
- Input size: 256x256

## Decisions
- Used U-Net for its efficiency and good generalization on small datasets.
- Pretrained encoder to reduce training time.
- Multi-class segmentation with 91 categories.

## Results
- Trained for 40 epochs (~1.5 hrs on 1 GPU 4090 RTX Nvidia)
- Validation accuracy: ~72%
- Model converged stably using Adam and lr=1e-3

## Sample Predictions

| Input Image | Ground Truth | Prediction |
|-------------|--------------|------------|
| ![](examples/img1.jpg) | ![](examples/mask1.png) | ![](examples/pred1.png) |
| ![](examples/img2.jpg) | ![](examples/mask2.png) | ![](examples/pred2.png) |

## Tools Used
- PyTorch

## Reproducibility

```bash
uv venv
uv pip install -r requirements.txt
python scripts/train.py
