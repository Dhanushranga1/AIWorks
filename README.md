#  Medical Image Segmentation with U-Net

## Project Overview  
This project focuses on segmenting brain tumor regions from MRI scans using a U-Net model. The main goal was to build a working image segmentation project using Convolutional Neural Networks (CNNs), specifically U-Net.

---

## Objectives  
- Learn the basics of neural networks and CNNs  
- Understand U-Net architecture and how it applies to segmentation  
- Train a U-Net model on medical MRI images to predict tumor regions  
- Evaluate the model using standard segmentation metrics  
- Visualize the predicted outputs against ground truth masks

---

## What I Did  
- Learned neural network basics from Philip-Hua’s book and Sentdex tutorials  
- Built a U-Net model with encoder–bottleneck–decoder structure  
- Preprocessed `.tif` MRI images and masks  
- Trained the model with **128×128** input initially (worked fine in memory)  
- Moved to **256×256** for better segmentation — but Colab RAM crashed  
- Solved it by:
  - Using `train_test_split()` to split paths into train and validation  
  - Creating a **custom batch generator** to load data on-the-fly  
  - Generator handles **resizing, normalization, and mask binarization**  
- Trained for 20 epochs  
- Visualized predicted masks against actual ground truth

---

## U-Net Architecture (256×256 Input)

### Input
- Input size: **256 × 256 × 3**

### Encoder (Downsampling Path)
- **Block 1**  
  - Conv2D(64, 3×3, ReLU) → BatchNorm  
  - Conv2D(64, 3×3, ReLU) → BatchNorm  
  - MaxPooling2D  
- **Block 2**  
  - Conv2D(128, 3×3, ReLU) → BatchNorm  
  - Conv2D(128, 3×3, ReLU) → BatchNorm  
  - MaxPooling2D  
- **Block 3**  
  - Conv2D(256, 3×3, ReLU) → BatchNorm  
  - Conv2D(256, 3×3, ReLU) → BatchNorm  
  - MaxPooling2D  

### Bottleneck
- Conv2D(512, 3×3, ReLU) → BatchNorm  
- Conv2D(512, 3×3, ReLU) → BatchNorm  

### Decoder (Upsampling Path)
- **Block 1**  
  - Conv2DTranspose(256) → Concatenate with encoder Block 3  
  - 2×Conv2D(256, 3×3, ReLU) → BatchNorm  
- **Block 2**  
  - Conv2DTranspose(128) → Concatenate with encoder Block 2  
  - 2×Conv2D(128, 3×3, ReLU) → BatchNorm  
- **Block 3**  
  - Conv2DTranspose(64) → Concatenate with encoder Block 1  
  - 2×Conv2D(64, 3×3, ReLU) → BatchNorm  

### Output Layer
- Conv2D(1, 1×1, sigmoid) → Binary mask prediction

---

## Model Details  
| Parameter      | Value            |
|----------------|------------------|
| Loss Function  | Binary Crossentropy + Dice Loss |
| Optimizer      | Adam             |
| Batch Size     | 8                |
| Epochs         | 20               |
| Input Size     | 256×256          |

---

##  Results

### Current (256×256 Model)  
- **Mean Dice Coefficient:** `0.7409`  
- **Mean IoU Score:** `0.7097`

###  Previous (128×128 Model)  
- **Mean Dice Coefficient:** `0.6580`  
- **Mean IoU Score:** `0.4903`

### Improvement Summary
___________________________________________________
| Metric        | 128×128 | 256×256 | Improvement |
|---------------|---------|---------|-------------|
| Dice Score    | 0.6580  | 0.7409  | +0.0829     |
| IoU Score     | 0.4903  | 0.7097  | +0.2194     |
___________________________________________________
By increasing the input resolution and loading data in batches, the model was able to capture more spatial detail and generalize better.

---

## Problems Faced  
- At 128×128 size, loading all images into memory worked fine  
- But 256×256 caused **Colab to crash** due to RAM overload  
- Fixed it by:
  - **Splitting data paths** into train/val using `train_test_split()`  
  - Creating a **custom Keras generator** that loads images/masks in batches  
  - Generator resizes, normalizes, and binarizes masks on the fly  
  - Helped save memory and made training stable

---

## Improvements Made  
- Used batch-wise loading to avoid memory crashes  
- Trained on higher resolution input (256x256)  
- Added dropout layers for better generalization  
- Visualized predictions and compared them with ground truth masks

---

## Deliverables  
- Trained U-Net model (.h5)  
- Full Jupyter Notebook with:
  - Preprocessing  
  - Custom generator  
  - Model building and training  
  - Evaluation metrics  
  - Visualization  
- Comparison between low and high resolution models  
- Clear performance metrics (Dice & IoU)

---

