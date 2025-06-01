# Medical Image Segmentation with U-Net

## Project Overview  
This project focuses on segmenting brain tumor regions from MRI scans using a U-Net model. The main goal was to build a basic but working image segmentation pipeline using Convolutional Neural Networks (CNNs), specifically U-Net.

## Objectives  
- Learn the basics of neural networks and CNNs  
- Understand U-Net architecture and how it applies to segmentation  
- Train a U-Net model on medical MRI images to predict tumor regions  
- Evaluate the model using standard segmentation metrics  
- Visualize the predicted outputs against ground truth masks

## What I Did  
- Learned neural network fundamentals from Philip-Hua's book and YouTube (Sentdex)  
- Explored CNN components like convolutions, ReLU, pooling, and upsampling  
- Implemented a basic U-Net architecture with encoder, bottleneck, and decoder  
- Preprocessed `.tif` images and masks, resized them to 128x128  
- Trained the model on the dataset for 20 epochs  
- Evaluated using Dice and IoU  
- Visualized predictions to compare actual vs predicted masks

## U-Net Architecture (Simplified)  
- **Input size:** 128x128x3  
- **Encoder:**  
  - 2 Conv2D layers → MaxPooling → Dropout (64 to 128 filters)  
- **Bottleneck:**  
  - 2 Conv2D layers with 256 filters  
- **Decoder:**  
  - Conv2DTranspose for upsampling  
  - Skip connections from encoder  
  - Dropout and Conv2D layers (128 to 64 filters)  
- **Output:**  
  - Conv2D(1, 1, activation='sigmoid') for binary segmentation

## Model Details  
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Batch size:** 8  
- **Epochs:** 20  
- **Image size:** 128x128  

## Results  
- **Dice Coefficient:** `0.6580`  
- **IoU Score:** `0.4903`  

Some predictions were very close to the ground truth, but for a few complex cases the model failed to detect any tumor regions. Performance was reasonable considering the simple architecture and low image resolution.

## Improvements Possible  
- Increasing the number of epochs (comes at the cost of training time)  
- Adding more Conv layers or deeper encoder-decoder blocks  
- Using batch normalization and dropout more strategically  
- Trying data augmentation to improve generalization  

## Deliverables  
- Trained U-Net model  
- Jupyter Notebook containing the full workflow (preprocessing, training, evaluation)  
- Segmentation result visualizations  
- Dice and IoU evaluation metrics
