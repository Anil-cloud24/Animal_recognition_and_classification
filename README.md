# 🐾 Animal10 Classification using CNN
This repository contains a **Convolutional Neural Network (CNN) implementation in PyTorch to classify images into 10 different animal classes from the Animals-10 dataset**.
The model is trained with image augmentations, early stopping, and accuracy/loss visualization.

---

## 📌 Features
- Data preprocessing & augmentation using torchvision.transforms.
- CNN architecture with 4 convolutional layers, dropout, and ReLU activations.
- Custom training loop with early stopping.
- GPU support with an easy-to-use device loader.
- Accuracy and loss visualization over epochs.

---

## 📂 Dataset

The dataset should be structured as follows:
```text
animals10/raw-img/
    ├── butterfly/
    ├── cat/
    ├── chicken/
    ├── cow/
    ├── dog/
    ├── elephant/
    ├── horse/
    ├── sheep/
    ├── spider/
    └── squirrel/
```

You can download it from Kaggle:
Animals-10 Dataset

---

## ⚙️ Installation

### Clone the repository:
git clone https://github.com/Anil-Venkat-Venkatachalam/animal10-cnn.git
cd animal10-cnn


### Install dependencies:
pip install torch torchvision matplotlib numpy


### Place the dataset in the correct folder:
/kaggle/input/animals10/raw-img

---

## 🚀 Usage
Run the training script:
The script:
- Loads & augments the dataset.
- Splits into training and validation sets.
- Trains the CNN model.
- Plots accuracy and loss curves.

---

## 🏗 Model Architecture
- Conv2d(3, 32) → ReLU → MaxPool2d
- Conv2d(32, 64) → ReLU → MaxPool2d
- Conv2d(64, 128) → ReLU → MaxPool2d
- Conv2d(128, 128) → ReLU → MaxPool2d
- Flatten → Dropout(0.5)
- Linear(128*8*8 → 256) → ReLU → Dropout(0.5)
- Linear(256 → 10)

---

## 📊 Results
- Accuracy vs Epochs
- Loss vs Epochs

---

## 💡 Improvements
- Experiment with deeper architectures (e.g., ResNet, EfficientNet).
- Add learning rate scheduling.
- Try transfer learning for faster convergence.

---
