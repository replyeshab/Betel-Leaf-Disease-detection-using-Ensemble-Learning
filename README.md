# Betel Leaf Disease Classification Using Transfer Learning

![Python](https://img.shields.io/badge/Python-ML%20Project-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Task](https://img.shields.io/badge/Task-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-lightgrey)

A computer vision project exploring deep learning and transfer learning for betel leaf disease classification. The project compares three pretrained CNN architectures and combines their predictions using a hard-voting ensemble.

**Project presentation:** [View the Canva presentation](https://canva.link/u2yt6t20hjkok2g)

## Overview

Plant diseases can affect crop quality and agricultural productivity. This project explores the use of image classification to identify visible conditions in betel leaves using deep learning.

The main objective is to compare pretrained convolutional neural networks, evaluate their classification performance, and investigate whether combining their predictions provides a useful alternative to relying on one model.

### Key features

- Transfer learning using pretrained CNN architectures
- Comparison of three individual models
- Hard-voting ensemble for combining predictions
- Evaluation using accuracy, classification reports, and confusion matrices
- Prediction and evaluation utilities separated from the original notebook workflow

## Models

The project investigates the following architectures:

| Model | Approach |
|---|---|
| MobileNetV2 | Transfer learning for image classification |
| NASNetMobile | Transfer learning for image classification |
| EfficientNetV2-S | Transfer learning for image classification |
| Hard-voting ensemble | Combines predictions from the three models |

The ensemble selects the class receiving the most votes from the individual models.

## Methodology

The original project workflow includes:

1. Loading and preparing betel leaf images.
2. Resizing input images to 224 × 224 pixels.
3. Applying data augmentation during training.
4. Training or fine-tuning pretrained CNN architectures.
5. Evaluating individual model predictions.
6. Combining predicted classes through hard voting.
7. Comparing model performance using classification metrics and confusion matrices.

The preprocessing used during inference must match the preprocessing used to train each saved model.

## Dataset

**Dataset reference:** [Betel Leaf Image Dataset from Bangladesh — Mendeley Data, Version 2](https://data.mendeley.com/datasets/g7fpgj57wc/2) ,  [Comprehensive Betel Leaf Disease Dataset for Advanced Pathology Research
](https://data.mendeley.com/datasets/vpzkntzjty/1) 

The published dataset describes four conditions:

- Healthy leaf
- Dried leaf
- Bacterial leaf disease
- Fungal brown spot disease

The current notebook and saved-model interface use three labels:

`Healthy_Leaf`, `Leaf_Rot`, and `Leaf_Spot`

The original project presentation also describes a collection of 10,060 images obtained from two websites. The exact relationship between that collection, the linked dataset, augmentation, and the three output labels must be verified before claiming that the reported results were obtained from the unmodified Mendeley dataset.

**Dataset attribution:** Rashid, M. R. A., et al. (2024). *Betel Leaf Image Dataset from Bangladesh*, Version 2. Mendeley Data. https://doi.org/10.17632/g7fpgj57wc.2

The dataset is published under a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). Please refer to the original dataset page for attribution and licensing requirements.

## Results

The following figures are transcribed from the original project presentation. They are **reported results, not independently revalidated results from this repository**.

| Model | Reported test accuracy |
|---|---:|
| EfficientNetV2-S | 64.69% |
| NASNetMobile | 96.94% |
| MobileNetV2 | 95.21% |
| Hard-voting ensemble | 95.48%* |

\*The ensemble accuracy shown in the presentation does not match the accuracy implied by its displayed confusion matrix. The test-set evaluation should be rerun using the same fixed test examples before these figures are treated as final.


## Getting Started

### Requirements

- Python 3.10 or 3.11
- TensorFlow/Keras
- NumPy
- scikit-learn
- Pillow

### Installation

Clone the repository:

```bash
git clone https://github.com/replyeshab/Betel-Leaf-Disease-detection-using-Ensemble-Learning
.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running Inference

The inference utility is designed to use existing trained model files; it does not need to train the models again.

Place your saved model files on your local machine:

- `mobilenet_model.h5`
- `final_nasnet.h5`
- `final_effnetv2.h5`

Then use the inference script with a leaf image and the correct paths to your models. Check the class-index order and preprocessing settings against the original training run before relying on predictions.

Model weights and dataset files are intentionally excluded from version control. See the `models/README.md` file for details.

## Demo and Presentation

[Open the project presentation in Canva](https://canva.link/u2yt6t20hjkok2g)

A recorded demo can be added here after exporting the video from Canva. A useful demo should show an input image, the individual model predictions, the ensemble decision, and the verified evaluation results.

## Limitations

- The existing model interface uses three output classes, while the linked Mendeley dataset describes four.
- Dataset provenance and the exact class mapping need to be documented.
- The reported ensemble accuracy and confusion matrix are inconsistent.
- Performance on a curated test set does not establish reliability on all real-world leaf images.
- The model is an experimental classification tool, not a substitute for expert agricultural diagnosis.

## Future Improvements

- Validate all existing models on one fixed, reproducible test set.
- Regenerate the metrics table and confusion matrices directly from evaluation outputs.
- Document the exact dataset split, class mapping, and preprocessing configuration.
- Add a simple web interface for uploading a leaf image.
- Add a recorded demo and example predictions.
- Improve experiment tracking and model-version documentation.

## Acknowledgements

Thanks to the authors of the [Betel Leaf Image Dataset from Bangladesh](https://data.mendeley.com/datasets/g7fpgj57wc/2) and  [Comprehensive Betel Leaf Disease Dataset for Advanced Pathology Research](https://data.mendeley.com/datasets/vpzkntzjty/1)for making the dataset available for research.
