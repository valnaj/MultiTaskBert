# Multi-Task Text Classification Model for Indicator Sets [UC3]
 
This project involves a multi-task text classification model designed to predict multiple labels for text input columns. It leverages BERT for feature extraction and has separate classification heads for each label. The model is trained, fine-tuned, and used for inference with the following detailed steps and files.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Files and Directories](#files-and-directories)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Fine-Tuning](#model-training-and-fine-tuning)
6. [Inference](#inference)
7. [Retraining the Model](#retraining-the-model)
8. [Technical Sustainability](#technical-sustainability)
9. [Contact Information](#contact-information)

## Introduction

This repository contains the implementation of a multi-task BERT-based text classification model. The model is trained to predict multiple labels for given indicator statements and definitions. It handles tasks such as direction of change, value format, unit of measurement, subject, sector, and more.

## Requirements

The required Python packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Files and Directories

### Data

Not available as this project has been anonymized to follow the client's requirements.

### Label Encoders

- `label_encoders/`: Directory containing label encoders for each target column.

### Saved Model(s)

- `saved model/uc3_model.ckpt` Latest trained model should be stored for inference. (not available here)

### Other files

- `num_classes_list.pkl`: Contains the number of classes for each target column.
- `preprocessed_dataset.csv`: Preprocessed dataset (used by the inference script).
- `tokenized_dataset.pkl`: BERT-Tokenized dataset.

## Data Preprocessing
(notebook: UC3-multitask-classification-fine-tuning.ipynb)

Data preprocessing is handled in the notebook UC3-multitask-classification-fine-tuning.ipynb. The main steps include:

1. Loading the dataset and dropping rows with NaN in 'feature_1'.
2. Lowercasing text in relevant columns.
3. Combining relevant text columns into a single feature.
4. Tokenizing the combined features using BERT tokenizer.
5. Label encoding the target columns.
6. Saving the preprocessed dataset and label encoders.

## Model Training and Fine-Tuning 
(notebook: `UC3-multitask-classification-fine-tuning.ipynb`)

The training and fine-tuning are performed using PyTorch Lightning:

- Model Architecture: The MultiTaskBERT class includes a BERT backbone for feature extraction and multiple classification heads for each target label. Each head is a linear layer that outputs logits for a specific classification task (see diagram)

![Model Architecture](diagram/UC3%20Model.png)
 
- **Custom Loss Function**: The loss function used is a combination of cross-entropy losses for each classification head. Each task-specific loss is computed separately and then summed to obtain the total loss.
- **Metrics**: The primary metric used for evaluation is the weighted sum of accuracies across all tasks. This metric computes the accuracy for each task and then sums these accuracies, taking into account the number of classes in each task (weighted), to provide an overall performance measure.
- **Optimizer**: The optimizer used is AdamW, with a learning rate determined through hyperparameter optimization.
- **Training Process**: The training process involves monitoring validation accuracy and early stopping based on validation loss to prevent overfitting.
- **Hyperparameter Optimization**: Conducted using Optuna (best of 50 trials of 10 epochs each).

The notebook includes the steps for loading data, defining the model, training, and evaluating performance:

1. **Data preparation**: Load and preprocess the dataset. Tokenize the text using BERT tokenizer. Encode target labels using label encoders.
2. **Model definition** (see notebook for class MultiTaskBERT)
3. **Hyperparameter Optimization**: Use Optuna to find the best hyperparameters such as learning rate, batch size, dropout rate, and hidden size.
4. **Training and Validation**: Train the model with early stopping and checkpointing to save the best model based on validation accuracy.
5. Evaluate the model's performance on the validation set using weighted accuracy metrics.

## Inference 
(script: `UC3-script.py`)

The script used for inference includes:

- Loading the pre-trained model.
- Preprocessing the input data.
- Performing predictions.
- Saving the results to an output file.

## Retraining the model

*Note: A GPU is required to train the model. The script can be run on CPU (using the pre-trained model) if needed but it will take 2-3 minutes to run.*

### Case 1: When new values are added and we wish to retrain the model (same number of tasks/heads/columns).

1. **Update the Dataset**: Add the new value to the relevant column in the dataset.
2. **Label Encoding**: Ensure the new value is included in the label encoders by re-fitting them.
3. **Optional**: Fine tune the model: Only if needed
4. **Retrain the model**: Follow the same steps in the training notebook to retrain the model with the updated dataset and label encoders (make sure that the labels are the number of classes match)
5. **Update hyperparameters** (if needed): Update the hyperparameters used in the script if the model was retuned (optional only if retuning makes senes as its easier to just retrain the model with the same previous hyperparameters)

### Case 2: When a New Column/Task is Added (e.g. new target)

1. **Update the Dataset and Label Encoders**:
   - Add the new column to the dataset.
   - Make sure to update the `target_columns` list to add the new target in both the script and training notebook.
   - Create and fit a new label encoder for the new column.

2. **Retrain the model**:
   - The model architecture dynamically adjusts the number of classification heads based on the num_classes_list (in the `MultiTaskBERT` class).
   - Retrain the model with the updated dataset, including the new category.
  
## Technical sustainability

### Options for Using Online Hosted GPUs
**Google Colab**: Offers free and paid options for accessing GPUs. It's suitable for small to medium-sized projects.
- Pros: Free tier available, easy to set up, integrates with Google Drive.
- Cons: Limited usage time on the free tier, potential disconnections.
- Usage: Upload your notebook and data to Google Colab, and start training.

**Kaggle Kernels**: Another free option for using GPUs, suitable for smaller datasets and projects.
- Pros: Free to use, integrates with Kaggle datasets and competitions.
- Cons: Limited GPU availability and runtime.
- Usage: Create a new notebook on Kaggle (Reuse the notebook), upload your data, and start training.

Since the amount of data is relatively low, free GPUs should be enough to run training. 
Getting hardware or using other paid compute will be more expensive but more reliable.
Using CPU as compute will work fine for inference but I would not recommend using it for training.

  
## Contact Information

For any questions or concerns, please contact:
- Name: [Valentin Najean]
- Email: [najeanvalentin@gmail.com]
- GitHub: https://github.com/valnaj
