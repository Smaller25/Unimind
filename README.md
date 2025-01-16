# Unimind
Unimind: cross-subject fMRI Image Reconstruction. 


# Cross-subject fMRI Image Reconstruction using Contrastive Learning

This repository contains the implementation of a cross-subject fMRI image reconstruction framework based on contrastive learning. The project focuses on decoding and reconstructing images from fMRI signals using a generalizable and efficient deep learning architecture.

## Overview
<img src="/asset/task_explain.png" width="700" height="600"/>

Traditional single-subject fMRI decoding models face limitations such as:
- High computational cost for training separate models per subject
- Scarcity of large datasets due to prolonged recording sessions
- Poor generalizability to unseen subjects

This project addresses these challenges by introducing a cross-subject learning framework that aligns latent semantic spaces across multiple subjects.

## Key Features
<img src="/asset/new_pipeline.jpg" width="600" height="500"/>
- **Contrastive Learning with Siamese Networks**: Aligns semantic representations across subjects for better generalization.
- **ConvMixer Encoder**: Lightweight convolutional architecture for faster training and efficient feature extraction.
- **Double-Conditioned Latent Diffusion Model (DC-LDM)**: Enhances decoding consistency while maintaining semantic fidelity.
- **Cross-Subject Adaptation**: Enables the model to work on diverse subjects with minimal fine-tuning.



## Methodology
1. Pretraining: The fMRI encoder-decoder is pretrained on a large-scale unpaired dataset (e.g., Human Connectome Project).
2. Contrastive Learning: A Siamese network aligns fMRI embeddings from multiple subjects using a CLIP-based pseudo-labeling approach.
3. Image Reconstruction: DC-LDM generates high-fidelity images by leveraging fMRI features and temporal embeddings.

## Results
- Improved low-level (e.g., SSIM, MSE) and high-level (e.g., LPIPS, Inception Score) performance on benchmark datasets (e.g., NSD, GOD).
- Efficient memory usage and parameter reduction compared to transformer-based models.

## Applications
- **Neuroimaging Research**: Improved understanding of brain functions through non-invasive analysis.
- **Healthcare**: Assisting patients with communication impairments.
- **AI and Cognitive Science**: Bridging human perception and artificial intelligence.

## Usage
To replicate the results or extend this framework, follow the steps below:
1. Clone this repository.
2. Prepare the fMRI datasets (e.g., NSD, GOD).
3. Train the model using the provided training scripts.
4. Evaluate the model on test datasets using the evaluation metrics provided.

## Future Work
- Explore time-series fMRI signal analysis for video reconstruction.
- Integrate semantic information from non-visual brain regions to reconstruct more abstract information (e.g., emotions).
- Develop additional cross-subject learning methodologies to enhance robustness.

## Citation
If you use this code, please cite the original thesis:
Sohyung Kim, Cross-subject fMRI Image Reconstruction using Contrastive Learning. Seoul National University, February 2025. (exact reference will be updated soon)
