# Ensemble Models Using ResNet Embeddings

## Overview
This project explores ensemble learning using image embeddings extracted from a pretrained
ResNet-18 model. Multiple classifiers are trained on the embeddings and combined using
a soft-voting ensemble strategy.

## Methodology
- Pretrained ResNet-18 is used as a fixed feature extractor
- Image embeddings are extracted from CIFAR-10 images
- Logistic Regression, SVM, and MLP classifiers are trained
- Predictions are combined using soft voting

## Results
| Model | Accuracy |
|------|----------|
| Logistic Regression | 0.792 |
| SVM | 0.8147 |
| MLP | 0.8054 |
| **Ensemble** | **0.82** |

## Conclusion
The ensemble model consistently outperforms individual classifiers, demonstrating
improved robustness and generalization when combining multiple learners on
ResNet-based embeddings.