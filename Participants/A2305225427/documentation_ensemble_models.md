## Celestial Vision — Ensemble Learning with ResNet Embeddings

### Overview
This project applies ensemble learning on deep image embeddings extracted using a pretrained ResNet50 network. Instead of training a CNN end-to-end, transfer learning is used to generate compact feature vectors, which are then classified using traditional machine learning models.

### Methodology
- Extracted 2048-dimensional embeddings from ResNet50.
- Trained three classifiers:
  - Logistic Regression
  - Support Vector Machine (RBF Kernel)
  - Random Forest
- Combined predictions using Soft Voting Ensemble.
- Compared individual and ensemble model performance.

### Observations
- SVM achieved the strongest individual performance.
- Random Forest struggled due to high dimensionality.
- Ensemble improved prediction stability and robustness.
- Soft voting provided smoother probability-based decisions.

### Conclusion
Ensembling multiple classifiers trained on ResNet embeddings improves classification reliability and overall performance, making it a strong approach for astronomical image classification.
