# 🧬 B-cell Antibody Valence Prediction with Multimodal Learning

This project explores a multimodal machine learning framework for predicting B-cell epitope valence, a critical component of immune response modeling and vaccine design. By integrating biological tabular features with transformer-based protein sequence embeddings (ProtBERT), we aim to improve prediction accuracy and uncover key structural markers of immune activity.

## 📌 Project Overview

Accurate prediction of B-cell epitopes—protein regions that trigger antibody production—is crucial in the development of targeted vaccines. We design and compare both interpretable and high-performance models to:
- Predict epitope valence using protein sequence data and biological covariates.
- Explore different embedding extraction methods using **ProtBERT**.
- Integrate attention mechanisms for biologically meaningful representation learning.
- Evaluate model fairness and performance under high dimensionality and class imbalance.

Our multimodal approach (ProtBERT + biological features) outperformed baseline tabular models, improving accuracy by 7.2%.

## 🧪 Dataset

- **Source**: [Kaggle Epitope Prediction Dataset](https://www.kaggle.com/datasets/futurecorporation/epitope-prediction)
- **Samples**: ~12,000 peptides
- **Features**:
  - 8 tabular biological covariates (e.g., antigenicity, hydrophobicity, surface accessibility)
  - Parent and peptide protein sequences
- **Label**: Binary classification (epitope-inducing vs. non-inducing)

## ⚙️ Methods

### 🔬 Embedding Engineering
- **Model**: [ProtBERT](https://huggingface.co/Rostlab/prot_bert)
- **Embedding strategies**:
  1. Parent-only sequence
  2. Peptide-only sequence ✅ (Best performance)
  3. Masked parent-peptide alignment
  4. Concatenated parent + peptide
  5. Subsequence attention
- **Attention layer**: Weighted pooling to focus on biologically significant residues

### 📉 Dimensionality Reduction
- **PCA** applied to 1024-d embeddings → 50 components explaining 92.4% variance
- Combined with 8 original tabular features

### 🤖 Classifiers Used
- **Tree-based**: CART, Random Forest, XGBoost ✅
- **Others**: Logistic Regression, Shallow Neural Networks, Optimal Classification Trees (OCT)
- **Metric Evaluation**: Accuracy, Precision, Recall, AUC, F1 Score

## 📊 Results

| Model Type     | Best Model   | AUC    | F1 Score | Accuracy |
|----------------|--------------|--------|----------|----------|
| Tabular Only   | XGBoost      | 0.81   | 0.75     | 78.3%    |
| Multimodal     | XGBoost ✅   | **0.8761** | **0.8059** | **85.5%** |
| Interpretation | OCT          | 0.79   | 0.74     | 76.9%    |

- Peptide-only embeddings with XGBoost performed best.
- Attention-based embedding + tabular fusion improved generalization.
- Interpretable models (e.g., OCT) offered good trade-offs for applications needing transparency.

## 📌 Key Takeaways

- Multimodal approaches improve predictive performance for immunological tasks.
- Sequence embeddings help capture long-range dependencies missed by tabular features.
- Attention and dimensionality reduction are crucial to balance complexity and interpretability.
- The framework can inform the design of safer, more effective vaccines.

## 🛠 Technologies

- Python, scikit-learn, pandas, seaborn
- ProtBERT (via HuggingFace)
- PyTorch (Neural network and attention layers)
- PCA, XGBoost, Fairlearn (for future fairness extension)

## 🧑‍🔬 Contributors

- **Sierra Gong**: Classification pipeline, model evaluation, interpretability
- **Phillip Nelson**: Embedding engineering, attention modeling, deep learning implementation

## 📈 Future Directions

- Integrate AlphaFold or ESM embeddings for structural insights
- Test additional peptide sequence masking and enhancement strategies
- Expand dataset and add infection-specific features
- Explore fairness-aware modeling to mitigate biases in biomedical prediction
