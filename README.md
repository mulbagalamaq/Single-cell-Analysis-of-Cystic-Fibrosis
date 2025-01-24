
# Single-cell RNA-seq Analysis of Cystic Fibrosis

This repository contains the code and analysis for a single-cell RNA-seq study comparing cystic fibrosis (CF) and normal cell types. The project leverages K-nearest neighbors (KNN) algorithms, differential expression analysis, and visualization techniques to identify key differences in gene expression and cell type behavior.

---

## About the Project

### Background
Cystic fibrosis is a genetic disorder that affects the lungs and other organs. This project aims to analyze single-cell RNA-seq data to identify cell types and tissues most affected by CF. By comparing CF and normal samples, we can uncover molecular mechanisms underlying disease progression and potential therapeutic targets.

### Key Objectives
1. **Data Integration**: Integrate gene expression and protein data from CITE-seq.
2. **Machine Learning**: Apply KNN clustering and classification to identify CF-affected cell types.
3. **Differential Expression**: Identify differentially expressed genes (DEGs) and protein-gene correlations.
4. **Visualization**: Generate publication-ready plots, including dot plots and bar graphs, to summarize findings.


---

## Key Features

### 1. **Data Preprocessing**
- Load and preprocess single-cell RNA-seq data from `.npz` and `.csv` files.
- Extract counts and metadata for CF and normal samples.
- Normalize and transpose count matrices for downstream analysis.

### 2. **KNN Analysis**
- Determine the optimal number of neighbors (`k`) using cross-validation.
- Compute distances between CF and normal samples.
- Normalize distances for comparison across cell types and tissues.

### 3. **Differential Expression**
- Identify differentially expressed genes (DEGs) using Wilcoxon rank-sum tests.
- Compute protein-gene correlations to uncover regulatory relationships.

### 4. **Visualization**
- **Dot Plot**: Highlight most and least affected cell types based on normalized distances.
- **Bar Graph**: Compare mean differences in CF and normal tissues.

---

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `seaborn`, `matplotlib`, `scanpy`

## Results

### Key Findings
- **Most Affected Cell Types**: Identified top 5 cell types with the highest normalized distances, indicating significant differences in CF samples.
- **Tissue Differences**: Lung and respiratory airway tissues showed the highest mean differences between CF and normal samples.
- **Protein-Gene Correlations**: Uncovered strong correlations between specific proteins and genes, suggesting potential regulatory mechanisms.

### Visualizations
- **Dot Plot**: Visualizes relative distances between CF and normal cell types.

---
### License
This project is licensed under the MIT License. 