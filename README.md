# Hidden in Plain Sight: Vector Embeddings give away Demographic Information

This repository investigates **Fairness in vector embeddings** derived from **chest X-ray images**.  
We study fairness in embeddings generated from three different **CLIP-based models** for medical imaging:

1. **CXR Foundation**  
2. **MedCLIP**  
3. **BiomedCLIP**

For each embedding type, we train and evaluate **multiple models**:  
- **Multi-Layer Perceptron (MLP)**  
- **Logistic Regression (LR)**  
- **Random Forest (RF)**  
- **XGBoost (XGB)**  

The repository includes scripts for processing embeddings, training, and testing models on both the **MIMIC-CXR** and **CheXpert** datasets.

---

## Prerequisites

- **Python 3.8**
- **Conda** (for environment management)

### Installation

1. Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Clone the **Google Health imaging research** repository (if needed):
    ```bash
    git clone https://github.com/Google-Health/imaging-research.git
    ```

3. Create and activate a **Conda environment**:
    ```bash
    conda create --name fairness-ai-env python=3.8
    conda activate fairness-ai-env
    ```

4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Running the Scripts**

### **1. Data Processing (MIMIC-CXR)**
Before training models, you must process the **MIMIC-CXR** dataset:

1. Extract embeddings and demographic metadata:
    ```bash
    python src/extraction_metadata_embedding_mimic_cxr.py
    ```

2. Apply a new encoding for **insurance data**:
    ```bash
    python src/extraction_new_insurance.py
    ```

---

## **2. Training & Testing Models on MIMIC-CXR**
This section covers training/testing models using the three embedding types.

### **CXR Foundation (MIMIC-CXR)**
1. **Train models** on CXR Foundation embeddings:
    ```bash
    python src/train_cxr-foundation_mimic.py
    ```

2. **Test models** on CXR Foundation embeddings:
    ```bash
    python src/test_cxr-foundation_mimic.py
    ```

### **MedCLIP (MIMIC-CXR)**
1. **Train models** on MedCLIP embeddings:
    ```bash
    python src/train_medclip_mimic.py
    ```

2. **Test models** on MedCLIP embeddings:
    ```bash
    python src/test_medclip_mimic.py
    ```

### **BiomedCLIP (MIMIC-CXR)**
1. **Train models** on BiomedCLIP embeddings:
    ```bash
    python src/train_biomedclip_mimic.py
    ```

2. **Test models** on BiomedCLIP embeddings:
    ```bash
    python src/test_biomedclip_mimic.py
    ```

---

## **3. Training & Testing Models on CheXpert**
The same analysis is conducted on the **CheXpert dataset**.

### **Preprocessing CheXpert Data**
1. **Preprocess the CheXpert metadata**:
    ```bash
    python src/preprocess_chexpert.ipynb
    ```

### **CXR Foundation (CheXpert)**
1. **Train models** on CXR Foundation embeddings:
    ```bash
    python src/train_cxr-foundation_chexpert.py
    ```

2. **Test models** on CXR Foundation embeddings:
    ```bash
    python src/test_cxr-foundation_chexpert.py
    ```

### **MedCLIP (CheXpert)**
1. **Train models** on MedCLIP embeddings:
    ```bash
    python src/train_medclip_chexpert.py
    ```

2. **Test models** on MedCLIP embeddings:
    ```bash
    python src/test_medclip_chexpert.py
    ```

### **BiomedCLIP (CheXpert)**
1. **Train models** on BiomedCLIP embeddings:
    ```bash
    python src/train_biomedclip_chexpert.py
    ```

2. **Test models** on BiomedCLIP embeddings:
    ```bash
    python src/test_biomedclip_chexpert.py
    ```

---

## **4. Results**
- **Processed embeddings and metadata** are stored in the `./data` directory.
- **Plots and performance metrics** (accuracy, precision, recall, F1-score, ROC-AUC) are saved in the `./fig` directory.
- Model performance across **different embeddings and datasets** is compared to evaluate fairness.

---

## **License**
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## **Acknowledgments**
- **MIMIC-CXR dataset** provided by the **MIT Laboratory for Computational Physiology**.
- **CheXpert dataset** from Stanford University.
- **Google Health Imaging Research** repository for additional resources and medical imaging tools.

---
