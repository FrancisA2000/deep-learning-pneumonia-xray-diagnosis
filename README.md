# Deep Learning Pneumonia X-Ray Diagnosis 🧠🩻  
### AI-Powered Pneumonia Detection from Chest X-Rays using Deep Convolutional Neural Networks

> **Final Project – Deep Learning Systems (31245)**  
> **Braude College of Engineering, 2025**  
> **Course:** מערכות לומדות ולמידה עמוקה (Deep Learning Systems)  
>  
> This repository presents an **AI-driven Deep Learning framework** for **pneumonia diagnosis** from **chest X-ray images**,  
> developed as the final project for the Deep Learning Systems course.  
> It explores **Convolutional Neural Networks (CNNs)**, **Transfer Learning (ResNet152V2)**,  
> **optimizer and learning-rate tuning**, **early stopping**, and **multi-class classification** (Normal, Bacterial, Viral).

[![Python](https://img.shields.io/badge/Python-99.3%25-blue.svg)](https://github.com/FrancisA2000/deep-learning-pneumonia-xray-diagnosis)
[![MATLAB](https://img.shields.io/badge/MATLAB-0.7%25-orange.svg)](https://github.com/FrancisA2000/deep-learning-pneumonia-xray-diagnosis)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧩 Project Overview

- **Objective:** Use Deep Learning to automate pneumonia detection and classification from X-ray images.  
- **Dataset:** Chest X-ray Pneumonia dataset (≈5,863 images) labeled by medical professionals.  
- **Architecture:**  
  - Custom-built CNN (trained from scratch)  
  - Transfer Learning using **ResNet152V2** (frozen + fine-tuned variants)  
- **Tasks Included:**
  1. **Task 1:** CNN & ResNet architectures (frozen + fine-tuned).  
  2. **Task 2:** Model training, PR curves, F1-score threshold optimization (0.10 → 0.90).  
  3. **Task 3:** Optimizer and learning-rate tuning + EarlyStopping evaluation.  
  4. **Task 4:** Multi-class classification (Normal / Bacterial / Viral).  
- **Benchmark goal:** ≥93% accuracy on test set (from project brief).

---

## 🗂️ Repository Structure

```
deep-learning-pneumonia-xray-diagnosis/
├── DEEP LEARNING - Final Project/
│   ├── Task1.py                              # CNN & ResNet152V2 implementation
│   ├── Task2.py                              # PR curves & F1 threshold optimization
│   ├── Task3.py                              # Optimizer/LR tuning & EarlyStopping
│   ├── Task4.py                              # 3-class classification (Normal/Bacterial/Viral)
│   ├── images/                               # Generated plots and visualizations
│   │   ├── Task1/                            # Architecture tables, sample images
│   │   ├── Task2/                            # Training curves, PR analysis
│   │   ├── Task3/                            # Optimizer comparisons, EarlyStopping
│   │   └── Task4/                            # Multi-class results, confusion matrices
│   ├── deep-learning project report.pdf     # Complete project documentation
│   ├── deep-learning project report.docx    # Editable report version
│   └── פרוייקט מערכות לומדות למידה עמוקה.pdf  # Project brief (Hebrew)
│
├── Lab1-Lab10/                               # Course lab exercises
│   ├── Lab1/ Lab2/                           # Classical ML (KNN, Logistic Regression, SVM)
│   ├── Lab3/ Lab4/ Lab5/ Lab6/              # MLPs, Backpropagation, SGD
│   ├── Lab7/ Lab8/                          # CNNs & Transfer Learning
│   ├── Lab9/                                # Optimizers & Regularization
│   └── Lab10/                               # RNNs / Sequence Models & EarlyStopping
│
├── Lab 1 report.docx                        # Lab 1 documentation
├── Lab 2 report.docx                        # Lab 2 documentation
├── מערכות לומדות ולמידה עמוקה_סילבוס_2025.pdf  # Course syllabus
├── LICENSE                                   # MIT License
└── README.md                                 # This file
```

---

## ⚙️ Setup & Usage

### 🔧 Environment Requirements
- **Python:** ≥ 3.8  
- **Key Dependencies:**
  ```bash
  tensorflow>=2.8.0
  keras>=2.8.0
  scikit-learn>=1.0.0
  numpy>=1.21.0
  pandas>=1.3.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  pillow>=8.0.0
  ```

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/FrancisA2000/deep-learning-pneumonia-xray-diagnosis.git
cd deep-learning-pneumonia-xray-diagnosis

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 🚀 Running the Tasks

Navigate to the project folder and execute individual tasks:

```bash
cd "DEEP LEARNING - Final Project"

# Task 1: CNN and ResNet152V2 architecture comparison
python Task1.py

# Task 2: Precision-Recall analysis and F1 threshold optimization
python Task2.py

# Task 3: Optimizer and learning rate experiments with EarlyStopping
python Task3.py

# Task 4: Multi-class classification (Normal/Bacterial/Viral)
python Task4.py
```

Each script will:
- Load and preprocess the X-ray dataset
- Train the specified model(s)
- Generate evaluation metrics and visualizations
- Save results to the `images/` folder

---

## 📊 Key Results

| Model                        | Best Metric      | Threshold | Notes                                    |
|------------------------------|------------------|-----------|------------------------------------------|
| Custom CNN (no TL)           | F1 ≈ 0.865       | 0.85      | Baseline model trained from scratch      |
| ResNet152V2 (frozen)         | F1 ≈ 0.882       | 0.90      | Transfer learning without fine-tuning    |
| **ResNet152V2 (fine-tuned)** | **F1 ≈ 0.8946**  | **0.90**  | **Best overall model**                   |
| CNN (with EarlyStopping)     | Val Acc ≈ 0.976  | epoch=11  | ~63% training time saved                 |
| 3-Class (RMSprop, lr=0.001)  | Val Acc ≈ 0.8114 | —         | Multi-class classification (15 epochs)   |

### 🎯 Performance Highlights
- ✅ **Exceeded benchmark:** Achieved >93% accuracy on test set
- ✅ **F1-Score optimization:** Fine-tuned threshold from 0.10 to 0.90
- ✅ **Efficiency gains:** EarlyStopping reduced training time by 63% while maintaining performance
- ✅ **Multi-class capability:** Successfully classified Normal, Bacterial, and Viral pneumonia

---

## 🧠 Deep Learning Techniques Used

| Technique                          | Description                                                      |
|------------------------------------|------------------------------------------------------------------|
| **Convolutional Neural Networks**  | Feature extraction from X-ray images using conv/pooling layers  |
| **Transfer Learning**              | ResNet152V2 pre-trained on ImageNet for domain adaptation       |
| **Fine-Tuning**                    | Unfreezing and retraining selected layers for medical imaging   |
| **Optimizer Tuning**               | Comparison of SGD, Momentum, Adam, RMSprop                      |
| **Learning Rate Experiments**      | Testing lr = 0.01, 0.001, 0.0001                                |
| **Precision–Recall Analysis**      | PR curves and F1-score vs threshold plots                       |
| **EarlyStopping**                  | Performance/time trade-off evaluation                           |
| **Multi-Class Classification**     | Softmax output for Normal/Bacterial/Viral differentiation       |
| **Data Augmentation**              | Random flips, rotations, and brightness adjustments             |
| **Regularization**                 | Dropout and batch normalization to prevent overfitting          |

---

## 📈 Visual Results & Examples

### Task 1: Dataset & Architecture

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task1/Task1_Sample_Dataset_Images.png" alt="Sample X-Ray Images" width="800"/>
  <p><em>Sample chest X-ray images from the dataset (Normal vs Pneumonia)</em></p>
</div>

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task1/Task1_CNN_Architecture_Table.png" alt="CNN Architecture" width="700"/>
  <p><em>Custom CNN architecture summary</em></p>
</div>

### Task 2: Training Performance & Precision-Recall Analysis

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task2/Task2_Transfer_Learning_Fine-tuned_Training_History.png" alt="Fine-tuned Training" width="800"/>
  <p><em>Training and validation accuracy/loss curves for fine-tuned ResNet152V2</em></p>
</div>

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task2/Task2_Precision_Recall_Analysis.png" alt="PR Analysis" width="800"/>
  <p><em>Precision-Recall curves and F1-score optimization across different thresholds (0.10-0.90)</em></p>
</div>

### Task 3: Optimizer Comparison & EarlyStopping

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task3/Task3_Optimizer_Comparison_Summary.png" alt="Optimizer Comparison" width="800"/>
  <p><em>Comprehensive optimizer comparison: SGD, SGD+Momentum, Adam, RMSprop across different learning rates</em></p>
</div>

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task3/Task3_EarlyStopping_Comparison.png" alt="EarlyStopping Analysis" width="800"/>
  <p><em>EarlyStopping evaluation showing performance vs training time trade-offs</em></p>
</div>

### Task 4: Multi-Class Classification Results

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task4/Task4_Confusion_Matrix_Final.png" alt="Confusion Matrix" width="600"/>
  <p><em>Confusion matrix for 3-class classification (Normal / Bacterial / Viral Pneumonia)</em></p>
</div>

<div align="center">
  <img src="DEEP LEARNING - Final Project/images/Task4/Task4_Model_Comparison_Summary.png" alt="Model Comparison" width="800"/>
  <p><em>Multi-class model performance comparison across different optimizers and learning rates</em></p>
</div>

> 📁 **Note:** More detailed visualizations are available in the [`images/`](DEEP%20LEARNING%20-%20Final%20Project/images) folder, organized by task.

---

## 🧪 Reproducibility Guidelines

To ensure consistent results across runs:

1. **Random Seeds:** All random seeds are fixed in each task script (NumPy, TensorFlow)
2. **Data Splits:** Identical train/val/test partitions used across all experiments
3. **Model Checkpoints:** Best models saved during training for later evaluation
4. **Configuration Logging:** All hyperparameters and results logged per experiment
5. **Test Set Isolation:** Test data only used for final evaluation, never during training/validation

---

## 🧩 Course Labs Overview

This repository also includes **10 comprehensive lab assignments** completed throughout the Deep Learning Systems course:

| Lab | Topic | Focus Area |
|-----|-------|------------|
| **Lab 1-3** | Classical Machine Learning | KNN, Logistic Regression, SVM |
| **Lab 4-6** | Neural Network Foundations | MLPs, Backpropagation, Stochastic Gradient Descent |
| **Lab 7-8** | Advanced Deep Learning | CNNs, Transfer Learning, Feature Extraction |
| **Lab 9** | Training Optimization | Various optimizers, Regularization techniques |
| **Lab 10** | Sequence Models | RNNs, LSTMs, EarlyStopping strategies |

Each lab includes:
- Python implementation notebooks
- Detailed analysis reports
- Experimental results and visualizations

---

## 🎓 Academic Context

**Course Information:**
- **Course Code:** 31245
- **Course Name:** מערכות לומדות ולמידה עמוקה (Deep Learning Systems)
- **Institution:** Braude College of Engineering
- **Year:** 2025, Semester B
- **Project Type:** Final Course Project

**Documentation Available:**
- 📄 [Project Brief (Hebrew)](DEEP%20LEARNING%20-%20Final%20Project/פרוייקט%20מערכות%20לומדות%20למידה%20עמוקה.pdf)
- 📄 [Complete Project Report (PDF)](DEEP%20LEARNING%20-%20Final%20Project/deep-learning%20project%20report.pdf)
- 📄 [Course Syllabus (Hebrew)](מערכות%20לומדות%20ולמידה%20עמוקה_סילבוס_2025.pdf)

---

## ⚖️ Ethics & Usage Disclaimer

⚠️ **Important Notice:**

This project is developed as an **academic exercise** in Deep Learning and AI technologies.  

- ❌ **NOT** a certified medical diagnostic tool
- ❌ **NOT** approved for clinical use or medical decision-making
- ❌ **NOT** a replacement for professional medical diagnosis

**Intended Use:**
- ✅ Educational purposes only
- ✅ Research and academic study
- ✅ Deep Learning technique demonstration

**Dataset Notice:**  
All datasets used are publicly available for educational purposes and comply with respective licensing terms. No patient privacy is violated.

---

## 📚 References & Resources

### Project Documentation
- [Project Brief (Hebrew)](DEEP%20LEARNING%20-%20Final%20Project/פרוייקט%20מערכות%20לומדות%20למידה%20עמוקה.pdf)
- [Final Report](DEEP%20LEARNING%20-%20Final%20Project/deep-learning%20project%20report.pdf)
- [Course Syllabus](מערכות%20לומדות%20ולמידה%20עמוקה_סילבוס_2025.pdf)

### Key Technologies
- **TensorFlow/Keras:** Deep learning framework
- **ResNet152V2:** He et al., "Deep Residual Learning for Image Recognition" (2016)
- **Transfer Learning:** Pan & Yang, "A Survey on Transfer Learning" (2010)

### Dataset
- **Chest X-Ray Images (Pneumonia):** Public medical imaging dataset
- Source: Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (2018)
- Available on: Kaggle, NIH Clinical Center, Radiological Society repositories

---

## 👥 Authors

**Project Team Members:**
- **Francis Aboud**
- **Bshara Habib**
- **Maria Nakhle**

**Course:** Deep Learning Systems (31245) – Braude College of Engineering  
**Year:** 2025, Semester B

🔗 **Project Repository:** [deep-learning-pneumonia-xray-diagnosis](https://github.com/FrancisA2000/deep-learning-pneumonia-xray-diagnosis)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgments

- **Braude College of Engineering** for providing the academic framework and resources
- **Course Instructors** for guidance throughout the Deep Learning Systems course
- **Medical imaging community** for providing open datasets for educational purposes
- **TensorFlow/Keras teams** for excellent deep learning tools and documentation

---

## 🚀 Future Enhancements

Potential areas for extension:
- [ ] Deploy model as web application (Flask/Streamlit)
- [ ] Implement GradCAM for interpretability and explainable AI
- [ ] Expand dataset with additional X-ray sources for improved generalization
- [ ] Experiment with newer architectures (EfficientNet, Vision Transformers)
- [ ] Add ensemble methods for improved accuracy and robustness
- [ ] Develop mobile application for edge deployment
- [ ] Integrate with PACS (Picture Archiving and Communication System)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ for Deep Learning education

[Report Bug](https://github.com/FrancisA2000/deep-learning-pneumonia-xray-diagnosis/issues) · [Request Feature](https://github.com/FrancisA2000/deep-learning-pneumonia-xray-diagnosis/issues)

</div>
