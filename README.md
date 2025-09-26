# Computer Vision – CNN on CIFAR-10

Convolutional Neural Networks (CNNs) applied to the **CIFAR-10 dataset** using **TensorFlow/Keras**.  
This project explores baseline CNNs, data augmentation, hyperparameter tuning, and evaluation with accuracy, F1-score, and confusion matrix.

📺 **Demo video:** [Watch on YouTube](https://youtu.be/wXUQDArZj-Q)  
📑 **Full report:** See [`report.pdf`](./report.pdf)

---

## 🚀 Features
- CNN architecture built with the Keras Sequential API  
- Data preprocessing: normalization + one-hot encoding  
- Data augmentation (rotation, shift, flip, zoom)  
- Hyperparameter tuning with **Keras Tuner**  
- Regularization: EarlyStopping & LearningRateScheduler  
- Evaluation with **accuracy, loss curves, F1-score, confusion matrix**  
- Saved trained model (`.h5`) for reuse (excluded from repo if too large)  

---

## 📊 Results

### Model Performance
- **Baseline CNN**: validation accuracy started at ~46% in the first epoch, improved to ~67% after 5 epochs.  
- **With augmentation + hyperparameter tuning**:  
  - Best validation accuracy: **72.4%**  
  - Final test accuracy: **67.3%**  
  - F1-score: Balanced across most classes, weaker in **cat vs dog** and **deer vs horse**.  

### Learning Curves
- Training vs validation accuracy showed **overfitting in baseline**,  
  fixed with dropout, data augmentation, and learning rate scheduling.  
- Loss curves showed smoother convergence with tuned hyperparameters.  

### Confusion Matrix
- Strong predictions in **airplane, automobile, truck**.  
- Frequent confusion in **animals with visual similarity** (cats vs dogs, deer vs horse).  

📊 See [`examples/`](./examples) for full plots and confusion matrix.  

---

## 🛠️ Tech Stack
- **Python 3.x**  
- **TensorFlow / Keras**  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- scikit-learn  
- Keras Tuner  
- Jupyter Notebook  

---

## ▶️ Quickstart
```bash
# 1) Clone the repo
git clone https://github.com/myriamlmiii/computer-vision-cnn-cifar10.git
cd computer-vision-cnn-cifar10

# 2) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the notebook
jupyter notebook cnn_cifar10.ipynb
