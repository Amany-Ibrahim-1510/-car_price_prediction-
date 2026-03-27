# 🚗 Car Price Prediction

> #MachineLearning #DataScience #Python #AI #PortfolioProject

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

End-to-end Machine Learning project to predict **car prices** based on real-world dataset features. This project demonstrates a complete ML pipeline from **EDA → preprocessing → modeling → evaluation → deployment-ready structure**.

---

## 📑 Table of Contents

* [📌 Project Overview](#-project-overview)
* [🎯 Objective](#-objective)
* [🧠 Key Features](#-key-features)
* [📂 Dataset](#-dataset)
* [🏗️ Project Structure](#️-project-structure)
* [⚙️ Installation & Setup](#️-installation--setup)
* [🔍 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [🧹 Data Preprocessing](#-data-preprocessing)
* [🤖 Modeling](#-modeling)
* [🧪 Testing](#-testing)
* [💾 Model Saving](#-model-saving)
* [📜 Logging System](#-logging-system)
* [⚙️ Configuration & Constants](#️-configuration--constants)
* [🛠️ Helper Utilities](#️-helper-utilities)
* [📊 Sample Workflow](#-sample-workflow)
* [🚀 Deployment](#-deployment-optional)
* [📄 Report](#-report)
* [📌 Future Improvements](#-future-improvements)
* [👨‍💻 Author](#-author)

---

## 🎯 Objective

Build a highly accurate model that predicts car prices and generalizes well on unseen data.

---

## 🧠 Key Features

* Full ML pipeline implementation
* Clean modular code (production-style)
* Multiple models comparison
* Logging system for experiments
* Config-driven architecture
* Ready for deployment

---

## 📂 Dataset

* Source: Kaggle
* Contains:

  * Train dataset (training + validation)
  * Test dataset (final evaluation)

⚠️ Note: Dataset is not included due to size. Download and place it in the `data/` folder.

---

## 🏗️ Project Structure

```
car_price_prediction/
│
├── data/
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── helper.py
│   ├── config.py
│   ├── enums.py
│
├── models/
├── logs/
├── report/
│   └── report.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/Amany-Ibrahim-1510/-car_price_prediction-.git
cd car_price_prediction
python -m venv venv
venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
```

---

## 🔍 Exploratory Data Analysis (EDA)

### ✔️ What was done

* Data understanding
* Missing values analysis
* Distribution analysis
* Correlation heatmap
* Outlier detection

### 📊 Insights

* Strong correlation between price and features like year, mileage
* Outliers exist in price
* Categorical variables need encoding

### 📝 EDA Conclusion

Used to guide preprocessing:

* Handle missing values
* Feature encoding
* Scaling

---

## 🧹 Data Preprocessing

* Missing values handling
* Encoding categorical features
* Feature scaling
* Train/Validation split

---

## 🤖 Modeling

### Models Tried

* Linear Regression
* Decision Tree
* Random Forest
* XGBoost

### 📈 Evaluation Metrics

* MAE
* MSE
* RMSE
* R² Score

### 🏆 Best Model

Selected based on lowest error + best validation performance.

---

## 🧪 Testing

* Evaluated on unseen test dataset
* Confirms model generalization

---

## 💾 Model Saving

* Saved using `.pkl`
* Load without retraining

---

## 📜 Logging System

* Logs stored in `logs/`
* Track:

  * Model performance
  * Hyperparameters

---

## ⚙️ Configuration & Constants

* `config.py` → settings
* `enums.py` → constants

---

## 🛠️ Helper Utilities

Inside `helper.py`:

* Load/save data
* Save/load model
* Logging
* Config handling

---

## 📊 Sample Workflow

```bash
python src/preprocessing.py
python src/modeling.py
```

---

## 🚀 Deployment (Optional)

Future extension:

* Flask / FastAPI API
* Web interface

---


## 📌 Future Improvements

* Hyperparameter tuning
* Feature engineering
* Deep learning models
* Real-time prediction API

---


---

## ⭐ If you like this project

Give it a ⭐ on GitHub!

---

## 📬 Contact

Feel free to connect for collaboration or questions.

---

# 🔥 Advanced Additions

## 📊 EDA Visualizations (Examples)

Add screenshots like:

* Distribution of car prices
* Correlation heatmap
* Price vs mileage scatter plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['price'], kde=True)
plt.title("Price Distribution")
plt.show()
```

---


## 🌐 Simple Flask API (Deployment Ready)

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)
model = pickle.load(open("models/model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = list(data.values())
    prediction = model.predict([features])
    return jsonify({"price": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
```

---

## 📦 requirements.txt Example

```
pandas
numpy
scikit-learn
matplotlib
seaborn
flask
xgboost
```


