# Titanic Survival Prediction

[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Score-76.79%25-blue)](https://www.kaggle.com/competitions/titanic)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)](https://github.com/)

## Project Overview

Machine learning project to predict passenger survival on the Titanic using ensemble classification methods. This project demonstrates a complete data science workflow: exploratory data analysis, data preprocessing, feature engineering, model development, hyperparameter tuning, and ensemble methods.

**Final Results:**
- **Best Model:** Hard Voting Ensemble (Random Forest + XGBoost + Gradient Boosting)
- **Cross-Validation Accuracy:** 84.7%
- **Kaggle Leaderboard Score:** 76.79%

## Business Problem

The sinking of the Titanic resulted in the death of 1,502 out of 2,224 passengers and crew. This project aims to predict which passengers survived based on features such as socio-economic status, age, gender, and family relationships. The analysis reveals insights into survival factors during the disaster and demonstrates practical machine learning techniques.

## Key Findings

- **Gender:** Female survival rate (74.2%) vastly exceeded male (18.9%) - "women and children first"
- **Class:** Clear socioeconomic gradient - 1st class (63%), 2nd class (47%), 3rd class (24%)
- **Family Size:** Mid-sized families (2-4 members) had optimal survival rates
- **Title Engineering:** Extracting titles from names (Mr., Mrs., Miss., Master.) provided strong predictive signal

## Project Structure

```
titanic-survival/
├── data/
│   ├── raw/                    # Original Kaggle datasets
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # Cleaned and engineered features
│       ├── train_preprocessed.csv
│       ├── test_preprocessed.csv
│       ├── train_features.csv
│       ├── test_features.csv
│       └── test_passenger_ids.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and visualization
│   ├── 02_preprocessing.ipynb         # Data cleaning and imputation
│   ├── 03_feature_engineering.ipynb   # Feature creation
│   └── 04_modeling.ipynb              # Model training and evaluation
├── models/                     # Saved trained models (.pkl)
├── reports/
│   ├── figures/               # Generated visualizations
│   └── Titanic_ML_Report.docx # Final analysis report
├── submissions/
│   └── submission.csv         # Kaggle submission file
├── README.md
└── requirements.txt
```

## Methodology

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Analyzed missing data patterns (Age: 19.9%, Cabin: 77%, Embarked: 0.2%)
- Visualized survival distributions across features
- Identified key correlations with survival

### 2. Data Preprocessing (`02_preprocessing.ipynb`)
- Imputed Age using median grouped by Pclass and Sex
- Filled Embarked missing values with mode ('S')
- Extracted Deck from Cabin; created 'Unknown' category for missing
- Encoded categorical variables (Sex, Embarked, Deck)

### 3. Feature Engineering (`03_feature_engineering.ipynb`)
- **Title:** Extracted from passenger names (Mr, Mrs, Miss, Master, Rare)
- **FamilySize:** SibSp + Parch + 1
- **IsAlone:** Binary flag for solo travelers
- **AgeBin:** Categorical age groups (Child, Teenager, Young Adult, Adult, Senior)
- **FareBin:** Quartile-based fare categories

### 4. Model Development (`04_modeling.ipynb`)
- Established baselines: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with GridSearchCV and RandomizedSearchCV
- Ensemble methods: Voting Classifiers, Stacking Classifier
- Cross-validation with 5-fold stratified splits

## Results

| Model | CV Accuracy | Type |
|-------|-------------|------|
| **Hard Voting (RF+XGB+GB)** | **84.70%** | Ensemble |
| Soft Voting (RF+XGB+GB) | 84.42% | Ensemble |
| Stacking Classifier | 84.00% | Ensemble |
| XGBoost (Tuned) | 83.71% | Single |
| Random Forest (Tuned) | 83.43% | Single |
| Gradient Boosting (Tuned) | 83.29% | Single |
| Logistic Regression (Tuned) | 82.31% | Single |

### Top 5 Most Important Features
1. **Title_Mr** - Adult male indicator (lowest survival group)
2. **Sex_encoded** - Gender (females 4x more likely to survive)
3. **Pclass** - Socioeconomic status proxy
4. **Fare** - Ticket price (correlated with class)
5. **Deck_Unknown** - Missing cabin info (lower class indicator)

## Key Lessons Learned

The 8-point gap between CV accuracy (84.7%) and Kaggle score (76.8%) highlights important real-world ML considerations:

- **Overfitting Risk:** Complex ensembles can memorize training patterns that don't generalize
- **CV Optimism:** Cross-validation on training data can be optimistic
- **Feature Leakage:** Feature engineering decisions informed by training distributions

This gap reinforces why held-out test sets and production monitoring are essential.

## Technologies Used

- **Python 3.11**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn, XGBoost
- **Development:** Jupyter Notebook

## Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebooks in order:
```
01_data_exploration.ipynb → 02_preprocessing.ipynb → 03_feature_engineering.ipynb → 04_modeling.ipynb
```

## Future Improvements

- [ ] Feature selection to reduce overfitting
- [ ] SHAP values for model interpretability
- [ ] Experiment with LightGBM and neural networks
- [ ] Build interactive web application
- [ ] Nested cross-validation for more robust evaluation

## Data Source

- **Source:** [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **Training samples:** 891 passengers
- **Test samples:** 418 passengers
- **Target Variable:** Survived (0 = No, 1 = Yes)

## Author

**Steven Barnhart**
- GitHub: [@Barnie-hub](https://github.com/Barnie-hub)


## Acknowledgments

- Kaggle for providing the dataset and competition platform
- The Titanic ML community for insights and benchmarks

---

*Project completed December 2025*
