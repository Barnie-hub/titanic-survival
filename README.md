# Titanic Survival Prediction

## Project Overview
Machine learning project to predict passenger survival on the Titanic using various classification algorithms. This project demonstrates data cleaning, exploratory data analysis, feature engineering, and model optimization techniques.

## Business Problem
The sinking of the Titanic resulted in the death of 1,502 out of 2,224 passengers and crew. This project aims to predict which passengers survived based on features such as socio-economic status, age, gender, and family relationships. This analysis provides insights into the social dynamics and survival factors during maritime disasters.

## Data Source
- **Source**: Kaggle Titanic Competition Dataset
- **Training samples**: 891 passengers
- **Test samples**: 418 passengers
- **Target Variable**: Survived (0 = No, 1 = Yes)

### Features
- **PassengerId**: Unique identifier
- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Passenger name
- **Sex**: Gender
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Fare**: Passenger fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

## Project Structure
```
titanic-survival/
├── data/
│   ├── raw/              # Original datasets
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/        # Cleaned and engineered features
├── notebooks/
│   ├── 01_EDA.ipynb     # Exploratory Data Analysis
    ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── models/              # Saved trained models
├── reports/             # Generated analysis reports
│   └── figures/        # Generated graphics
├── config/             # Configuration files
└── tests/              # Unit tests
```

## Methodology

### 1. Data Exploration & Cleaning
- Analyze missing data patterns
- Understand feature distributions
- Identify correlations with survival

### 2. Feature Engineering
- Extract titles from names
- Create family size features
- Bin continuous variables
- Handle missing values strategically

### 3. Model Development
- Baseline: Logistic Regression
- Tree-based: Random Forest, XGBoost
- Ensemble methods
- Cross-validation strategy

### 4. Model Optimization
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Model interpretation

## Key Findings
*[To be updated as project progresses]*

- Initial observations:
  - Gender played a significant role in survival
  - Passenger class correlated with survival rates
  - Age and family size showed interesting patterns

## Results
*[To be updated with final metrics]*

- Best Model: [TBD]
- Accuracy: [TBD]
- Precision: [TBD]
- Recall: [TBD]
- F1 Score: [TBD]

## Technologies Used
- **Python 3.11**
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Development**: Jupyter Notebook

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
- Start with `01_EDA.ipynb` for data exploration
- Follow through each numbered notebook sequentially

## Future Improvements
- [ ] Implement deep learning models
- [ ] Create web application for predictions
- [ ] Add SHAP values for model interpretability
- [ ] Develop automated pipeline

## Author
**Steven Barbaro**
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [@yourusername]
- Email: your.email@example.com

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Kaggle for providing the dataset
- Titanic competition community for insights
- DataCamp for educational resources

---
*Last Updated: November 2024*