# IT3091-House-Price-Prediction
IT3091 Machine Learning Group Project - Ames Housing Price Prediction

# IT3091 House Price Prediction

## Machine Learning Group Project

**Module:** IT3091 - Machine Learning  
**Group:** 2026-AI-08K  
**Track:** Guided Data Track  
**Domain:** Real Estate Analytics  
**Primary Lens:** Price Prediction  
**Secondary Lens:** Valuation Feature Analysis  

---

## 1. Project Overview

This project develops a machine learning solution for predicting residential
property sale prices using the Ames Housing / House Prices dataset.

The project is part of the IT3091 Machine Learning Group Assignment.

The main goal is to build and evaluate regression models that can estimate
the expected sale price of a residential property from its available
characteristics.

The predicted price is treated as a decision-support estimate rather than
a replacement for professional property valuation.

---

## 2. Business Problem

A real estate organisation wants to use historical property information
to support property pricing and valuation decisions.

The project focuses on the following question:

> Given the characteristics of a residential property, what sale price
> should be expected?

The model will provide an estimated property price that can be used as
supporting evidence when comparing or evaluating properties.

---

## 3. Project Lenses

### Primary Lens — Price Prediction

The primary task is to predict `SalePrice`, which is a continuous numerical
target.

Therefore, the machine learning problem is:

**Supervised Regression**

### Secondary Lens — Valuation Feature Analysis

The secondary analysis will investigate which property characteristics have
the strongest relationship with predicted or observed property values.

This secondary lens supports the main price prediction task and will not
be treated as a separate machine learning project.

---

## 4. Dataset

The project uses the:

**Ames Housing / House Prices dataset**

Source:

**Kaggle - House Prices: Advanced Regression Techniques**

### Dataset files

- `train.csv`
- `test.csv`

### Training data

- 1,460 property records
- 81 columns
- `SalePrice` is available as the target variable

### Test data

- 1,459 property records
- 80 predictor columns
- `SalePrice` is not available

### Unit of Analysis

One row represents one residential property.

---

## 5. Machine Learning Task

The project will follow a supervised regression workflow.

The planned process is:

1. Understand the business problem
2. Understand the dataset
3. Perform exploratory data analysis
4. Investigate data quality
5. Perform preprocessing
6. Perform feature engineering where justified
7. Build a baseline regression model
8. Train at least three alternative regression models
9. Validate and compare the models
10. Analyse model errors
11. Interpret important valuation features
12. Develop an evidence-based recommendation
13. Document limitations and responsible use

---

## 6. Model Evaluation

The project will use suitable regression metrics, including:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)

Models will be compared using a consistent validation strategy.

The final model decision will consider model performance, validation results,
interpretability, data characteristics, and the project requirements.

---

## 7. Project Structure

```text
IT3091-House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── README.md
│
├── notebooks/
│
├── docs/
│  
├── reports/
│
├── README.md
├── requirements.txt
└── .gitignore