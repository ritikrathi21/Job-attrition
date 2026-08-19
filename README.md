# HR Attrition Analytics Internship Project

This project uses the real HR attrition dataset in this folder and builds a people-analytics workflow to predict which employees are likely to leave and why.

## What is included
- Real dataset loading from the project folder
- Data cleaning and preprocessing for numerical and categorical features
- Exploratory data analysis using charts and summary insights
- Model training with logistic regression and random forest
- Risk scoring for employees likely to leave
- Executive summary report in PDF format

## Project structure
- main.py — end-to-end pipeline using the real CSV file
- hr_attrition_analysis.ipynb — notebook version of the investigation
- WA_Fn-UseC_-HR-Employee-Attrition.csv — actual HR dataset for training
- outputs/ — charts, model artifact, risk list, and PDF report

## Run the project
1. Install dependencies:
   python -m pip install -r requirements.txt
2. Run the pipeline:
   python main.py

## Expected outputs
- outputs/attrition_eda.png
- outputs/attrition_model.pkl
- outputs/at_risk_employees.csv
- outputs/hr_attrition_report.pdf

## Business goal
The project answers two questions:
1. Who is likely to leave?
2. Why are they leaving?

The risk list output starts from index 0, matching your requirement.
