from pathlib import Path

import pandas as pd
import pickle
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = BASE_DIR / "outputs" / "attrition_model.pkl"
RISK_PATH = BASE_DIR / "outputs" / "at_risk_employees.csv"

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")


@st.cache_data
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours"], errors="ignore")
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    return df


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_risk_data() -> pd.DataFrame:
    try:
        return pd.read_csv(RISK_PATH)
    except FileNotFoundError:
        return pd.DataFrame()


def build_feature_row(values: dict) -> pd.DataFrame:
    features = list(values.keys())
    return pd.DataFrame([values], columns=features)


def main() -> None:
    df = load_dataset()
    model = load_model()
    risk_df = load_risk_data()

    st.title("HR Attrition Risk Dashboard")
    st.caption("Live employee attrition prediction using the trained machine learning model.")

    tab_predict, tab_insights, tab_watchlist = st.tabs(["Predict Risk", "Insights", "Watch List"])

    with tab_predict:
        st.subheader("Employee profile input")
        with st.form("predict_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=35)
                business_travel = st.selectbox("Business Travel", options=df["BusinessTravel"].unique().tolist(), index=0)
                daily_rate = st.number_input("Daily Rate", min_value=100, max_value=2000, value=800)
                department = st.selectbox("Department", options=df["Department"].unique().tolist(), index=0)
                distance = st.number_input("Distance From Home", min_value=1, max_value=40, value=10)
                education = st.number_input("Education", min_value=1, max_value=5, value=3)
                education_field = st.selectbox("Education Field", options=df["EducationField"].unique().tolist(), index=0)
                environment_satisfaction = st.number_input("Environment Satisfaction", min_value=1, max_value=4, value=3)
                gender = st.selectbox("Gender", options=df["Gender"].unique().tolist(), index=0)
                hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)
            with col2:
                job_involvement = st.number_input("Job Involvement", min_value=1, max_value=4, value=3)
                job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
                job_role = st.selectbox("Job Role", options=df["JobRole"].unique().tolist(), index=0)
                job_satisfaction = st.number_input("Job Satisfaction", min_value=1, max_value=4, value=3)
                marital_status = st.selectbox("Marital Status", options=df["MaritalStatus"].unique().tolist(), index=0)
                monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
                monthly_rate = st.number_input("Monthly Rate", min_value=1000, max_value=30000, value=15000)
                num_companies = st.number_input("Num Companies Worked", min_value=0, max_value=10, value=2)
                over_time = st.selectbox("Over Time", options=["Yes", "No"], index=1)
                percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, max_value=30, value=15)
            with col3:
                performance_rating = st.number_input("Performance Rating", min_value=1, max_value=4, value=3)
                relationship_satisfaction = st.number_input("Relationship Satisfaction", min_value=1, max_value=4, value=3)
                stock_option = st.number_input("Stock Option Level", min_value=0, max_value=3, value=1)
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)
                training_times = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)
                work_life_balance = st.number_input("Work Life Balance", min_value=1, max_value=4, value=3)
                years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)
                years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
                years_since_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
                years_with_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3)

            submit = st.form_submit_button("Predict risk")

        if submit:
            values = {
                "Age": age,
                "BusinessTravel": business_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance,
                "Education": education,
                "EducationField": education_field,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies,
                "OverTime": over_time,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StockOptionLevel": stock_option,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_promotion,
                "YearsWithCurrManager": years_with_manager,
                "EmployeeNumber": 9999,
            }

            feature_df = build_feature_row(values)
            prob = model.predict_proba(feature_df)[0, 1]
            risk_score = round(float(prob) * 100, 2)
            label = "High risk" if prob >= 0.5 else "Lower risk"

            st.markdown("---")
            colA, colB = st.columns(2)
            with colA:
                st.metric("Predicted Attrition Risk", f"{risk_score}%")
            with colB:
                st.metric("Risk Category", label)

            if prob >= 0.5:
                st.warning("This employee is at elevated risk of leaving. HR should review workload, compensation, and progression opportunities.")
            else:
                st.success("This employee appears relatively stable. Continue regular engagement and development check-ins.")

    with tab_insights:
        st.subheader("Dataset overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Employees", len(df))
        col2.metric("Attrition Rate", f"{df['Attrition'].mean() * 100:.1f}%")
        col3.metric("Avg Monthly Income", f"${df['MonthlyIncome'].mean():,.0f}")
        col4.metric("Avg Years at Company", f"{df['YearsAtCompany'].mean():.1f}")

        st.subheader("Key HR patterns")
        dept_risk = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False).reset_index()
        st.bar_chart(dept_risk.set_index("Department"), use_container_width=True)

        income_box = df[["MonthlyIncome", "Attrition"]].copy()
        st.bar_chart(income_box.groupby("Attrition")["MonthlyIncome"].mean().rename("Average Income"))

        overtime = df.groupby("OverTime")["Attrition"].mean().reset_index()
        st.line_chart(overtime.set_index("OverTime"), use_container_width=True)

    with tab_watchlist:
        st.subheader("Most at-risk employees")
        if risk_df.empty:
            st.info("The watch list is not available yet. Train the model and generate the CSV output first.")
        else:
            display_df = risk_df[["EmployeeNumber", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany", "Attrition_Probability"]].copy()
            display_df["Attrition_Probability"] = display_df["Attrition_Probability"].round(3)
            st.dataframe(display_df.head(15), use_container_width=True)


if __name__ == "__main__":
    main()
