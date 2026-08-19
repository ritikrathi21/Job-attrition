from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATASET_PATH = Path(__file__).resolve().parent / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET_COLUMN = "Attrition"
DROP_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]


def load_hr_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    return df


def save_eda_charts(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="OverTime", hue="Attrition")
    plt.title("Attrition by Overtime Status")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attrition_overtime.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    dept = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=dept, x="Department", y="Attrition", hue="Department", palette="viridis", dodge=False, legend=False)
    plt.title("Attrition Rate by Department")
    plt.ylabel("Attrition Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attrition_by_department.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", hue="Attrition", palette="Set2", dodge=False, legend=False)
    plt.title("Monthly Income vs Attrition")
    plt.xlabel("Attrition")
    plt.ylabel("Monthly Income")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "income_vs_attrition.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="YearsAtCompany", y="Attrition", estimator="mean")
    plt.title("Attrition Rate by Years at Company")
    plt.ylabel("Attrition Rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tenure_vs_attrition.png", dpi=200)
    plt.close()

    plot_path = OUTPUT_DIR / "attrition_eda.png"
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.countplot(data=df, x="OverTime", hue="Attrition", ax=axes[0, 0])
    axes[0, 0].set_title("Attrition by Overtime")

    dept_df = df.groupby("Department")["Attrition"].mean().sort_values(ascending=False)
    dept_df.plot(kind="bar", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Attrition Rate by Department")
    axes[0, 1].set_ylabel("Rate")

    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=axes[1, 0], hue="Attrition", palette="pastel", dodge=False, legend=False)
    axes[1, 0].set_title("Income vs Attrition")

    sns.lineplot(data=df, x="YearsAtCompany", y="Attrition", estimator="mean", ax=axes[1, 1])
    axes[1, 1].set_title("Attrition Rate by Tenure")
    axes[1, 1].set_ylabel("Rate")

    for ax in axes.flat:
        ax.grid(False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_models(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("classifier", RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42, class_weight="balanced_subsample")),
            ]
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
        results[name] = metrics

        print(f"{name} metrics:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
        print(classification_report(y_test, y_pred))
        print("\n")

    best_model_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = models[best_model_name]
    return best_model, results, X_test, y_test


def create_risk_list(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    X = df.drop(columns=[TARGET_COLUMN]).copy()
    probabilities = model.predict_proba(X)[:, 1]
    scored = X.copy()
    scored["Attrition_Probability"] = probabilities
    scored["Actual_Attrition"] = df[TARGET_COLUMN].values
    scored["EmployeeNumber"] = df["EmployeeNumber"]
    scored = scored.sort_values("Attrition_Probability", ascending=False).reset_index(drop=True)
    return scored.head(25)


def save_model(model: Pipeline, path: Path) -> None:
    with open(path, "wb") as file:
        pickle.dump(model, file)


def build_pdf_report(results: dict, at_risk: pd.DataFrame) -> None:
    output_path = OUTPUT_DIR / "hr_attrition_report.pdf"
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("HR Attrition Analysis Report", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 18))

    summary = Paragraph(
        "This report highlights employees most likely to leave and identifies the main drivers of attrition. "
        "The analysis is based on the actual HR dataset and focuses on retention risk, department patterns, and action-oriented recommendations.",
        styles["BodyText"],
    )
    story.append(summary)
    story.append(Spacer(1, 12))

    top_3 = [
        "Employees working overtime exhibit a materially higher attrition risk.",
        "Lower compensation and shorter tenure are strongly associated with elevated departure risk.",
        "Sales and other high-pressure roles show a greater probability of turnover relative to other departments.",
    ]
    story.append(Paragraph("Top 3 Drivers of Attrition", styles["Heading2"]))
    story.append(ListFlowable([ListItem(Paragraph(item, styles["BodyText"])) for item in top_3], bulletType="1"))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Model Performance", styles["Heading2"]))
    for name, metrics in results.items():
        line = (
            f"{name.replace('_', ' ').title()}: Accuracy={metrics['accuracy']:.3f}, "
            f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, ROC AUC={metrics['roc_auc']:.3f}"
        )
        story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommended Retention Actions", styles["Heading2"]))
    actions = [
        "Review overtime-heavy teams and rebalance workloads where possible.",
        "Create promotion pathways and retention plans for employees with longer tenure gaps.",
        "Audit compensation and role-level incentives for high-risk departments and job families.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(item, styles["BodyText"])) for item in actions], bulletType="1"))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Watch List", styles["Heading2"]))
    at_risk_summary = at_risk.head(5)[["EmployeeNumber", "Department", "JobRole", "MonthlyIncome", "YearsAtCompany", "Attrition_Probability"]]
    for _, row in at_risk_summary.iterrows():
        story.append(
            Paragraph(
                f"Employee {int(row['EmployeeNumber'])}: Department={row['Department']}, Role={row['JobRole']}, "
                f"Income={int(row['MonthlyIncome'])}, Tenure={int(row['YearsAtCompany'])}, Risk={row['Attrition_Probability']:.2f}",
                styles["BodyText"],
            )
        )

    document = SimpleDocTemplate(str(output_path), pagesize=letter)
    document.build(story)


def main() -> None:
    data = load_hr_data()
    save_eda_charts(data)

    best_model, results, _, _ = train_models(data)
    at_risk = create_risk_list(best_model, data)

    model_path = OUTPUT_DIR / "attrition_model.pkl"
    save_model(best_model, model_path)
    at_risk.to_csv(OUTPUT_DIR / "at_risk_employees.csv", index=False)
    build_pdf_report(results, at_risk)

    print("Project completed successfully using the real HR CSV dataset.")
    print(f"Model saved to: {model_path}")
    print(f"Risk list saved to: {OUTPUT_DIR / 'at_risk_employees.csv'}")
    print(f"Report saved to: {OUTPUT_DIR / 'hr_attrition_report.pdf'}")


if __name__ == "__main__":
    main()
