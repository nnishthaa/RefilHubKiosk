import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="ReFill Hub Dashboard", layout="wide")

st.markdown("<style>.sidebar * { color:white !important; }</style>", unsafe_allow_html=True)

def safe_image(path):
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    else:
        st.warning(f"⚠️ File not found: {path}")

with st.sidebar:
    safe_image("assets/refillhub_logo.png")
    st.markdown("<h3 style='text-align:center;'>ReFill Hub</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Smart • Sustainable • Simple</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🌱 What do you want to see?")
    menu = st.radio("", ["🏠 Dashboard Overview","🧩 About ReFill Hub","📊 Analysis","📥 Dataset Overview","⚙️ Model Settings","👥 Team Members"], label_visibility="collapsed")

@st.cache_data
def load_data():
    return pd.read_csv("ReFillHub_SyntheticSurvey.csv")

df = load_data()

if menu == "🏠 Dashboard Overview":
    st.title("🏠 Dashboard Overview")
    st.write("Welcome to the ReFill Hub Data Analytics Dashboard.")

elif menu == "🧩 About ReFill Hub":
    st.title("🧩 About ReFill Hub")
    st.write("ReFill Hub promotes sustainability through smart refill stations.")

elif menu == "📥 Dataset Overview":
    st.title("📥 Dataset Overview")
    st.dataframe(df)
    st.write(df.describe())

elif menu == "⚙️ Model Settings":
    st.title("⚙️ Model Settings")
    st.write("Hyperparameter settings coming soon.")

elif menu == "👥 Team Members":
    st.title("👥 Team Members")
    st.write("Kshitij — Data Strategy
Jay — Visualisation
Group 8 — MGB Dubai 2025")

elif menu == "📊 Analysis":
    st.title("📊 Complete Analysis Panel")

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Classification Models","🌀 Clustering Engine","📈 Regression Lab","💡 Insights & Personas"])

    with tab1:
        st.header("🤖 Classification Models")
        target = "Likely_to_Use_ReFillHub"
        X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        trained = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            trained[name] = model
            pred = model.predict(X_test)
            results.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)])

        st.subheader("📊 Model Performance Comparison")
        st.dataframe(pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score"]))

        st.subheader("📈 ROC Curve Comparison")
        fig, ax = plt.subplots(figsize=(5,3))
        for name, model in trained.items():
            prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.header("🌀 Clustering Engine")
        st.write("K-Means clustering coming soon.")

    with tab3:
        st.header("📈 Regression Lab")
        y_reg = df["Willingness_to_Pay_AED"]
        X_reg = pd.get_dummies(df.drop(columns=["Willingness_to_Pay_AED"]), drop_first=True)
        Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

        reg_models = {"Linear Regression": LinearRegression(),"Ridge Regression": Ridge(),"Lasso Regression": Lasso()}
        summary = []

        for name, model in reg_models.items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mae = np.mean(np.abs(pred-yte))
            rmse = np.sqrt(np.mean((pred-yte)**2))
            r2 = model.score(Xte,yte)
            summary.append([name, mae, rmse, r2])

        st.subheader("📋 Regression Summary Table")
        st.dataframe(pd.DataFrame(summary, columns=["Model","MAE","RMSE","R² Score"]))

    with tab4:
        st.header("💡 Insights & Personas")
        st.markdown("""
        - High-income customers prefer mall-based refill stations  
        - Eco-conscious users show highest refill likelihood  
        - Glass-container customers show strongest retention  
        - Combo refills increase overall spending  
        - Gen Z responds best to sustainability messaging  
        """)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(data=df, x="Age_Group", y="Try_Refill_Likelihood", ax=ax)
        st.pyplot(fig)

        st.subheader("📌 Insight Flowchart")
        safe_image("assets/insight_flowchart.png")
