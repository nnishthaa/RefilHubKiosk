import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="ReFill Hub Analytics Dashboard", layout="wide")

st.markdown(
    "<style>.sidebar * { color:white !important; }</style>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.image("assets/refillhub_logo.png", use_container_width=True)
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
    st.write("Welcome to the ReFill Hub Dashboard.")

elif menu == "🧩 About ReFill Hub":
    st.title("🧩 About ReFill Hub")
    st.write("ReFill Hub is a smart sustainable refill station concept.")

elif menu == "📥 Dataset Overview":
    st.title("📥 Dataset Overview")
    st.dataframe(df)
    st.write(df.describe())

elif menu == "⚙️ Model Settings":
    st.title("⚙️ Model Settings")
    st.write("Coming soon.")

elif menu == "👥 Team Members":
    st.title("👥 Team Members")
    st.write("Kshitij – Data Strategy")

elif menu == "📊 Analysis":
    st.title("📊 Complete Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Classification Models","🌀 Clustering Engine","📈 Regression Lab","💡 Insights & Personas"])

    # Classification
    with tab1:
        st.header("🤖 Classification Models")
        target = "Likely_to_Use_ReFillHub"
        X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = []
        trained = {}

        for name, m in models.items():
            m.fit(X_train, y_train)
            trained[name] = m
            pred = m.predict(X_test)
            results.append([name, accuracy_score(y_test,pred), precision_score(y_test,pred), recall_score(y_test,pred), f1_score(y_test,pred)])

        st.dataframe(pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score"]))

        fig, ax = plt.subplots(figsize=(5,3))
        for name, m in trained.items():
            prob = m.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.legend()
        st.pyplot(fig)

    # Clustering
    with tab2:
        st.header("🌀 Clustering Engine")
        st.write("Coming soon.")

    # Regression
    with tab3:
        st.header("📈 Regression Lab")
        y_reg = df["Willingness_to_Pay_AED"]
        X_reg = pd.get_dummies(df.drop(columns=["Willingness_to_Pay_AED"]), drop_first=True)
        Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2)

        reg_models = {"Linear Regression": LinearRegression(),"Ridge Regression": Ridge(),"Lasso Regression": Lasso()}
        summary = []

        for name, m in reg_models.items():
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            mae = np.mean(np.abs(pred-yte))
            rmse = np.sqrt(np.mean((pred-yte)**2))
            r2 = m.score(Xte,yte)
            summary.append([name, mae, rmse, r2])

        st.dataframe(pd.DataFrame(summary, columns=["Model","MAE","RMSE","R²"]))

    # Insights
    with tab4:
        st.header("💡 Insights & Personas")
        st.write("1. High-income customers prefer malls.")
        st.write("2. Eco-conscious users show high adoption.")
        st.write("3. Glass container users stay longer.")
        st.write("4. Combo refills boost spending.")
        st.write("5. Gen Z reacts best to ads.")
        st.image("assets/insight_flowchart.png")
