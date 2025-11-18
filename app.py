
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="ReFill Hub Dashboard", layout="wide")

# Embedded images
logo_b64 = "iVBORw0KGgoAAAANSUhEUgAAAZAAAADICAIAAABJdyC1AAADdElEQVR4nO3aQW7iMABA0aSay/ZC5bh0EQmhAClMQfQ7761Q5Npm82WcztPXBJDw8e4NANxLsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIEOwgAzBAjIEC8gQLCBDsIAMwQIyBAvIECwgQ7CADMECMgQLyBAsIOPfuzfwtxw/j9M0zYf59Hlx68l/L/GbGWC35unr3Vt4th+zcj5gY9j54PMBl08uZ54P80aYNma48yvAPg17wlqScfw8rg5H82E+1eQyQ6cxv1l09fC01mrRjWxd3eF07XR2ObkTHAPbxR3WrR96S9GWz0vInrvitFmNy/3cM+f5X92q1aMzQ8Wwwdo+Qy2eG6nVzK+YFnZu2GBdvSZ/XaFe5/wYCDs38qX71ZudldVl0KM27pju/HX20GuBW3dY22NgGAMGa4d+fO0IYxj2LeEe+O8H9sYJC8gY9tIdGI9gARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkPENf86tsisnMFgAAAAASUVORK5CYII="
flow_b64 = "iVBORw0KGgoAAAANSUhEUgAAAZAAAADICAIAAABJdyC1AAADUklEQVR4nO3YQW6bUBRAUVJ1qWZVLDYZULlfkFA7VgNXOmeEETwYXT38dru9TwAFv85+AYBHCRaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWT8PvsFHrUsf4/n+dUh35iw3vvtRwOvu2iwxjxN0zTPf0qxOf+seX51AnCiiwZr9c91ZtyY7hvQsmx/7qdt1qXNNft7v7oR+EmX/g9rWY4WorEda6Q2xmxNn31Urmf21+yP78Pvux7w89ob1t7YoAdHrV37tIybWgHnuvSG9ayDshwsa89+5R3vfcD/c+lgjV0YM7Ee7z/cVuNa9NU1B8/aPGgz1n9YcKK32+397HcAeMilNyyAkWABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGYIFZAgWkCFYQIZgARmCBWQIFpAhWECGYAEZggVkCBaQIVhAhmABGR/ZRF8zgtwsPAAAAABJRU5ErkJggg=="

def show_img(b64):
    st.image(b64, use_container_width=True)

st.markdown("<style>.sidebar * { color:white !important; }</style>", unsafe_allow_html=True)

with st.sidebar:
    show_img(logo_b64)
    st.markdown("<h3 style='text-align:center;'>ReFill Hub</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Smart • Sustainable • Simple</p>", unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("", ["🏠 Dashboard Overview","🧩 About ReFill Hub","📊 Analysis","📥 Dataset Overview","⚙️ Model Settings","👥 Team Members"], label_visibility="collapsed")

df = pd.read_csv("csv/ReFillHub_SyntheticSurvey.csv")

st.image(logo_b64, use_container_width=True)

if menu == "🏠 Dashboard Overview":
    st.title("🏠 Dashboard Overview")
    st.write("Welcome to the ReFill Hub Data Analytics Dashboard.")

elif menu == "🧩 About ReFill Hub":
    st.title("🧩 About ReFill Hub")
    st.write("ReFill Hub promotes sustainability through refill stations.")

elif menu == "📥 Dataset Overview":
    st.title("📥 Dataset Overview")
    st.dataframe(df)
    st.write(df.describe())

elif menu == "⚙️ Model Settings":
    st.title("⚙️ Model Settings")
    st.write("Settings coming soon.")

elif menu == "👥 Team Members":
    st.title("👥 Team Members")
    st.write("Kshitij — Data Strategy\nJay — Visualisation")

elif menu == "📊 Analysis":
    st.title("📊 Complete Analysis Panel")
    tab1, tab2, tab3, tab4 = st.tabs(["Classification","Clustering","Regression","Insights"])

    with tab1:
        st.header("Classification Models")
        target = "Likely_to_Use_ReFillHub"
        X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

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
            results.append([name, accuracy_score(y_test,pred), precision_score(y_test,pred), recall_score(y_test,pred), f1_score(y_test,pred)])

        st.dataframe(pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score"]))

        fig, ax = plt.subplots(figsize=(5,3))
        for name, model in trained.items():
            prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0,1],[0,1],"--",color="gray")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.header("Clustering Coming Soon")

    with tab3:
        st.header("Regression Models")
        y_reg = df["Willingness_to_Pay_AED"]
        X_reg = pd.get_dummies(df.drop(columns=["Willingness_to_Pay_AED"]), drop_first=True)
        Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, test_size=0.2)
        reg_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso()
        }
        summary = []
        for name, model in reg_models.items():
            model.fit(Xtr,ytr)
            pred = model.predict(Xte)
            mae = np.mean(np.abs(pred-yte))
            rmse = np.sqrt(np.mean((pred-yte)**2))
            r2 = model.score(Xte,yte)
            summary.append([name, mae, rmse, r2])
        st.dataframe(pd.DataFrame(summary, columns=["Model","MAE","RMSE","R²"]))

    with tab4:
        st.header("Insights")
        st.markdown("""
        - High-income customers prefer malls  
        - Eco-conscious users adopt more  
        - Glass container users retain longer  
        - Combo refills increase spending  
        - Gen Z reacts best to sustainability campaigns  
        """)
        show_img(flow_b64)
