import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Polynomial Regression Calculator", page_icon="🎀", layout="centered")

st.title("📈 Polynomial Regression Calculator")
st.markdown("---")

st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #000000;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        padding: 2rem 3rem !important;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-top: 4rem !important;
        max-width: 60%;
        border: 2px solid #FFE4E1;
    }
    header {
        background-color: transparent !important;
    }
    div.stButton > button {
        background-color: #CDF0EA;
        color: #4b004b;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton {
        padding: 2rem 0;
    }
    div.stButton > button:hover {
        background-color: #bde0da;
        color: #FF1493;
        cursor: pointer;
    }
    input[type="range"] {
        accent-color: #FF1493;
    }
    .stTextInput>div>input {
        border: 2px solid #ff66b2;
        border-radius: 5px;
        padding: 5px 10px;
        color: #4b004b;
        font-weight: 600;
    }
    .stDataFrame table {
        border: 1px solid #ff66b2 !important;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)

x_values = st.text_input("Enter x values (comma-separated):", "1, 2, 3, 4, 5")
y_values = st.text_input("Enter y values (comma-separated):", "1, 4, 9, 16, 25")

degree = st.slider("🎚️ Select degree of polynomial:", min_value=1, max_value=10, value=2, step=1)

if st.button("▶️ Run Polynomial Regression"):
    x = np.array([float(i) for i in x_values.split(",")]).reshape(-1, 1)
    y = np.array([float(i) for i in y_values.split(",")])

    st.markdown("### 📝 Inputs")
    st.write(f"**x values:** {x.flatten().tolist()}")
    st.write(f"**y values:** {y.tolist()}")
    st.write(f"**Degree:** {degree}")

    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)

    intercept = model.intercept_
    coefs = model.coef_

    significant_coefs = [(i, c) for i, c in enumerate(coefs) if abs(c) > 1e-8]
    max_power = max(i for i, c in significant_coefs) if significant_coefs else 0

    terms = []
    if abs(intercept) > 1e-8:
        terms.append(f"{intercept:.2f}")

    for i in range(1, len(coefs)):
        coef = coefs[i]
        if abs(coef) > 1e-8:
            power_str = "x" if i == 1 else f"x^{i}"
            formatted = f"{abs(coef):.2f}{power_str}"
            if not terms:
                terms.append(formatted if coef > 0 else f"-{formatted}")
            else:
                sign = "+" if coef > 0 else "-"
                terms.append(f"{sign} {formatted}")


    equation = "y = " + " ".join(terms) if terms else "y = 0"

    st.markdown("""
    <h2 style='color:#FF1493; font-weight:bold; font-size:28px; margin-bottom:2px;'>
        🧮 Polynomial Equation
    </h2>
    """, unsafe_allow_html=True)
    st.markdown(f"<p style='color:#FF1493; font-size:20px; font-weight:bold; margin-top:0;'>{equation}</p>", unsafe_allow_html=True)

    st.markdown(f"**Effective polynomial degree:** {max_power}")

    st.write("### Coefficients")
    for i in range(1, len(coefs)):
        st.write(f"a{i} (x^{i}) = {coefs[i]}")
    st.write(f"Intercept (x^0) = {intercept}")

    st.markdown("### 📋 Table Comparing Actual vs Predicted Values")
    df = pd.DataFrame({'x': x.flatten(), 'Actual y': y, 'Predicted y': y_pred})
    st.dataframe(df)

    r2 = r2_score(y, y_pred)

    x_dense = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    x_dense_poly = poly.transform(x_dense)
    y_dense_pred = model.predict(x_dense_poly)

    st.write("### 📊 Graph")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(x, y, color='#FF6B6B', label='Actual', s=100, alpha=0.6)
    ax.plot(x_dense, y_dense_pred, color='#4ECDC4', label='Predicted', linewidth=3)
    ax.set_xlabel("x", fontsize=12, fontweight='bold')
    ax.set_ylabel("y", fontsize=12, fontweight='bold')
    ax.set_title("Polynomial Regression", fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    st.pyplot(fig)

    st.markdown("### 🎯 R² Score")
    st.write("The R² value indicates how well the model fits the data. A value closer to 1 means a better fit.")
    st.markdown(f"**R² score:** {r2:.4f}")
