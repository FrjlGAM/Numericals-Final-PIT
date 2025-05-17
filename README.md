Polynomial Regression Calculator

A Streamlit-based web app for performing and visualizing polynomial regression on custom datasets. Users can input their own data, select the polynomial degree, and view the resulting regression equation, prediction comparison, and graph.

-------------------------------------------------------------------------------------------------------------------------------------------------

Method Explanation

Polynomial regression is a type of regression analysis where the relationship between the independent variable x and the dependent variable y is modeled as an nth-degree polynomial.
It extends linear regression by allowing for nonlinear relationships.

This app uses:

* PolynomialFeatures from sklearn.preprocessing to transform input features.
* LinearRegression to fit the model.
* r² score to evaluate how well the polynomial model fits the data.

-------------------------------------------------------------------------------------------------------------------------------------------------

How to Run the App

Requirements:

* Python 3.7 or higher
* Required libraries:
  pip install streamlit numpy pandas scikit-learn matplotlib

To run:

1. Save the app as app.py
2. Open your terminal in the same directory
3. Run the command:
   streamlit run app.py

-------------------------------------------------------------------------------------------------------------------------------------------------

Sample Inputs and Expected Output

Sample Input:

* x values: 1, 2, 3, 4, 5
* y values: 1, 4, 9, 16, 25
* Degree: 2

Expected Output:

* Polynomial Equation:
  y = 1.0000x^2
* Effective polynomial degree: 2
* R² score: 1.0000 (perfect fit)
* Graph: Shows a smooth parabolic curve with data points exactly on the curve
* Table: Actual y and Predicted y columns will match exactly
