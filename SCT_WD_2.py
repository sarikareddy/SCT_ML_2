import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
df = pd.read_csv("/content/house_price_dataset.csv")
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
plt.figure(figsize=(6,4))
plt.scatter(y_pred, X_test['Bedrooms'])
plt.xlabel("Predicted House Price ($)")
plt.ylabel("Number of Bedrooms")
plt.title("Predicted House Price vs Bedrooms")
plt.grid(True)
plt.show()
plt.figure(figsize=(6,4))
plt.scatter(y_pred, X_test['Bathrooms'])
plt.xlabel("Predicted House Price ($)")
plt.ylabel("Number of Bathrooms")
plt.title("Predicted House Price vs Bathrooms")
plt.grid(True)
plt.show()