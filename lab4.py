import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження даних
df = pd.read_csv("Walmart.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Візуалізація тенденцій продажів
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Weekly_Sales', hue='Store', data=df)
plt.title("Тенденції тижневих продажів по магазинах")
plt.xlabel("Дата")
plt.ylabel("Тижневі продажі")
plt.legend(title="Магазин")
plt.show()

# Аналіз продажів у святкові та несвяткові тижні
plt.figure(figsize=(8, 6))
sns.boxplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title("Продажі: Святкові та звичайні тижні")
plt.xlabel("Святковий тиждень")
plt.ylabel("Тижневі продажі")
plt.xticks([0, 1], ['Не святковий', 'Святковий'])
plt.show()

# Кореляційний аналіз
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Матриця кореляцій")
plt.show()

# Побудова моделі прогнозування
X = df[['Store', 'Unemployment', 'CPI', 'Fuel_Price', 'Temperature', 'Holiday_Flag']]
y = df['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
