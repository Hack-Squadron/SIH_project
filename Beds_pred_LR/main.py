import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Month': ['Jan-06', 'Feb-06', 'Mar-06', 'Apr-06', 'May-06', 'Jun-06', 'Jul-06', 'Aug-06', 'Sep-06', 'Oct-06', 'Nov-06'],
    'Available_Beds': [4.7, 6.5, 4.9, 6.1, 5.7, 5.4, 5.3, 5.6, 5.2, 5.2, 4.8]
}
df = pd.DataFrame(data)

df['Month_Num'] = range(1, len(df) + 1)

X = df[['Month_Num']]
y = df['Available_Beds']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
future_months = pd.DataFrame({'Month_Num': [len(df) + 1]})
future_pred = model.predict(future_months)
print(f'Predicted availability for Dec-06: {future_pred[0]}')
#

