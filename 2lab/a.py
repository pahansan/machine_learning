import pandas as pd

salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')

print(salary.columns)

y = salary['Salary']
X = salary[['Experience Years']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, random_state=2529)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

print(model.intercept_)
print(model.coef_)

y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

print('Абсолютная ошибка', mean_absolute_error(y_test,y_pred))
print('Абсолютная ошибка в процентах', mean_absolute_percentage_error(y_test,y_pred))
print('Квадратичная ошибка', mean_squared_error(y_test,y_pred))
