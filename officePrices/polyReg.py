# Enter your code here. Read input from STDIN. Print output to STDOUT

import sys
import numpy as np

import pandas as pd

F, N = map(int, input().split())
# col "2" is price
train = pd.DataFrame(np.array([input().split() for _ in range(N)], float))
T = int(input())
test = pd.DataFrame(np.array([input().split() for _ in range(T)], float))


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
poly_variables = poly.fit_transform(train.iloc[:,:-1])

poly_variables_test = poly.transform(test)

#from sklearn.model_selection import train_test_split
#poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, #train.price, test_size = 0.1, random_state = 4)

from sklearn.linear_model import LinearRegression
model = LinearRegression()


# overfit much?
model.fit(poly_variables, train.iloc[:,-1])

pred = model.predict(poly_variables_test)

print(*pred, sep="\n")
