from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_intercept(xs, ys, m):
	b = mean(ys) - m * mean(xs)
	return b

# To calculate the best fit slope
def best_fit_slope_and_intercept(xs, ys):
	m = ((mean(xs) * mean(ys)) - mean(xs * ys))
	m = m / ((mean(xs) * mean(xs)) - mean(xs * xs))
	b = best_fit_intercept(xs, ys, m)
	return m, b

def predict_y(x, m, b):
	y = (m * x) + b
	return y

# distance between the y of the line and the data
def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)

# How good is the best fit line
def coefficient_determination(ys_orig, ys_line):
	ys_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, ys_mean_line)
	return 1 - (squared_error_regr / squared_error_y_mean)

# The data
# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 8, 7], dtype=np.float64)
xs, ys = create_dataset(40, 10, 2, correlation='pos')

m, b = best_fit_slope_and_intercept(xs, ys)
regression_line = [(m*x)+b for x in xs]

r_squared = coefficient_determination(ys, regression_line)
print(r_squared)

known_x = 9
unknown_y = predict_y(known_x, m, b)

plt.scatter(xs,ys)
plt.scatter(known_x, unknown_y, color = 'g')
plt.plot(xs, regression_line)
plt.show()
