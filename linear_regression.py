'''
My implementation of linear regression
'''

import numpy as np
import copy as cp


class MultivariateLinearRegression():

    def __init__(self) -> (None):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, x, y):
        # function that will fit the training data
        x = self._transform_x(x)
        y = self._transform_y(y)
        beta = self._estimate_coefficients(x, y)

        # intercept is the 0th component of the beta vector
        self.intercept = beta[0]
        # the coefficients are the rest of the beta vector
        self.coefficients = beta[1:]

    def predict(self, x):

        '''
        y = beta_0*x_0 + beta_1*x_1 + ...
        the beta_0*x_0 term is actually the intercept, which shouldn't be multiplied by x
        need to insert 1s at the 0th component of x to account for this
        '''
        predictions = []

        for index, row in x.iterrows():
            values = row.values
            pred = np.multiply(values, self.coefficients)
            pred = sum(pred)
            pred += self.intercept
            predictions.append(pred)

        return predictions
    
    def r_squared(self, y_true, y_pred):
        '''
        r-squared for evaluating the model
        r2 = 1 - (sum of squared residual errors)/(sum of squared total errors)
        '''

        y_values = y_true.values
        y_avg = np.mean(y_values)

        residual_sum_squares = 0
        total_sum_squares = 0

        for i in range(len(y_values)):
            residual_sum_squares += (y_values[i] - y_pred[i])**2
            total_sum_squares += (y_values[i] - y_avg)**2
        
        return 1 - residual_sum_squares/total_sum_squares
        
 
    def _transform_x(self, x):
        x = cp.deepcopy(x)
        x.insert(0, 'ones', np.ones( (x.shape[0], 1) ) )
        return x.values

    def _transform_y(self, y):
        y = cp.deepcopy(y)
        return y.values
    
    def _estimate_coefficients(self, x, y):
        '''
        estimate beta
        beta = (x^T * x)^-1 * x^T * y
        '''

        xT = np.transpose(x)
        inverse = np.linalg.inv(xT.dot(x))

        coefficients = inverse.dot(xT).dot(y)
        return coefficients

class UnivariateLinearRegression():

    def __init__(self) -> None:
        self.coefficient = None
        self.intercept = None

    def fit(self, x, y):
        self.coefficient = self._coefficient_estimate
        self.intercept = self._compute_intercept
    
    def predict(self, x):
        '''
        y_pred = computed_intercept + computed_coefficient * x
        '''

        x_times_coeff = np.multiply(x, self.coefficient)
        return np.add(x_times_coeff, self.intercept)
            

    def r_squared(self, y_true, y_pred):

        numerator = 0
        denominator = 0

        for i in range(len(y_true)):
            numerator += (y_true[i] - y_pred[i]) **2
            denominator += (y_true - np.average(y_true))**2
        
        return 1 - numerator/denominator

    def _compute_intercept(self, x, y):
        '''
        intercept = y_bar - coefficient*xbar
        '''

        return np.average(y) - self.coefficient * np.average(x)

    def _coefficient_estimate(self, x, y):
        '''
        y = beta0 + beta1*x
        beta1 = cov(x,y)/var(x)

        cov(x,y) = sum( (x-x_avg) * (y-y_avg) ) / (len(x)-1)
        var(x) = sum( (x-x_avg)**2 )/(len(x)-1)
        the len(x)-1 in both denominators cancel, so I will ignore them
        '''
        numerator = 0
        denominator = 0

        for i in range(len(x)):
            xbar = np.average(x)
            ybar = np.average(y)

            numerator += (x[i] - xbar)*(y[i] - ybar)
            denominator += (x[i] - xbar)**2

        self.coefficient = numerator/denominator
        return self.coefficient