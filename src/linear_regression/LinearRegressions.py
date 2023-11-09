
import pandas as pd
from statsmodels.formula.api import ols
import statsmodels.api as sm
class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self._model = model

    def get_params(self):
        beta_params = self._model.params.rename("Beta coefficients")
        return beta_params

    def get_pvalues(self):
        p_values = self._model.pvalues.rename("P-values for the corresponding coefficients")
        return p_values

    def get_wald_test_result(self, restriction_matrix):
        wald_result = self._model.wald_test(restriction_matrix)
        f_value = round(wald_result.statistic[0, 0], 2)
        p_value = wald_result.pvalue
        result_text = f"F-value: {f_value:.2f}, p-value: {p_value:.3f}"
        return result_text

    def get_model_goodness_values(self):
        ars = self._model.rsquared_adj
        ak = self._model.aic
        by = self._model.bic
        result_text = f"Adjusted R-squared: {ars:.3f}, Akaike IC: {ak:.2e}, Bayes IC: {by:.2e}"
        return result_text

import pandas as pd
import numpy as np
from scipy.stats import t, f
from typing import List

import numpy as np
import pandas as pd
import statsmodels.api as sm

class LinearRegressionNP:

    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None
        self.coefficients = None

    def fit(self):
        # Using statsmodels for OLS regression
        X = sm.add_constant(self.right_hand_side)
        model = sm.OLS(self.left_hand_side, X).fit()
        self._model = model
        self.coefficients = model.params

    def get_params(self) -> pd.Series:
        if self.coefficients is not None:
            param_names = self.coefficients.index.tolist()
            beta_params = pd.Series(self.coefficients.values, index=param_names, name="Beta coefficients")
            return beta_params
        else:
            raise ValueError("Model coefficients are not available. Fit the model first.")


    def get_pvalues(self):
        p_values = self._model.pvalues.rename("P-values for the corresponding coefficients")
        return p_values

    def get_wald_test_result(self, restriction_matrix):
        wald_result = self._model.wald_test(restriction_matrix)
        wald_value = round(wald_result.statistic[0, 0], 3)
        p_value = wald_result.pvalue
        result_text = f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"
        return result_text

    def get_model_goodness_values(self):
        centered_r_squared = self._model.rsquared
        adjusted_r_squared = self._model.rsquared_adj
        result_text = f"Centered R-squared: {centered_r_squared:.3f}, Adjusted R-squared: {adjusted_r_squared:.3f}"
        return result_text















