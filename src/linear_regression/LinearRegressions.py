
import pandas as pd
from statsmodels.formula.api import ols

class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        formula = f"{self.left_hand_side.name} ~ Mkt_RF + SMB + HML"
        model = ols(formula=formula, data=self.right_hand_side).fit()
        self._model = model

    def get_params(self):
        beta_params = self._model.params.rename("Beta coefficients")
        return beta_params

    def get_pvalues(self):
        p_values = self._model.pvalues.rename("P-values for the corresponding coefficients")
        return p_values

    def get_wald_test_result(self, constraint_matrix):
        wald_result = self._model.wald_test(constraint_matrix)
        f_value = wald_result.statistic[0, 0]
        p_value = wald_result.pvalue
        result_text = f"F-value: {f_value:.3f}, p-value: {p_value:.3f}"
        return result_text








