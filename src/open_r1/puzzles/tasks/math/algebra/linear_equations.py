import re

import numpy as np

from ....base_config import BaseConfig
from ....base_task import BaseTask


class LinearEquationConfig(BaseConfig):
    min_coefficient: int = -10
    max_coefficient: int = 10
    min_var_value = -10
    max_var_value = 10


class LinearEquationTask(BaseTask):
    config_class = LinearEquationConfig

    def generate_sample(self, rng: np.random.Generator):
        variable_names = ("x", "y", "z", "a", "b", "c")
        config = self.config
        var_name = rng.choice(variable_names)
        var_coefficient = 0
        while var_coefficient == 0:
            # We can't have the variable's coefficient be 0, so keep sampling until we get a non-zero one
            var_coefficient = rng.integers(config.min_coefficient, config.max_coefficient, endpoint=True)
        constant = rng.integers(config.min_coefficient, config.max_coefficient, endpoint=True)
        while var_coefficient == 1 and constant == 0:
            # We can't have the variable's coefficient be 1 and the constant be 0, as this is a trivial equation
            # so keep rerolling until it isn't
            constant = rng.integers(config.min_coefficient, config.max_coefficient, endpoint=True)
        var_value = int(rng.integers(config.min_var_value, config.max_var_value, endpoint=True))
        rhs = var_coefficient * var_value + constant

        base_text = f"Solve for {var_name} in the following equation:\n\n"
        if constant < 0:
            equation = f"{var_coefficient}{var_name} - {-constant} = {rhs}"
        elif constant > 0:
            equation = f"{var_coefficient}{var_name} + {constant} = {rhs}"
        else:
            equation = f"{var_coefficient}{var_name} = {rhs}"

        return {"prompt": base_text + equation, "ground_truth": var_value}

    @staticmethod
    def verify(output, answer):
        # Look for number-like sequences
        numbers = re.findall(r"(\d+|\d+\.\d+)", output)
        if not numbers:
            return 0.0  # If we can't find a number, it's incorrect
        try:
            # We assume the last number in the text is the model's answer, just in case any CoT is in there
            prediction = float(numbers[-1])
        except ValueError:
            return 0.0  # If we can't parse a number, it's incorrect
        return float(prediction == answer)
