import typing as T

SimulationType = T.Literal["SIMBA", "SWIFT"]

CamelsSimulationType = T.Literal["SIMBA", "Swift-EAGLE", "IllustrisTNG", "Astrid"]
CamelsSet = T.Literal["CV", "1P", "LH"]
Volume = T.Literal[25, 50]


class OneP:
    """
    The 1P (one parameter) set of CAMELS changes only one pararameter at a time, all running on the same initial conditions.
    There are currently 28 parameters which are varried, and each parameter has 5 possible values.

    Parameters
    ----------
    param_number
        The parameter being varied.
        Must be an integer in the range [1, 28].
    variation
        Which value of the parameter.
        Can be an integer in the range [-2, 2] or a corresponding string ["n2", "n1", "0", "1", "2"].
    """
    def __init__(
        self,
        param_number: int,
        variation: T.Literal["n2", "n1", "0", "1", "2", -2, -1, 0, 1, 2],
    ) -> None:
        if param_number not in range(1, 29):
            raise ValueError(f"The 1P set only has 28 parameters. Not {param_number}")
        self.param = param_number
        if isinstance(variation, int):
            if variation < 0:
                self.variation = f"n{-variation}"
            else:
                self.variation = str(variation)
        else:
            self.variation = variation

    def __str__(self) -> str:
        return f"1P_p{self.param}_{self.variation}"


class CV:
    """
    The CV (cosmic variance) set of CAMELS does not change any parameters.
    Instead, only the initial conditions are varied.

    Parameters
    ----------
    run_number
        The run in the CV set to consider.
        Must be an integer in the range [0, 26].
    """
    def __init__(self, run_number) -> None:
        if run_number not in range(27):
            raise ValueError("The CV set ranges from 0 to 26")
        self.run = run_number

    def __str__(self) -> str:
        return f"CV_{self.run}"


class LH:
    """
    The LH (latin hypercube) set of CAMELS changes 6 parameters at once.

    Parameters
    ----------
    run_number
        The run in the LH set to consider.
        Must be an integer in the range [0, 999].
    """
    def __init__(self, run_number) -> None:
        if run_number not in range(1000):
            raise ValueError("The LH set ranges from 0 to 999")
        self.run = run_number

    def __str__(self) -> str:
        return f"LH_{self.run}"
