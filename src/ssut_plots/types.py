import typing as T

SimulationType = T.Literal["SIMBA", "SWIMBA"]

CamelsSimulationType = T.Literal["SIMBA", "Swift-EAGLE", "IllustrisTNG", "Astrid"]
CamelsSet = T.Literal["CV", "1P", "LH"]
Volume = T.Literal[25, 50]


class OneP:
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
    def __init__(self, run_number) -> None:
        if run_number not in range(27):
            raise ValueError("The CV set ranges from 0 to 26")
        self.run = run_number

    def __str__(self) -> str:
        return f"CV_{self.run}"


class LH:
    def __init__(self, run_number) -> None:
        if run_number not in range(1000):
            raise ValueError("The LH set ranges from 0 to 999")
        self.run = run_number

    def __str__(self) -> str:
        return f"LH_{self.run}"
