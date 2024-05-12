from .Categorical import Categorical

class CategoricalContinuous(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datetype = float