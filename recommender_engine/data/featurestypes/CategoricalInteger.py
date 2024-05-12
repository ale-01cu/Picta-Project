from .Categorical import Categorical

class CategoricalInteger(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datetype = int