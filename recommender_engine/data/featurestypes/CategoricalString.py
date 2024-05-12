from .Categorical import Categorical

class CategoricalString(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datetype = str