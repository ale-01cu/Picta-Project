from typing import Text

class Categorical():
    def __init__(self) -> None:
        self.datetype = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.datatype}"
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, self):
            return self.datatype == value.datatype
        return False
    

class CategoricalContinuous(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datetype = float


class CategoricalInteger(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datetype = int


class CategoricalString(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datetype = str


class StringText():
    def __init__(self) -> None:
        self.datatype = str

    def __str__(self) -> str:
        return f"{self/__class__.__name__} {self.datatype}"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Text):
            return self.datatype == value.datatype
        return False
    

features_types_map = {
    CategoricalContinuous.__name__: CategoricalContinuous,
    CategoricalInteger.__name__: CategoricalInteger,
    CategoricalString.__name__: CategoricalString,
    StringText.__name__: StringText,
}