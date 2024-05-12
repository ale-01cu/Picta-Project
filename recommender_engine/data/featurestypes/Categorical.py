class Categorical():
    def __init__(self) -> None:
        self.datetype = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.datatype}"
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, self):
            return self.datatype == value.datatype
        return False