class StringText():
    def __init__(self) -> None:
        self.datatype = str

    def __str__(self) -> str:
        return f"{self/__class__.__name__} {self.datatype}"

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Text):
            return self.datatype == value.datatype
        return False