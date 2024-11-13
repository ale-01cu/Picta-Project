from typing import Text
import tensorflow as tf

class Categorical():
    def __init__(self) -> None:
        self.datatype = None
        self.label = ""

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.datatype}"
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, self):
            return self.datatype == value.datatype
        return False
    

class CategoricalContinuous(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datatype = tf.int64
        self.label = "Continuo Categorico"


class CategoricalInteger(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datatype = tf.int64
        self.label = "Entero Categorico"


class CategoricalString(Categorical):
    def __init__(self) -> None:
        super().__init__()
        self.datatype = tf.string
        self.label = "Texto Categorico"


class StringText():
    def __init__(self) -> None:
        self.datatype = tf.string
        self.label = "Texto Largo"

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

def map_feature(to_class, feature_type):
    if to_class:
        return features_types_map[feature_type]
    
    else: return feature_type.__name__
