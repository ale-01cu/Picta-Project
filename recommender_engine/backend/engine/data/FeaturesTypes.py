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
    
    def cast(self, value):
        try:
            if self.datatype == tf.int64:
                # return tf.cast(value, tf.int64)
                return int(value)
            elif self.datatype == tf.string:
                return str(value)
                # return tf.convert_to_tensor(value, dtype=tf.string)
            else:
                raise ValueError("Tipo de dato no soportado para casteo")
        except (ValueError, TypeError) as e:
            print(f"Error al castear {value} a {self.datatype}: {e}")
            return None
    

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
    
    def cast(self, value):
        try:
            # return tf.convert_to_tensor(value, dtype=tf.string)
            return str(value)
        except (ValueError, TypeError) as e:
            print(f"Error al castear {value} a {self.datatype}: {e}")
            return None
    

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
