class SplitIntoTrainAndTestException(Exception):
    def __init__(self, message="Error al dividir el dataset en entrenamiento y prueba"):
        self.message = message
        super().__init__(self.message)