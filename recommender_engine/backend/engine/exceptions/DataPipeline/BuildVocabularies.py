class BuildVocabulariesException(Exception):
    def __init__(self, message="Error al construir los vocabularios"):
        self.message = message
        super().__init__(self.message)