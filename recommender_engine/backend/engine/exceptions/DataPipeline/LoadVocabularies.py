class LoadVocabulariesException(Exception):
    def __init__(self, message="Error al cargar los vocabularios"):
        self.message = message
        super().__init__(self.message)