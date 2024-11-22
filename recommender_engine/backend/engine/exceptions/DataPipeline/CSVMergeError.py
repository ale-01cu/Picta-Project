class CSVMergeErrorException(Exception):
    """Excepción lanzada cuando no se pueder mergear un .csv."""
    def __init__(self):
        super().__init__(f"Error inesperado al mergear dataframes.")