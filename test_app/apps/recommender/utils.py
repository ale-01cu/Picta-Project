import json
import os

def update_json(file_path, data):
    """
    Crea un archivo JSON si no existe o lo modifica si existe.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    else:
        with open(file_path, 'r+') as f:
            json_data = json.load(f)
            json_data.update(data)
            f.seek(0)
            json.dump(json_data, f, indent=4)
            f.truncate()


def read_json(file_path):
    """
    Lee un archivo JSON y devuelve sus datos.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data