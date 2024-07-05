import os
import random

def get_path(filename):
    current_file_path = os.path.dirname(__file__)
    return os.path.join(current_file_path, filename)


def get_id():
    return random.randint(1, 1000000)