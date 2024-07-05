from django.core.management.base import BaseCommand
from apps.recommender.tasks import read_dataset
import os
import time

class Command(BaseCommand):
    def handle(self, *args, **options):
        # Ejecutar script aquí
        # read_dataset()
        # Abrimos una consola nueva utilizando el comando "start" en Windows
        # o "open" en macOS/Linux
        if os.name == 'nt':  # Windows
            os.system('start cmd /k python -c "import time while True: print(\"hola\") time.sleep(1)"')
        elif os.name == 'posix':  # macOS/Linux
            os.system('open -a Terminal --args python -c "import time; while True: print(\"hola\"); time.sleep(1)"')

        # Si no se puede abrir una consola nueva, imprimimos un mensaje de error
        else:
            print("No se puede abrir una consola nueva en este sistema operativo")