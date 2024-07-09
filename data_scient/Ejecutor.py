import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class Ejecutor:
    @staticmethod
    async def ejecutar_async(funcion, args_list):
        async def wrapper(args):
            if asyncio.iscoroutinefunction(funcion):
                return await funcion(*args)
            else:
                return await asyncio.to_thread(funcion, *args)

        inicio = time.time()
        tareas = [wrapper(args) for args in args_list]
        resultados = await asyncio.gather(*tareas)
        fin = time.time()
        tiempo_total = fin - inicio
        return resultados, tiempo_total

    @staticmethod
    def ejecutar_paralelo(funcion, args_list, num_nucleos=None):
        if num_nucleos is None:
            num_nucleos = multiprocessing.cpu_count()

        inicio = time.time()
        with ProcessPoolExecutor(max_workers=num_nucleos) as executor:
            resultados = list(executor.map(funcion, args_list))
        fin = time.time()
        tiempo_total = fin - inicio
        return resultados, tiempo_total

def ejecutar_async(funcion, args_list):
    return asyncio.run(Ejecutor.ejecutar_async(funcion, args_list))

# def ejecutar_paralelo(funcion, args_list, num_nucleos=None):
#     return Ejecutor.ejecutar_paralelo(funcion, args_list, num_nucleos)


def ejecutar_paralelo(num_nucleos, task, args):
    inicio = time.time()
    with ProcessPoolExecutor(max_workers=num_nucleos) as executor:
        resultados = list(executor.map(task, args))
    fin = time.time()
    print(f"Paralelo ({num_nucleos} núcleos): {fin - inicio:.2f} segundos")
    return resultados