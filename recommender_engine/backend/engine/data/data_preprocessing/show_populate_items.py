import matplotlib.pyplot as plt
import pandas as pd
from engine.data.DataPipeline import DataPipeline

def show_populate_items(df, column_name):
    # Contar la cantidad de veces que aparece cada publication_id
    counts = df.groupby(column_name).size().reset_index(name='counts')
    
    # Ordenar los resultados por la cantidad de publicaciones en orden descendente
    sorted_counts = counts.sort_values(by='counts', ascending=False)

    print(sorted_counts[sorted_counts[column_name] == 387])

    # Seleccionar las 10 publicaciones más frecuentes
    top_publications = sorted_counts.head(20)
    

    print(top_publications)
    
    # Graficar los resultados
    plt.figure(figsize=(10, 6))
    plt.barh(top_publications[column_name], top_publications['counts'], color='skyblue')
    plt.xlabel('Count')
    plt.ylabel('Publication ID')
    plt.title('Top 10 Most Frequent Publications')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pipe = DataPipeline()
    df, = pipe.read_csv_data(paths=[
        "../../../../datasets/vistas_no_nulas.csv"
    ])
    colunm_name = 'publicacion_id'

    show_populate_items(df=df, column_name=colunm_name)