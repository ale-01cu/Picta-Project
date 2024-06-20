import pandas as pd

# Carga tu dataset
df = pd.read_csv('ruta_a_tu_archivo.csv')

# Asegúrate de que tu columna de fechas esté en formato datetime
df['fecha'] = pd.to_datetime(df['fecha'])

# Ordena el dataframe por fecha
df = df.sort_values('fecha')

# Agrupa por película y cuenta las vistas en un periodo de tiempo específico (por ejemplo, semanal)
tendencias = df.groupby(['pelicula', pd.Grouper(key='fecha', freq='W')])['vistas'].sum().reset_index()

# Ordena por fecha y vistas para ver las películas con más vistas en cada semana
tendencias = tendencias.sort_values(['fecha', 'vistas'], ascending=[True, False])

# Filtra para obtener las películas en tendencia en la última semana
ultima_semana = tendencias[tendencias['fecha'] == tendencias['fecha'].max()]

# Las películas en tendencia serán las que tienen más vistas en la última semana
peliculas_en_tendencia = ultima_semana.sort_values('vistas', ascending=False)
