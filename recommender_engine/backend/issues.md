### 1 - Simpre recomienda lo mismo.

## Nombres del problema: "popularity bias" or "item dominance" problem

# Pregunta para AI 
tengo un problema muy serio, estoy creando un sistema de recomendaciones utilizando la libreria tensorflow, keras y tensorflow_recommenders de google, ya el recomendador funciona, osea se entrena con los datos y genera las recomendaciones, hasta ahora solo tiene un modelo de recuperacion para los datos implicitos y un modelo de renking para los datos explicitos, de datos explicitos estoy utilizando likes de publicaciones, para los implicitos el historial de clicks de los usuarios y los candidatos son las publicaciones, ahora el modelo me esta dando un problema muy serio y es que siempre recomienda lo mismo sea al usuario que sea una vez es entrenado. Investigando un poco la puntuacion que esta devolviendo para cada candidato generado al parecer hay candidatos que sobrepasan mucho por encima del resto por igual para todos los usuarios, osea segun tengo entendido esto significa que esos candidatos son muy similares a todos los vectores generados por los datos de contexto (usuario o query) por tanto siempre salen como recomendados, como puedo solucionar este problema ??

# Posibles soluciones
- Agregar una capa de regularizacion L2 al final de cada torre. x
- Average Recommendation Popularity (ARP)
- The Average Percentage of Long Tail Items (APLT) metric 
- The Average Coverage of Long Tail items (ACLT) metric 
- Position Aware Learning (PAL)
- xQUAD Framework

## Problema Revuelto
Se debe a que los datos que le estaba pasando de prueba no estaban dentro de los datos con los que se entreno el modelo por tanto el mismo recomendaba los candidatos mas convenientes para todo el mundo. Esto es una tecnica de arranque en frio.