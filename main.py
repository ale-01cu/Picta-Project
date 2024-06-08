from generator.main import (
    generate_ratings_with_timestamp,
    generate_item_to_item_dataset,
    generate_user_clicks_history,
    generate_candidate_sequence,
    generate_likes_with_timestamp,
    generate_positive_features_with_timestamp
)

import pandas as pd
import csv


def to_tsv(filename):
    df = pd.read_csv('I:/UCI/tesis/Picta-Project/datasets/picta_publicaciones_procesadas_sin_nulas_v2.csv')
    data = dict(df)
    del data['Unnamed: 0']
    # df = df.drop('descripcion', axis=1)
    df = df.drop('nombre', axis=1)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # Escribe los encabezados de las columnas
        writer.writerow(data.keys())
        # Escribe los datos
        writer.writerows(zip(*data.values()))




def main():
    generate_likes_with_timestamp()


if __name__ == '__main__':
    main()