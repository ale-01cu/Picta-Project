from engine.data.DataPipeline import DataPipeline
import matplotlib.pyplot as plt

def show_likes_distribution():
    pipe = DataPipeline()
    likes_df, = pipe.read_csv_data(paths=[
        "../../../../datasets/likes.csv"
    ])
    print(likes_df)


    likes = likes_df['valor'].value_counts()[1]
    dislikes = likes_df['valor'].value_counts()[0]

    print(f"Likes: {likes}")
    print(f"Dislikes: {dislikes}")

    prop_likes = likes / (likes + dislikes)
    prop_dislikes = dislikes / (likes + dislikes)

    print(f"Proporción de likes: {prop_likes:.2f}")
    print(f"Proporción de dislikes: {prop_dislikes:.2f}")

    likes_df['valor'].value_counts().plot(kind='bar')
    plt.title('Distribución de likes y dislikes')
    plt.xlabel('valor')
    plt.ylabel('Frecuencia')
    plt.show()


if __name__ == "__main__":
    show_likes_distribution()