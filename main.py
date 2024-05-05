from generator.main import (
    generate_ratings_with_timestamp,
    generate_item_to_item_dataset,
    generate_user_clicks_history,
    generate_candidate_sequence,
    generate_likes_with_timestamp
)

def main():
    generate_likes_with_timestamp()

if __name__ == '__main__':
    main()