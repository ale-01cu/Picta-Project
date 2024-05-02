from generator.main import (
    generate_ratings_with_timestamp,
    generate_item_to_item_dataset,
    generate_user_clicks_history
)

def main():
    generate_user_clicks_history()

if __name__ == '__main__':
    main()