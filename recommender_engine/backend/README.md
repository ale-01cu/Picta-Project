# Recommendation System Project Documentation

## 1. Overview

This project is a recommendation system built with FastAPI and TensorFlow. It provides an API to train, manage, and use recommendation models. The system is designed around a multi-stage pipeline that includes candidate retrieval and ranking to generate personalized recommendations.

The project also includes a simple web interface for managing model configurations and recommendation engines.

## 2. Technologies Used

- **Backend:** FastAPI
- **Machine Learning Engine:** TensorFlow, TensorFlow Recommenders
- **Data Manipulation:** Pandas, NumPy
- **Database:**
  - **MongoDB:** To store configurations for models, engines, users, and platform tokens.
  - **SQLite:** Used by SQLAlchemy for metadata management of models and engines (although there seems to be a dual configuration with `settings/db.py` and `engine/db/config.py`).
- **Frontend (Templates):** Jinja2
- **Authentication:** JWT (JSON Web Tokens) with Passlib for password hashing.

## 3. Project Structure

The project is organized into the following main directories within `backend/`:

- **`app/`**: Contains the logic of the FastAPI application.
  - **`middlewares/`**: FastAPI middlewares, such as authentication (`auth.py`).
  - **`routes/`**: Defines the API endpoints for different functionalities (authentication, configuration, engine actions, etc.).
  - **`schemas/`**: Contains Pydantic schemas for input and output data validation.
  - **`templates/`**: Jinja2 HTML templates for the user interface.

- **`engine/`**: The core of the recommendation system.
  - **`actions/`**: Scripts for training, fine-tuning, and using the models.
  - **`data/`**: Classes and functions for the data pipeline, including preprocessing and feature handling.
  - **`db/`**: Database configuration (SQLAlchemy) and CRUD operations for models and engines.
  - **`models/`**: Defines the TensorFlow Recommenders models (`RetrievalModel`, `LikesModel`) and configuration classes.
  - **`stages/`**: Defines the stages of the recommendation pipeline (retrieval, ranking).

- **`settings/`**: Configuration files for the database (MongoDB and SQLAlchemy).

## 4. Key Components

### 4.1. API Endpoints (Routes)

The API provides the following groups of endpoints:

- **Authentication (`auth_routes.py`):**
  - `/signup`: Registration of new users.
  - `/signin`: Login and generation of JWT tokens.

- **Model Configuration (`config_routes.py`):**
  - `/config`: Lists, creates, updates, and deletes model configurations.

- **Recommendation Engines (`engine_routes.py`):**
  - `/engines`: CRUD for recommendation engines, which combine retrieval and ranking models.

- **Engine Actions (`engine_actions_routes.py`):**
  - `/train/{engine_id}`: Starts the training process for a specific engine.
  - `/tunning/{engine_id}`: Starts the fine-tuning process.
  - `/recommend/{user_id}`: Generates recommendations for a user.

- **Others:**
  - `features_routes.py`: Lists the available feature types.
  - `authorizate_plataforms.py`: Manages API tokens for external platforms.
  - `home.py`: Routes for the home page and user data visualization.

### 4.2. Recommendation Engine

The recommendation engine follows a two-stage pipeline:

1.  **Retrieval:**
    - **Model:** `engine/models/RetrievalModel.py`.
    - **Purpose:** To generate a large set of relevant candidates from the entire item catalog.
    - **Logic:** Uses a two-tower model to learn embeddings for users and items. The task is to predict which items a user has seen.

2.  **Ranking:**
    - **Model:** `engine/models/LikesModel.py`.
    - **Purpose:** To take the candidates from the retrieval stage and rank them to produce an ordered list of recommendations.
    - **Logic:** It is a classification model that predicts the probability of a user "liking" an item. It uses a deep neural network on the concatenated embeddings of the user and the item.

### 4.3. Data Pipeline (`engine/data/DataPipeline.py`)

This class is responsible for:
- Reading data from CSV files.
- Converting Pandas DataFrames to `tf.data.Dataset`.
- Building vocabularies for categorical features.
- Splitting the data into training, validation, and test sets.
- Caching the datasets for efficient training.

### 4.4. Persistence

- **MongoDB:** It is the main database for configuration. It stores:
  - `Engine`: Configurations of the recommendation engines.
  - `ModelConfigCollection`: Detailed configurations of the models (hyperparameters, data paths, etc.).
  - `users`: User credentials.
  - `plataform_tokens`: API tokens for other platforms.

- **SQLite (via SQLAlchemy):** Appears to be a legacy or developing system for managing metadata of models and engines, with CRUD operations defined in `engine/db/cruds/`.

## 5. Getting Started

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the database:**
    - Make sure MongoDB is running at `mongodb://localhost:27017/`.
    - The SQLite database (`engine_db.db`) will be created automatically.

3.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```

4.  **Interact with the application:**
    - Access `http://127.0.0.1:8000` in your browser to see the web interface.
    - Use the interface to create and manage model and engine configurations.
    - Trigger the training and recommendation processes through the UI or directly via the API.
