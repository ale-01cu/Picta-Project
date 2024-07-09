@echo off
call ../venv_recommender\Scripts\activate
start cmd /k "python manage.py runserver"
start cmd /k "cd client && npm run dev"