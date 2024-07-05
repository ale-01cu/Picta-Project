@echo off
call ../.venv\Scripts\activate
start cmd /k "python manage.py runserver"
start cmd /k "cd client && npm run dev"