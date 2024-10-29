from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Crear un objeto engine que se conecte a la base de datos SQLite
engine = create_engine('sqlite:///engine/engine_db.db')
Base = declarative_base()


SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
