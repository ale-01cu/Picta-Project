from pymongo import MongoClient

# Conectar a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Recommneder']
engine_collection = db['Engine']
config_collection = db['ModelConfigCollection']
user_collection = db['users']
