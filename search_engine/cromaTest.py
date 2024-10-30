import chromadb

# Initialize ChromaDB client
# Initialize ChromaDB client
client = chromadb.Client()

# Creating a collection
neo_collection = client.create_collection(name="neo")

# Inspecting a collection
print(neo_collection)

# Changing the collection name and inspecting it again
neo_collection.modify(name="mr_anderson")
print(neo_collection)

# Counting items in a collection
item_count = neo_collection.count()
print(f"Count of items in collection: {item_count}")

# Get or Create a new collection, change the distance function
trinity_collection = client.get_or_create_collection(
    name="trinity", metadata={"hnsw:space": "cosine"}
)
print(trinity_collection)



print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # Adding raw documents
trinity_collection.add(
    documents=["I know kung fu.", "There is no spoon."], ids=["quote_1", "quote_2"]
)

# Counting items in a collection
item_count = trinity_collection.count()
print(f"Count of items in collection: {item_count}")

# Get items from the collection
items = trinity_collection.get()
print(items)

# Or we can use the peek method
trinity_collection.peek(limit=5)
