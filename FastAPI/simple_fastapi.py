from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Pydantic model for POST data
class Item(BaseModel):
    name: str
    price: float

# In-memory storage
items = {}

# GET endpoint
@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id in items:
        return {"item_id": item_id, "item": items[item_id]}
    return {"error": "Item not found"}

# POST endpoint
@app.post("/items/")
def create_item(item_id: int, item: Item):
    items[item_id] = item.dict()
    return {"message": "Item created successfully", "item_id": item_id, "item": items[item_id]}

# To Test the API:
# GET:  http://127.0.0.1:8000/items/1
# POST: http://127.0.0.1:8000/items/?item_id=1
# Body: {"name": "Laptop", "price": 1200.5}
