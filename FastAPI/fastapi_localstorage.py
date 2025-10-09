from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

app = FastAPI()

# Pydantic model for POST data
class Item(BaseModel):
    name: str
    price: float

# File path for storage
DATA_FILE = "items.json"
# Helper functions to read/write JSON file
def load_items():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}
def save_items(items):
    with open(DATA_FILE, "w") as f:
        json.dump(items, f, indent=4)

# GET endpoint
@app.get("/items/{item_id}")
def read_item(item_id: int):
    items = load_items()
    if str(item_id) in items:
        return {"item_id": item_id, "item": items[str(item_id)]}
    return {"error": "Item not found"}

# POST endpoint
@app.post("/items/")
def create_item(item_id: int, item: Item):
    items = load_items()
    items[str(item_id)] = item.dict()
    save_items(items)
    return {"message": "Item created successfully", "item_id": item_id, "item": items[str(item_id)]}
