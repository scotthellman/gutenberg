from fastapi import FastAPI
from pydantic import BaseModel

from gutenrag.rag import rag

app = FastAPI()


class Query(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query")
def read_item(query: Query):
    result = rag(query.query)
    return {"result": result}
