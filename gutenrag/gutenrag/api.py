from fastapi import FastAPI
from pydantic import BaseModel

from gutenrag.db import MODELS
from gutenrag.rag import rag

app = FastAPI()


class Query(BaseModel):
    query: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/query")
async def read_item(query: Query):
    result = await rag(query.query, embedding_model=MODELS[0])
    return {"result": result}
