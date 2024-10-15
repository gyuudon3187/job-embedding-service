from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

model = TextEmbedding('BAAI/bge-base-en-v1.5')
client = QdrantClient(host="localhost", port=6333)
collection_name = "descriptions"

def initialize_collections():
    if not client.collection_exists(collection_name):
       client.create_collection(
          collection_name=collection_name,
          vectors_config=VectorParams(size=768, distance=Distance.COSINE),
       )

def embed_description(description):
    return list(model.embed([description]))[0]


def get_upsert_and_return_response_func(job_id, embedding, description):
    def upsert_and_return_response():
        return client.upsert(
            collection_name="descriptions",
            points=[
                PointStruct(
                    id=job_id,
                    vector=embedding.tolist(),
                    payload={'description': description}
                )
            ]
        ).model_dump_json(), 201

    return upsert_and_return_response

def search_qdrant(embedding):
    return client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=1,
        with_payload=True,
        with_vectors=True
    )
