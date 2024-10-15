import os
import sys
import signal
import psycopg2
import numpy as np
from numpy.linalg import norm
from flask import Flask, request, jsonify
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

app = Flask(__name__)
model = TextEmbedding('BAAI/bge-base-en-v1.5')
client = QdrantClient(host="localhost", port=6333)
collection_name = "descriptions"

db_connection = None

def connect_to_db():
    global db_connection
    try:
        db_connection = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "job_service_dev"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        print("Database connection established.")
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        sys.exit(1)

def initialize_collections():
    if not client.collection_exists(collection_name):
       client.create_collection(
          collection_name=collection_name,
          vectors_config=VectorParams(size=768, distance=Distance.COSINE),
       )

@app.route('/embed', methods=['POST'])
def embed_into_qdrant():
    data = request.get_json()

    description = data.get('description')
    if not description:
        return jsonify({'error': 'No description provided'}), 400

    job_id = data.get('id')
    if not job_id:
        return jsonify({'error': 'No job_id provided'}), 400

    embedding = list(model.embed([description]))[0]

    search_result = client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=1,
        with_payload=True,
        with_vectors=True
    )

    upsert_and_return_response = get_upsert_and_return_response(job_id, embedding, description)
    if not search_result:
        return upsert_and_return_response()

    sought_datapoint = search_result[0]

    if sought_datapoint.vector is None or sought_datapoint.payload is None:
        return jsonify({'error': 'Datapoint has no vector or payload'}), 404

    close_vector = np.array(sought_datapoint.vector)
    close_id = sought_datapoint.id
    close_description = sought_datapoint.payload.get('description')
    if np.array_equal(close_vector, embedding):
        return get_similar_description_and_id_json(close_id, close_description)

    try:
        companies, titles, urls = get_job_details(str(sought_datapoint.id))
    except Exception as e:
        return jsonify({'errors': e}), 404

    identity_threshold = 0.995

    
    url = data.get('url')
    if field_is_substring_of_fields(url, urls):
        identity_threshold -= 0.15

    company = data.get('company')
    if field_is_substring_of_fields(company, companies):
        identity_threshold -= 0.05

    title = data.get('title')
    if field_is_substring_of_fields(title, titles):
        identity_threshold -= 0.025

    if satisfies_cos_sim_threshold(close_vector, embedding, identity_threshold):
        return get_similar_description_and_id_json(close_id, close_description), 201
    else:
        return upsert_and_return_response()


def get_upsert_and_return_response(job_id, embedding, description):
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

def get_job_details(job_id: str):
    if db_connection is None:
        raise RuntimeError("Cannot get job details because database connection is not initialized.")

    try:
        cursor = db_connection.cursor()
        query = "SELECT company, title, url FROM job_skillsets WHERE job_id = %s"
        cursor.execute(query, (job_id,))
        rows = cursor.fetchall()

        companies = populate_array_with_non_nulls_from_nth_column(0, rows)
        titles = populate_array_with_non_nulls_from_nth_column(1, rows)
        urls = populate_array_with_non_nulls_from_nth_column(2, rows)

        cursor.close()
        return companies, titles, urls

    except Exception as e:
        print(f"Error while retrieving data: {e}")
        raise e

def field_is_substring_of_fields(field, fields):
    return field and len(fields) > 0 and any(field in other_field for other_field in fields)

def populate_array_with_non_nulls_from_nth_column(n, rows):
    return [row[n] for row in rows if row[n] is not None]

def satisfies_cos_sim_threshold(a, b, threshold):
    cos_sim = np.dot(a, b) / (norm(a) * norm(b))

    return cos_sim > threshold

def get_similar_description_and_id_json(id, description):
    return jsonify({
        'id': id,
        'description': description
    })

def graceful_shutdown(signum, frame):
    connect_to_db()

    print("Shutting down gracefully...")
    if db_connection:
        db_connection.close()
        print("Database connection closed.")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

if __name__ == '__main__':
    connect_to_db()
    initialize_collections()
    app.run(host='0.0.0.0', port=5000)
