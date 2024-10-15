import numpy as np
from flask import Blueprint, request, jsonify
from services.qdrant_service import embed_description, get_upsert_and_return_response_func, search_qdrant, get_upsert_and_return_response_func
from services.db_service import get_job_details
from utils.helpers import satisfies_cos_sim_threshold, field_is_substring_of_fields

embed_blueprint = Blueprint('embed', __name__)

@embed_blueprint.route('/embed', methods=['POST'])
def embed_into_qdrant():
    data = request.get_json()

    description = data.get('description')
    if not description:
        return jsonify({'error': 'No description provided'}), 400

    job_id = data.get('id')
    if not job_id:
        return jsonify({'error': 'No job_id provided'}), 400

    embedding = embed_description(description)
    search_result = search_qdrant(embedding)

    upsert_and_return_response = get_upsert_and_return_response_func(job_id, embedding, description)
    if not search_result:
        return upsert_and_return_response()

    sought_datapoint = search_result[0]

    if sought_datapoint.vector is None or sought_datapoint.payload is None:
        return jsonify({'error': 'Datapoint has no vector or payload'}), 404

    close_vector = np.array(sought_datapoint.vector)
    close_id = sought_datapoint.id
    close_description = sought_datapoint.payload.get('description')
    if np.array_equal(close_vector, embedding):
        return jsonify({'id': close_id, 'description': close_description}), 200

    companies, titles, urls = get_job_details(str(sought_datapoint.id))

    identity_threshold = 0.995
    company = data.get('company')
    if field_is_substring_of_fields(company, companies):
        identity_threshold -= 0.05

    title = data.get('title')
    if field_is_substring_of_fields(title, titles):
        identity_threshold -= 0.025
    
    url = data.get('url')
    if field_is_substring_of_fields(url, urls):
        identity_threshold -= 0.15

    if satisfies_cos_sim_threshold(close_vector, embedding, identity_threshold):
        return jsonify({'id': close_id, 'description': close_description}), 200
    else:
        return upsert_and_return_response()
