import numpy as np
from numpy.linalg import norm

def satisfies_cos_sim_threshold(a, b, threshold):
    cos_sim = np.dot(a, b) / (norm(a) * norm(b))
    return cos_sim > threshold

def field_is_substring_of_fields(field, fields):
    return field and len(fields) > 0 and any(field in other_field for other_field in fields)

def populate_array_with_non_nulls_from_nth_column(n, rows):
    return [row[n] for row in rows if row[n] is not None]
