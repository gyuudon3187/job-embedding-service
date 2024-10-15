import os
import sys
import psycopg2
from utils.helpers import populate_array_with_non_nulls_from_nth_column

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
