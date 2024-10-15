import sys
from flask import Flask, request, jsonify
from routes.embed_routes import embed_blueprint
from services.db_service import connect_to_db, db_connection
from services.qdrant_service import initialize_collections
import signal

app = Flask(__name__)
app.register_blueprint(embed_blueprint)

def graceful_shutdown(signum, frame):

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
