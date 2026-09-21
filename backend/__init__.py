from backend import config

# Constrain native OpenMP and MKL thread pools for local runtime stability
config.setup_runtime_environment()

from flask import Flask, render_template, jsonify

from backend.routes.api import api_bp
from backend.services.knowledge_base import KnowledgeBase


def create_app():
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    app.config["KNOWLEDGE_BASE"] = KnowledgeBase.from_env()
    app.config.setdefault("MAX_CONTENT_LENGTH", config.MAX_CONTENT_LENGTH)

    app.register_blueprint(api_bp)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "Uploaded image file exceeds the maximum allowed size of 16MB.",
            "success": False,
        }), 413

    @app.get("/health")
    def health():
        return jsonify({
            "status": "healthy",
            "app": "PicWise",
            "version": "1.0.0",
        }), 200

    @app.get("/")
    def home():
        return render_template("home.html", active_page="dashboard")

    @app.get("/login")
    def login():
        return render_template("login.html", active_page="login")

    @app.get("/upload")
    def upload():
        return render_template("upload.html", active_page="scan")

    @app.get("/compare")
    def compare():
        return render_template("compare.html", active_page="compare")

    return app
