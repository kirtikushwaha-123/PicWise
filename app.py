import os

# Constrain native OpenMP and MKL thread pools for local runtime stability
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from backend import create_app


app = create_app()


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0").strip().lower() in ("1", "true")
    app.run(debug=debug_mode)
