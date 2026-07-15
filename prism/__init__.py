"""PRISM Worklet 8 — unified inference service.

One Flask app serving three real models through ONNX Runtime. The previous
deployment was a launcher page that reported every app as running and linked to
routes that returned 404; nothing here reports health it has not checked.
"""

from __future__ import annotations

import os

from flask import Flask

__version__ = "2.0.0"


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
        static_url_path="/static",
    )
    app.config.update(
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,  # batch CSV ceiling
        TEMPLATES_AUTO_RELOAD=bool(os.environ.get("PRISM_DEV")),
    )

    from .routes import bp

    app.register_blueprint(bp)
    return app
