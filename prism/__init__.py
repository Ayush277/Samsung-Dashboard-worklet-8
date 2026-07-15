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

    @app.context_processor
    def brand():
        return {"brand_logo": find_brand_logo()}

    return app


# Drop the official Samsung PRISM logo at prism/static/brand/logo.svg (or .png)
# and every page picks it up. Ordered by preference: SVG scales to any density.
_BRAND_CANDIDATES = ("logo.svg", "logo.png", "logo.webp", "logo.jpg")


def find_brand_logo() -> str | None:
    """Return the brand logo's static filename, or None if it isn't present.

    Templates fall back to a typographic lockup when this is None, so a missing
    asset renders as deliberate wordmark rather than a broken image icon.
    """
    brand_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "static", "brand")
    for name in _BRAND_CANDIDATES:
        if os.path.exists(os.path.join(brand_dir, name)):
            return f"brand/{name}"
    return None
