"""Vercel serverless entrypoint.

Vercel's Python runtime looks for a WSGI callable named `app` in this module.
The repo root is added to sys.path because the function is invoked from inside
api/, where the `prism` package would otherwise not be importable.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prism import create_app  # noqa: E402

app = create_app()
