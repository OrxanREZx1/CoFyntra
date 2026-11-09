"""Test settings: the normal settings with a throwaway secret and a fast password hasher.

The database still comes from DATABASE_URL (PostgreSQL in compose.yaml); unset it to run on SQLite.
"""

import os

os.environ.setdefault("DJANGO_SECRET_KEY", "test-only-not-a-secret")
# Tests never rely on an implicit backend: they inject MockBackend or test ServiceBackend explicitly.
os.environ["COFYTRA_AI_MODE"] = "service"

from cofytra_backend.settings import *  # noqa: E402,F403

PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
