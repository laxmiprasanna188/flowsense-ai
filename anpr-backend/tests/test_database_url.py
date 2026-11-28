"""Checks that the backend refuses to invent a database password."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def run_py(code, database_url):
    env = os.environ.copy()
    if database_url is None:
        env.pop("DATABASE_URL", None)
    else:
        env["DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND,
        env=env,
        capture_output=True,
        text=True,
    )


class DatabaseUrlTests(unittest.TestCase):
    def test_missing_url_fails_with_clear_error(self):
        # An empty value is still "not set", even if a .env file exists,
        # because a real environment variable wins over .env.
        result = run_py("import app.database", "")
        combined = result.stderr + result.stdout
        self.assertNotEqual(result.returncode, 0, combined)
        self.assertIn("DATABASE_URL is not set", combined)
        self.assertNotIn("buildathon:", combined)

    def test_plain_postgresql_url_uses_psycopg2_driver(self):
        code = (
            "from app.database import DATABASE_URL\n"
            "host = DATABASE_URL.split('@', 1)[1]\n"
            "assert DATABASE_URL.startswith('postgresql+psycopg2://'), DATABASE_URL\n"
            "assert host == 'localhost:5432/buildathon', host\n"
            "print('ok')\n"
        )
        result = run_py(
            code,
            "postgresql://localuser:localpass@localhost:5432/buildathon",
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("ok", result.stdout)
        self.assertNotIn("localpass", result.stdout)

    def test_psycopg2_url_is_kept(self):
        url = "postgresql+psycopg2://localuser:localpass@localhost:5432/buildathon"
        code = (
            "import os\n"
            "from app.database import DATABASE_URL\n"
            "assert DATABASE_URL == os.environ['DATABASE_URL']\n"
            "print('ok')\n"
        )
        result = run_py(code, url)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
