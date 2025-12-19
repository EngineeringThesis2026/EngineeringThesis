"""Copy secrets.toml.example to secrets.toml if it does not exist."""
from pathlib import Path

SOURCE = Path(".streamlit/secrets.toml.example")
TARGET = Path(".streamlit/secrets.toml")

if not TARGET.exists():
    TARGET.write_text(SOURCE.read_text())
    print(f"Created {TARGET} from {SOURCE}")
else:
    print(f"OK: {TARGET} exists")
