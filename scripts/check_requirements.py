"""Check if required tools are installed."""

import os
import sys

print("[1/3] Python:")
print(f"  Python {sys.version.split()[0]}")

print("[2/3] Docker:")
docker_ok = os.system("docker --version") == 0

print("[3/3] Docker Compose:")
compose_ok = os.system("docker compose version") == 0

if not docker_ok or not compose_ok:
    print("\nERROR: Install missing tools.")
    sys.exit(1)
