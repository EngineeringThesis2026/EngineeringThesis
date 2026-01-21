# Makefile

# Detect OS and set Python interpreter
ifeq ($(OS),Windows_NT)
    PYTHON = python
else
    PYTHON = python3
endif

.PHONY: run run-only stop lint help scrape setup logs restart

# Default command - display help
help:
	@echo "Available commands:"
	@echo "  make setup   - Check requirements and install dependencies"
	@echo "  make run     - Run the application (scrape + Docker)"
	@echo "  make stop    - Stop the application"
	@echo "  make restart - Restart the application"
	@echo "  make logs    - View application logs"
	@echo "  make lint    - Check, fix and verify code style (For developers)"
	@echo "  make help    - Show this help message"

# Setup development environment
setup:
	@echo "### Checking system requirements ###"
	@$(PYTHON) scripts/check_requirements.py
	@echo "### Installing Python dependencies ###"
	$(PYTHON) -m pip install -r requirements.txt
	@echo "### Checking configuration ###"
	@$(PYTHON) scripts/check_config.py
	@echo "### Setup complete! ###"
	@echo "Next steps:"
	@echo "  1. Edit .streamlit/secrets.toml and add OPENAI API key"
	@echo "  2. Run 'make run' to start the application"

# Scrape court rulings (hidden from help)
scrape:
	@echo "Downloading rulings..."
	$(PYTHON) app/scraper.py 5

# Main command: scrape + run Docker
run: scrape run-only

# Run Docker only (hidden from help)
run-only:
	@echo "Starting application..."
	docker compose up --build -d
	@$(PYTHON) scripts/open_browser.py
	@echo "Showing logs (Ctrl+C to exit)..."
	docker compose logs -f

# Stop the application
stop:
	@echo "Stopping..."
	docker compose down

# Check and fix code style
lint:
	@echo "Checking code style..."
	-black --check .
	-flake8 .
	@echo "Applying fixes..."
	black .
	@echo "Verifying..."
	flake8 .

# View application logs
logs:
	@echo "Viewing logs..."
	docker compose logs -f

# Restart the application
restart:
	@echo "Restarting..."
	docker compose restart