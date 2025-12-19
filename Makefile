# Makefile

.PHONY: run run-only stop lint help scrape

# Default command - display help
help:
	@echo "Available commands:"
	@echo "  make run   - Run the application (scrape + Docker)"
	@echo "  make stop  - Stop the application"
	@echo "  make lint  - Check code style (black, flake8)"
	@echo "  make help  - Show this help message"

# Scrape court rulings (hidden from help)
scrape:
	@echo "Downloading rulings..."
	python app/scraper.py

# Main command: scrape + run Docker
run: scrape run-only

# Run Docker only (hidden from help)
run-only:
	@echo "Starting application..."
	docker compose up --build

# Stop the application
stop:
	@echo "Stopping..."
	docker compose down

# Check code style
lint:
	@echo "Checking code style..."
	black --check .
	flake8 .