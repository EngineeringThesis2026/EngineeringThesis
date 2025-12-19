# Interactive Legal Advisory System

An AI-powered system for interactive legal consultations using natural language processing and vector search capabilities.

**PL:** Interaktywny system doradczy do konsultacji prawnych oparty na sztucznej inteligencji.

## Features

- Interactive chat interface powered by GPT models
- Vector-based legal document search using Qdrant
- Streamlit web interface for easy interaction
- RAG (Retrieval-Augmented Generation) for accurate legal advice

## Requirements

- Docker and Docker Compose
- Python 3.11.14+ (required for scraper)
- OpenAI API key
- Make (optional, for simplified commands)

## Installing Make

Make is a build automation tool that simplifies running project commands. It comes pre-installed on most Linux and macOS systems, but requires manual installation on Windows.

### Windows

**Option 1: Using winget (recommended - simplest)**
```powershell
winget install GnuWin32.Make
```
*Note: winget is pre-installed on Windows 11 and updated Windows 10*

**Option 2: Using Chocolatey**
```powershell
choco install make
```
*Note: Requires Chocolatey to be installed first - see [chocolatey.org/install](https://chocolatey.org/install)*

**Option 3: Using Scoop**
```powershell
scoop install make
```
*Note: Requires Scoop to be installed first*

After installation, restart your terminal and verify with:
```bash
make --version
```

### Linux (Debian/Ubuntu)

Make is usually pre-installed. If not:
```bash
sudo apt update
sudo apt install make
```

### macOS

Make is included with Xcode Command Line Tools:
```bash
xcode-select --install
```

Or using Homebrew:
```bash
brew install make
```

## Quick Start with Make (Recommended)

The easiest way to run the project:

1. Clone the repository:
```bash
git clone <repository-url>
cd EngineeringThesis
```

2. Run setup (checks requirements and installs dependencies):
```bash
make setup
```
This will verify Python, Docker, Docker Compose are installed and install all Python packages.

3. Edit `.streamlit/secrets.toml` and add your OpenAI API key.

4. Run the application:
```bash
make run
```

This command will automatically:
- Download legal rulings (scraper)
- Build and start Docker containers
- Open browser at http://localhost:8501

### Available Make Commands

| Command        | Description                                      |
|----------------|--------------------------------------------------|
| `make setup`   | Check requirements and install dependencies      |
| `make run`     | Run the full application (scrape + Docker)       |
| `make stop`    | Stop all running containers                      |
| `make restart` | Restart the application                          |
| `make logs`    | View application logs                            |
| `make lint`    | Check, auto-fix and verify code style (Dev)      |
| `make help`    | Display available commands                       |

**Common operations:**
```bash
make stop     # Stop the application
make logs     # View logs (useful for debugging)
make restart  # Quick restart
```

## Alternative: Using Docker Commands Directly

If you prefer not to use Make, you can run everything manually:

### Initial Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd EngineeringThesis
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Streamlit secrets:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml` and add your OpenAI API key.

### Running the Application

4. Download legal rulings:
```bash
python app/scraper.py
```

5. Start Docker containers:
```bash
docker compose up --build
```

6. Access the application at http://localhost:8501

### Common Operations

```bash
# Stop the application
docker compose down

# View logs
docker compose logs -f

# Restart containers
docker compose restart
```

**Docker services include:**
- Streamlit application (Python 3.11.14) on port 8501
- Qdrant vector database on port 6333
- Persistent storage for vector data

## Advanced: Local Setup Without Docker

For advanced users who want to run without Docker containers:

1. Ensure Python 3.11.14 is installed

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Streamlit secrets:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml` and add your OpenAI API key.

4. Start Qdrant manually:
```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

5. Run the Streamlit application:
```bash
streamlit run app/ui_streamlit.py
```

## Project Structure

```
EngineeringThesis/
├── .dockerignore              # Docker ignore patterns
├── .gitignore                 # Git ignore patterns
├── compose.yaml               # Docker Compose configuration
├── Dockerfile                 # Docker container definition
├── Makefile                   # Build automation
├── environment.yml            # Conda environment specification
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── .streamlit/
│   ├── secrets.toml           # OpenAI API key (not in git)
│   └── secrets.toml.example   # Template for secrets file
│
├── scripts/                   # Makefile helper scripts
│   ├── check_config.py        # Copy secrets.toml if missing
│   ├── check_requirements.py  # Verify Python, Docker, Compose
│   └── open_browser.py        # Open browser after startup
│
└── app/
    ├── llm_model.py           # GPT model interface
    ├── process_data.py        # PDF loading and embeddings
    ├── prompt_texts.py        # LLM prompt templates
    ├── scraper.py             # Court rulings downloader
    ├── ui_streamlit.py        # Streamlit web interface
    ├── upload_pdf_file.py     # PDF upload handling
    ├── vector_database.py     # Qdrant vector database client
    └── data/                  # Legal PDF documents
```