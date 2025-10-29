# Interactive Legal Advisory System

An AI-powered system for interactive legal consultations using natural language processing and vector search capabilities.

**PL:** Interaktywny system doradczy do konsultacji prawnych oparty na sztucznej inteligencji.

## Features

- Interactive chat interface powered by GPT models
- Vector-based legal document search using Qdrant
- Streamlit web interface for easy interaction
- RAG (Retrieval-Augmented Generation) for accurate legal advice

## Requirements

- Docker and Docker Compose (recommended)
- OpenAI API key
- OR Python 3.11.14+ (for local setup without Docker)

## Quick Start with Docker (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd EngineeringThesis
```

2. Configure Streamlit secrets:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Then edit `.streamlit/secrets.toml` and add your OpenAI API key.

3. Place your PDF legal documents in the `app/data/` directory.

4. Start all services with Docker Compose:
```bash
docker-compose up --build
```

5. Access the application at http://localhost:8501

6. To stop the services:
```bash
docker-compose down
```

**Docker services include:**
- Streamlit application (Python 3.11.14) on port 8501
- Qdrant vector database on ports 6333
- Persistent storage for vector data

## Local Setup (Alternative)

If you prefer to run without Docker:

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
streamlit hello
```

## Project Structure

```
EngineeringThesis/
├── .dockerignore              # Docker ignore patterns
├── .gitignore                 # Git ignore patterns
├── compose.yaml               # Docker Compose configuration
├── Dockerfile                 # Docker container definition
├── environment.yml            # Conda environment specification
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
│
├── .streamlit/
│   ├── secrets.toml           # OpenAI API key (not in git)
│   └── secrets.toml.example   # Template for secrets file
│
├── app/
│   ├── llm_model.py           # GPT model interface
│   ├── main.py                # Application entry point (deprecated)
│   ├── process_data.py        # PDF loading and embeddings
│   ├── ui_streamlit.py        # Streamlit web interface
│   ├── vector_database.py     # Qdrant vector database client
│   └── data/                  # Legal PDF documents (user-provided)
│
└── prototype_basic_logic_n_ui/
    └── main.py                # Early prototype code
```