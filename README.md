# Interactive Legal Advisory System

An AI-powered system for interactive legal consultations using natural language processing and vector search capabilities.

**PL:** Interaktywny system doradczy do konsultacji prawnych oparty na sztucznej inteligencji.

## Features

- Interactive chat interface powered by GPT models
- Vector-based legal document search using Qdrant
- Streamlit web interface for easy interaction
- RAG (Retrieval-Augmented Generation) for accurate legal advice

## Requirements

- Python 3.8+
- OpenAI API key
- Qdrant vector database

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd EngineeringThesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment:
   - Copy the example secrets file:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   - Edit `.streamlit/secrets.toml` and add your OpenAI API key

4. Start Qdrant (if running locally):
```bash
IN PROGRESS
```

## Usage

Run the Streamlit application:
```bash
IN PROGRESS
```

## Project Structure

```
│   .gitignore
│   README.md
│   requirements.txt
├───.streamlit
│   └───secrets.toml.example
│
├───app
│   │   llm_model.py
│   │   main.py
│   │   process_data.py
│   │   ui_streamlit.py
│   │   vector_database.py
│   │
│   └───data     
│
└───prototype_basic_logic_n_ui
    └───main.py
```