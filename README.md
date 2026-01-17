# RAG Pinecone FastAPI Application

A Retrieval-Augmented Generation (RAG) API built with FastAPI, OpenAI, and Pinecone for multi-domain question answering.

## Features

- Multi-domain routing using LLM
- Vector search with Pinecone
- Configurable database backend (SQLite or SQL Server)
- Chat history tracking
- RESTful API with OpenAPI documentation
- Optional Streamlit UI

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key and index
- Database (SQLite or SQL Server)

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Rag_pinecone
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `env.example` to `.env` and fill in your values:

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
OPENAI_API_KEY=your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=your-index-name
DB_CONNECTION_STRING=sqlite:///./aiChatbot001.db
```

### 5. Set up database

Ensure your database has a table named `aiChatbot001` (or as specified in `CONFIG_TABLE_NAME`) with the following columns:

- `available_domains` (TEXT/JSON): List of available domains
- `DOMAIN_TO_NAMESPACE` (TEXT/JSON): Mapping of domains to Pinecone namespaces
- `column_names` (TEXT/JSON): Column names for metadata filtering
- `EMBED_MODEL` (TEXT): OpenAI embedding model (e.g., "text-embedding-3-small")
- `EMBED_SIZE` (INTEGER): Embedding dimension (e.g., 1536)
- `ROUTER_MODEL` (TEXT): OpenAI model for routing (e.g., "gpt-4o-mini")

Example SQLite setup:

```sql
CREATE TABLE aiChatbot001 (
    available_domains TEXT,
    DOMAIN_TO_NAMESPACE TEXT,
    column_names TEXT,
    EMBED_MODEL TEXT,
    EMBED_SIZE INTEGER,
    ROUTER_MODEL TEXT
);

INSERT INTO aiChatbot001 VALUES (
    '["domain1", "domain2"]',
    '{"domain1": "namespace1", "domain2": "namespace2"}',
    '{"domain1": ["col1", "col2"]}',
    'text-embedding-3-small',
    1536,
    'gpt-4o-mini'
);
```

## Running the Application

### Local Development

#### Option 1: Run FastAPI only

```bash
python rag_pinecone_fastapi.py
```

The API will be available at `http://localhost:8000`

#### Option 2: Run with Streamlit UI

```bash
# Windows
run.bat

# Linux/Mac (create similar script)
python rag_pinecone_fastapi.py &
streamlit run streamlit_ui.py
```

### Docker Deployment

#### Build and run with Docker

```bash
# Build the image
docker build -t rag-pinecone-api .

# Run the container
docker run -d \
  --name rag-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/aiChatbot001.db:/app/aiChatbot001.db \
  -v $(pwd)/data:/app/data \
  rag-pinecone-api
```

#### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## API Endpoints

### Health Check
```
GET /health
```

### Get Configuration
```
GET /config
```

### Query
```
POST /query
Content-Type: application/json

{
  "query": "Your question here",
  "user_id": "user123",
  "top_k": 3
}
```

### Get History
```
GET /history/{user_id}
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `PINECONE_API_KEY` | Pinecone API key | Yes | - |
| `PINECONE_INDEX_NAME` | Pinecone index name | Yes | - |
| `PINECONE_CLOUD` | Pinecone cloud provider | No | `aws` |
| `PINECONE_REGION` | Pinecone region | No | `us-east-1` |
| `DB_TYPE` | Database type (`sqlite` or `sqlserver`) | No | `sqlserver` |
| `DB_CONNECTION_STRING` | Database connection string | Yes | - |
| `CONFIG_TABLE_NAME` | Configuration table name | No | `aiChatbot001` |
| `HOST` | Server host | No | `0.0.0.0` |
| `PORT` | Server port | No | `8000` |
| `HISTORY_FILE` | Chat history file path | No | `chat_history.json` |

## Deployment

### Cloud Platforms

#### Heroku

1. Create `Procfile`:
```
web: uvicorn rag_pinecone_fastapi:app --host 0.0.0.0 --port $PORT
```

2. Deploy:
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your-key
heroku config:set PINECONE_API_KEY=your-key
# ... set other env vars
git push heroku main
```

#### AWS Elastic Beanstalk / EC2

1. Use Docker deployment or install dependencies directly
2. Set environment variables
3. Use a process manager like `systemd` or `supervisord`

#### Google Cloud Run

```bash
gcloud run deploy rag-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name rag-api \
  --image rag-pinecone-api \
  --environment-variables @.env
```

## Troubleshooting

### Common Issues

1. **Database connection fails**
   - Verify `DB_CONNECTION_STRING` is correct
   - Ensure database file exists (SQLite) or server is accessible (SQL Server)
   - Check table exists with correct schema

2. **Configuration not loading**
   - Verify table name matches `CONFIG_TABLE_NAME`
   - Check JSON format in database columns
   - Review application logs for parsing errors

3. **Pinecone index not found**
   - Index will be created automatically if it doesn't exist
   - Verify `PINECONE_API_KEY` and permissions
   - Check `PINECONE_INDEX_NAME` is correct

4. **CORS errors**
   - Update `allow_origins` in `rag_pinecone_fastapi.py` for production
   - Ensure frontend URL is included in allowed origins

## License

[Your License Here]

## Support

For issues and questions, please open an issue in the repository.
