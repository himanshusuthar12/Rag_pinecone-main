#!/bin/bash

# Deployment script for RAG Pinecone API
# This script helps deploy the application to various platforms

set -e

echo "=========================================="
echo "RAG Pinecone API Deployment Script"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file from env.example"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check Docker (optional)
if command_exists docker; then
    echo "✅ Docker found: $(docker --version)"
    USE_DOCKER=true
else
    echo "⚠️  Docker not found, will use local Python"
    USE_DOCKER=false
fi

# Deployment method selection
echo ""
echo "Select deployment method:"
echo "1) Local (Python virtual environment)"
echo "2) Docker"
echo "3) Docker Compose"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Setting up local environment..."
        
        # Create virtual environment if it doesn't exist
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        
        echo ""
        echo "✅ Setup complete!"
        echo ""
        echo "To run the application:"
        echo "  source venv/bin/activate"
        echo "  python rag_pinecone_fastapi.py"
        ;;
    
    2)
        if [ "$USE_DOCKER" = false ]; then
            echo "❌ Docker is required for this option"
            exit 1
        fi
        
        echo ""
        echo "Building Docker image..."
        docker build -t rag-pinecone-api .
        
        echo ""
        echo "✅ Build complete!"
        echo ""
        echo "To run the container:"
        echo "  docker run -d --name rag-api -p 8000:8000 --env-file .env rag-pinecone-api"
        ;;
    
    3)
        if [ "$USE_DOCKER" = false ]; then
            echo "❌ Docker is required for this option"
            exit 1
        fi
        
        if ! command_exists docker-compose; then
            echo "❌ Error: docker-compose is not installed"
            exit 1
        fi
        
        echo ""
        echo "Starting with Docker Compose..."
        docker-compose up -d
        
        echo ""
        echo "✅ Deployment complete!"
        echo "API should be available at http://localhost:8000"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
        ;;
    
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Deployment script completed!"
echo "=========================================="
