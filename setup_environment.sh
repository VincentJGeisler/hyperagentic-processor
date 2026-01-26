#!/bin/bash

# Setup script for Hyperagentic Processor development environment
# This creates a clean conda environment to avoid polluting system Python

echo "🧬 Setting up Hyperagentic Processor environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -eq 0 ]; then
    echo "✅ Environment created successfully!"
    echo ""
    echo "🚀 To activate the environment, run:"
    echo "   conda activate hyperagentic-processor"
    echo ""
    echo "🧪 To run tests:"
    echo "   conda activate hyperagentic-processor"
    echo "   python test_functional_agents.py"
    echo ""
    echo "🐳 To run with Docker (alternative):"
    echo "   docker-compose up --build"
else
    echo "❌ Failed to create environment"
    exit 1
fi