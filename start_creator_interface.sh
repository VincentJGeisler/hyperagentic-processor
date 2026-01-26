#!/bin/bash
# Start the Creator Interface and Agent Universe in Docker

echo "🌟 Starting Hyperagentic Processor with Creator Interface..."
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env file..."
    echo "GROQ_API_KEY=your_api_key_here" > .env
    echo ""
    echo "❌ Please edit .env and add your Groq API key"
    echo "   Get your key at: https://console.groq.com"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo "✅ Environment configured"
echo ""

# Build and start containers
echo "🔨 Building containers..."
docker-compose build

echo ""
echo "🚀 Starting universe..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 5

# Check if services are running
if docker ps | grep -q "hyperagentic_universe"; then
    echo "✅ Agent Universe is running"
else
    echo "❌ Agent Universe failed to start"
    docker-compose logs agent_universe
    exit 1
fi

if docker ps | grep -q "creator_interface"; then
    echo "✅ Creator Interface is running"
else
    echo "❌ Creator Interface failed to start"
    docker-compose logs creator_interface
    exit 1
fi

echo ""
echo "🎉 Hyperagentic Processor is now running!"
echo ""
echo "📡 Access Points:"
echo "   Creator Interface (Web UI): http://localhost:3000"
echo "   Creator Interface (API):    http://localhost:8001"
echo "   Agent Universe (API):       http://localhost:8000"
echo "   Monitoring:                 http://localhost:9090"
echo ""
echo "🔍 View logs:"
echo "   docker-compose logs -f agent_universe"
echo "   docker-compose logs -f creator_interface"
echo ""
echo "🛑 Stop everything:"
echo "   docker-compose down"
echo ""
echo "⚡ The agents are now safely contained and awaiting divine messages..."
