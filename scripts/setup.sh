#!/bin/bash
echo "Setting up ASL Recognition Docker environment..."

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    echo ""
    echo "Please make sure XQuartz is installed and running:"
    echo "1. Install XQuartz: brew install --cask xquartz"
    echo "2. Start XQuartz"
    echo "3. In XQuartz Preferences > Security, enable 'Allow connections from network clients'"
    echo "4. Restart XQuartz"
    echo ""
    
    read -p "Press Enter when XQuartz is ready..."
    
    echo "Starting with macOS configuration..."
    docker-compose -f docker-compose-macos.yml up --build
    
else
    echo "This setup script is for macOS. For Windows, use setup.bat"
    echo "For other systems, you may need to manually configure Docker."
fi