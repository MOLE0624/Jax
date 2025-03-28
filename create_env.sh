#!/bin/bash

# Check if python3 is installed
if ! command -v python3.12 &> /dev/null
then
    echo "python3.12 could not be found, installing..."
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Detect system architecture and handle jaxlib installation
ARCH=$(uname -m)

if [[ "$ARCH" == "arm64" ]]; then
    echo "ARM architecture detected. Installing ARM-compatible jaxlib..."
else
    echo "x86 architecture detected. Keeping existing jaxlib installation."
fi

python3 -m pip install -U pip wheel setuptools

# Install dependencies from requirements.txt using python3.12 -m pip
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    python3 -m pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping dependency installation."
fi
