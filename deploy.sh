#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting deployment process for pions...${NC}"

# Check if virtual environment exists, create if it doesn't
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install development dependencies
echo -e "${YELLOW}Installing development dependencies...${NC}"
pip install build twine wheel pytest

# Install project dependencies
echo -e "${YELLOW}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Install the package in development mode
echo -e "${YELLOW}Installing package in development mode...${NC}"
pip install -e .

# Run tests if they exist
if [ -d "tests" ]; then
    echo -e "${YELLOW}Running tests...${NC}"
    pytest
fi

# Build the package
echo -e "${YELLOW}Building the package...${NC}"
python -m build

# Option to deploy to PyPI
if [ "$1" == "--pypi" ]; then
    echo -e "${YELLOW}Uploading to PyPI...${NC}"
    if [ "$2" == "--prod" ]; then
        python -m twine upload dist/*
    else
        python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    fi
fi

# Option to create a new git tag and push
if [ "$1" == "--tag" ]; then
    VERSION=$(grep "version" setup.py | grep -o "[0-9]\+\.[0-9]\+\.[0-9]\+")
    echo -e "${YELLOW}Creating git tag v$VERSION...${NC}"
    git tag -a "v$VERSION" -m "Version $VERSION"
    git push origin "v$VERSION"
fi

echo -e "${GREEN}Deployment process completed successfully!${NC}"

# Deactivate virtual environment
deactivate
