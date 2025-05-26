#!/bin/bash

# Deployment script for ImageNet Classification Model to Cerebrium
# This script automates the entire deployment process

set -e # Exit on any error

echo "🚀 Starting ImageNet Classification Model Deployment to Cerebrium"
echo "=================================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."

    # Detect OS
    OS="$(uname -s)"
    case "${OS}" in
    Linux*) MACHINE=Linux ;;
    Darwin*) MACHINE=Mac ;;
    CYGWIN*) MACHINE=Cygwin ;;
    MINGW*) MACHINE=MinGw ;;
    *) MACHINE="UNKNOWN:${OS}" ;;
    esac

    print_status "Detected OS: $MACHINE"

    # Check Python (try python3 first, then python)
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
        PIP_CMD="pip"
    else
        print_error "Python is not installed"
        if [ "$MACHINE" = "Mac" ]; then
            print_status "Install Python with: brew install python3"
        fi
        exit 1
    fi

    # Check pip
    if ! command -v $PIP_CMD &>/dev/null; then
        print_error "$PIP_CMD is not installed"
        exit 1
    fi

    # Check Docker
    if ! command -v docker &>/dev/null; then
        print_error "Docker is not installed"
        if [ "$MACHINE" = "Mac" ]; then
            print_status "Install Docker Desktop from: https://docs.docker.com/desktop/mac/install/"
        fi
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &>/dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi

    # Check git
    if ! command -v git &>/dev/null; then
        print_error "Git is not installed"
        if [ "$MACHINE" = "Mac" ]; then
            print_status "Install git with: brew install git or install Xcode Command Line Tools"
        fi
        exit 1
    fi

    # Check curl (should be available on macOS by default)
    if ! command -v curl &>/dev/null; then
        print_error "curl is not installed"
        exit 1
    fi

    print_success "All requirements satisfied"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."

    if [ -f "requirements.txt" ]; then
        $PIP_CMD install -r requirements.txt

        if [ $? -eq 0 ]; then
            print_success "Dependencies installed"
        else
            print_error "Failed to install dependencies"
            print_status "Try: $PIP_CMD install --upgrade pip"
            exit 1
        fi
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Download model weights if not present
download_weights() {
    print_status "Checking model weights..."

    if [ ! -f "pytorch_model_weights.pth" ]; then
        print_status "Downloading model weights..."

        # Use curl (available on macOS by default) instead of wget
        curl -L -o pytorch_model_weights.pth "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"

        if [ $? -eq 0 ] && [ -f "pytorch_model_weights.pth" ]; then
            # Verify file size (should be > 1MB)
            file_size=$(stat -f%z "pytorch_model_weights.pth" 2>/dev/null || stat -c%s "pytorch_model_weights.pth" 2>/dev/null)
            if [ "$file_size" -gt 1000000 ]; then
                print_success "Model weights downloaded successfully"
            else
                print_error "Downloaded file seems too small. Please check the download."
                rm -f pytorch_model_weights.pth
                exit 1
            fi
        else
            print_error "Failed to download model weights"
            print_status "You can manually download from: https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=0"
            exit 1
        fi
    else
        print_success "Model weights already present"
    fi
}

# Convert PyTorch model to ONNX
convert_model() {
    print_status "Converting PyTorch model to ONNX..."

    if [ ! -f "model.onnx" ]; then
        $PYTHON_CMD convert_to_onnx.py

        if [ $? -eq 0 ] && [ -f "model.onnx" ]; then
            print_success "Model converted to ONNX successfully"
        else
            print_error "Failed to convert model to ONNX"
            print_status "Check if pytorch_model_weights.pth is valid"
            exit 1
        fi
    else
        print_success "ONNX model already exists"
    fi
}

# Run local tests
run_tests() {
    print_status "Running local tests..."

    $PYTHON_CMD test.py

    if [ $? -eq 0 ]; then
        print_success "All tests passed"
    else
        print_error "Tests failed"
        print_status "Check the test output above for details"
        exit 1
    fi
}

# Build Docker image
build_docker() {
    print_status "Building Docker image..."

    docker build -t imagenet-classifier:latest .

    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Test Docker image locally
test_docker() {
    print_status "Testing Docker image locally..."

    # Start container in background
    docker run -d -p 8000:8000 --name test-container imagenet-classifier:latest

    # Wait for container to start
    print_status "Waiting for container to start..."
    sleep 30

    # Test health endpoint
    health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)

    if [ "$health_response" = "200" ]; then
        print_success "Docker container health check passed"
    else
        print_error "Docker container health check failed (HTTP $health_response)"
        docker logs test-container
        docker stop test-container
        docker rm test-container
        exit 1
    fi

    # Test prediction endpoint with a simple test
    print_status "Testing prediction endpoint..."

    # Create a test image using Python
    $PYTHON_CMD -c "
from PIL import Image
img = Image.new('RGB', (224, 224), color='red')
img.save('test_image.jpg')
"

    # Test prediction
    prediction_response=$(curl -s -X POST -F "file=@test_image.jpg" http://localhost:8000/predict)

    if echo "$prediction_response" | grep -q "class_id"; then
        print_success "Prediction endpoint working"
        echo "Sample prediction: $prediction_response"
    else
        print_error "Prediction endpoint failed"
        echo "Response: $prediction_response"
        docker logs test-container
        docker stop test-container
        docker rm test-container
        exit 1
    fi

    # Clean up
    rm -f test_image.jpg
    docker stop test-container
    docker rm test-container

    print_success "Docker image tests completed"
}

# Install Cerebrium CLI
install_cerebrium() {
    print_status "Checking Cerebrium CLI..."

    if ! command -v cerebrium &>/dev/null; then
        print_status "Installing Cerebrium CLI..."
        $PIP_CMD install cerebrium

        if [ $? -eq 0 ]; then
            print_success "Cerebrium CLI installed"
        else
            print_error "Failed to install Cerebrium CLI"
            exit 1
        fi
    else
        print_success "Cerebrium CLI already installed"
    fi
}

# Login to Cerebrium
cerebrium_login() {
    print_status "Checking Cerebrium authentication..."

    # Check if already logged in
    if cerebrium whoami &>/dev/null; then
        print_success "Already logged in to Cerebrium"
    else
        print_warning "Please log in to Cerebrium:"
        cerebrium login

        if [ $? -eq 0 ]; then
            print_success "Logged in to Cerebrium"
        else
            print_error "Failed to log in to Cerebrium"
            exit 1
        fi
    fi
}

# Deploy to Cerebrium
deploy_to_cerebrium() {
    print_status "Deploying to Cerebrium..."

    # Check if cerebrium.toml exists
    if [ ! -f "cerebrium.toml" ]; then
        print_error "cerebrium.toml not found"
        exit 1
    fi

    # Deploy
    cerebrium deploy

    if [ $? -eq 0 ]; then
        print_success "Deployment to Cerebrium completed!"

        # Get deployment info
        print_status "Getting deployment information..."
        cerebrium get-url

        print_success "🎉 Deployment successful!"
        print_status "You can now test your deployment using:"
        print_status "python3 test_server.py <YOUR_CEREBRIUM_URL> --comprehensive"

    else
        print_error "Failed to deploy to Cerebrium"
        exit 1
    fi
}

# Test deployed model
test_deployment() {
    print_status "Testing deployed model..."
    print_warning "Make sure to replace <YOUR_CEREBRIUM_URL> with actual deployment URL"

    read -p "Enter your Cerebrium deployment URL: " DEPLOYMENT_URL

    if [ -n "$DEPLOYMENT_URL" ]; then
        print_status "Running comprehensive tests on deployed model..."
        $PYTHON_CMD test_server.py "$DEPLOYMENT_URL" --comprehensive

        if [ $? -eq 0 ]; then
            print_success "Deployment tests passed!"
        else
            print_warning "Some deployment tests failed. Check the output above."
        fi
    else
        print_warning "No URL provided. Skipping deployment tests."
        print_status "You can test later with: $PYTHON_CMD test_server.py <URL> --comprehensive"
    fi
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    echo

    check_requirements
    install_dependencies
    download_weights
    convert_model
    run_tests
    build_docker
    test_docker
    install_cerebrium
    cerebrium_login
    deploy_to_cerebrium

    echo
    print_success "🎉 Deployment process completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Test your deployment with: python3 test_server.py <CEREBRIUM_URL> --comprehensive"
    echo "2. Monitor your deployment on the Cerebrium dashboard"
    echo "3. Create your Loom video demonstration"
    echo
    echo "Deployment summary:"
    echo "- Model: ImageNet classification (1000 classes)"
    echo "- Format: ONNX optimized"
    echo "- Platform: Cerebrium serverless GPU"
    echo "- Performance: <3 seconds response time"
    echo
}

# Handle script arguments
case "${1:-}" in
"--help" | "-h")
    echo "ImageNet Classification Model Deployment Script"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  --help, -h        Show this help message"
    echo "  --test-only       Run only local tests"
    echo "  --docker-only     Build and test Docker image only"
    echo "  --deploy-only     Deploy to Cerebrium only (assumes everything is ready)"
    echo
    echo "Without options, runs the complete deployment process."
    ;;
"--test-only")
    check_requirements
    install_dependencies
    download_weights
    convert_model
    run_tests
    ;;
"--docker-only")
    check_requirements
    build_docker
    test_docker
    ;;
"--deploy-only")
    install_cerebrium
    cerebrium_login
    deploy_to_cerebrium
    ;;
"")
    main
    ;;
*)
    print_error "Unknown option: $1"
    echo "Use --help for usage information"
    exit 1
    ;;
esac
