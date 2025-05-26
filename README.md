# ImageNet Classification Model - Cerebrium Deployment

This project deploys a PyTorch ImageNet classification model to Cerebrium serverless GPU platform using ONNX format and Docker containers.

## 🎯 Project Overview

- **Model** : ImageNet classification with 1000 classes
- **Input** : 224x224 RGB images
- **Output** : Class predictions with confidence scores
- **Performance** : <3 seconds response time
- **Platform** : Cerebrium serverless GPU deployment

## 📋 Requirements

- Python 3.9+
- Docker
- Cerebrium account (30 USD free credits)
- Git

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd imagenet-classifier-cerebrium
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

The model weights will be automatically downloaded, or manually:

```bash
wget -O pytorch_model_weights.pth "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
```

### 4. Convert Model to ONNX

```bash
python convert_to_onnx.py
```

### 5. Test Locally

```bash
python test.py
```

### 6. Deploy to Cerebrium

```bash
# Install Cerebrium CLI
pip install cerebrium

# Login to Cerebrium
cerebrium login

# Deploy using Docker
cerebrium deploy
```

## 🛠️ Detailed Usage

### Local Testing

Run comprehensive tests:

```bash
python test.py
```

Test categories:

- **Image Preprocessing** : RGB conversion, resizing, normalization
- **Model Loading** : ONNX model initialization
- **Predictions** : Single and batch inference
- **Performance** : Response time validation
- **Real Images** : Test with provided sample images

### Testing Deployed Model

After deployment, test the live endpoint:

```bash
# Basic health check
python test_server.py <CEREBRIUM_URL>

# Test with specific image
python test_server.py <CEREBRIUM_URL> --image path/to/image.jpg

# Comprehensive testing
python test_server.py <CEREBRIUM_URL> --comprehensive

# Performance testing
python test_server.py <CEREBRIUM_URL> --image path/to/image.jpg --performance
```

### Docker Deployment

Build and test locally:

```bash
# Build Docker image
docker build -t imagenet-classifier .

# Run locally
docker run -p 8000:8000 imagenet-classifier

# Test health endpoint
curl http://localhost:8000/health
```

### Single Prediction

```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image_file>
```

Response:

```json
{
  "class_id": 285,
  "class_name": "class_285",
  "confidence": 0.8234,
  "inference_time": 0.123,
  "filename": "test_image.jpg"
}
```

## 🧪 Testing Strategy

### Local Tests (`test.py`)

- Image preprocessing validation
- Model loading and initialization
- Inference accuracy and speed
- Error handling and edge cases

### Server Tests (`test_server.py`)

- API endpoint functionality
- Performance benchmarking
- Health monitoring
- Integration testing

### CI/CD Pipeline

- Automated testing on every commit
- Docker image build validation
- Security scanning with Trivy
- Code quality checks (linting, formatting)

## 🔍 Image Preprocessing

The model requires specific preprocessing:

1. **Convert to RGB** (if not already)
2. **Resize to 224x224** (bilinear interpolation)
3. **Normalize to [0,1]** (divide by 255)
4. **Apply ImageNet normalization** :

- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

1. **Convert HWC → CHW** format
2. **Add batch dimension**

### Testing

- Write unit tests for all functions
- Include integration tests
- Test edge cases and error conditions
- Validate performance requirements

## 🔒 Security

- No hardcoded credentials
- Input validation for all endpoints
- Error handling without information disclosure
- Regular dependency updates
- Container security scanning

---

**Note** : This is an assessment project for MTailor. The model and deployment are for demonstration purposes.
