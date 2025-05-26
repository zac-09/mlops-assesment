#!/usr/bin/env python3
"""
Test script specifically for Cerebrium deployed model.
Uses the correct Cerebrium API format.
"""

import requests
import json
import os
import time
import argparse
import logging
import base64
from PIL import Image
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CerebriumTester:
    """Test client for Cerebrium deployed model using correct API format."""
    
    def __init__(self, base_url: str, api_key: str = None):
        """Initialize tester with base URL and API key."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = self._get_api_key()
        
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            })
            logger.info("API key configured for authentication")
        else:
            logger.warning("No API key found")
    
    def _get_api_key(self) -> str:
        """Get API key from environment or Cerebrium CLI."""
        # Try environment variable first
        api_key = os.environ.get('CEREBRIUM_API_KEY')
        if api_key:
            return api_key
        
        # Try Cerebrium CLI
        try:
            import subprocess
            result = subprocess.run(['cerebrium', 'auth', 'print-token'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    
    def predict_image(self, image_path: str) -> dict:
        """
        Predict class for given image using Cerebrium API format.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            dict: Prediction results
        """
        try:
            logger.info(f"Predicting image: {image_path}")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Convert image to base64
            image_base64 = self.image_to_base64(image_path)
            
            # Prepare request data in Cerebrium format
            data = {
                "item": {
                    "image": image_base64
                }
            }
            
            # Make request to Cerebrium predict endpoint
            response = self.session.post(self.base_url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Prediction successful: {json.dumps(result, indent=2)}")
                return result
            else:
                logger.error(f"Prediction failed: {response.status_code} - {response.text}")
                return {"error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}
    
    def test_performance(self, image_path: str, num_requests: int = 5) -> dict:
        """Test performance with multiple requests."""
        try:
            logger.info(f"Testing performance with {num_requests} requests...")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Pre-encode image to base64
            image_base64 = self.image_to_base64(image_path)
            data = {"item": {"image": image_base64}}
            
            times = []
            successful_requests = 0
            
            for i in range(num_requests):
                start_time = time.time()
                
                response = self.session.post(self.base_url, json=data)
                
                end_time = time.time()
                request_time = end_time - start_time
                
                if response.status_code == 200:
                    times.append(request_time)
                    successful_requests += 1
                    logger.info(f"Request {i+1}/{num_requests}: {request_time:.3f}s")
                else:
                    logger.error(f"Request {i+1} failed: {response.status_code}")
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                metrics = {
                    "total_requests": num_requests,
                    "successful_requests": successful_requests,
                    "success_rate": successful_requests / num_requests,
                    "avg_response_time": avg_time,
                    "min_response_time": min_time,
                    "max_response_time": max_time
                }
                
                logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
                
                # Check performance requirement
                if avg_time <= 3.0:
                    logger.info(f"✅ Performance requirement met: {avg_time:.3f}s ≤ 3s")
                else:
                    logger.warning(f"⚠️ Performance requirement not met: {avg_time:.3f}s > 3s")
                
                return metrics
            else:
                logger.error("No successful requests")
                return {"error": "No successful requests"}
                
        except Exception as e:
            logger.error(f"Performance test error: {str(e)}")
            return {"error": str(e)}
    
    def create_test_image(self) -> str:
        """Create a temporary test image."""
        image = Image.new('RGB', (224, 224), color='blue')
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image.save(temp_file.name, 'JPEG')
        return temp_file.name
    
    def run_comprehensive_tests(self, test_image_path: str = None) -> bool:
        """Run comprehensive test suite."""
        logger.info("Starting Cerebrium comprehensive test suite...")
        
        # Use provided image or create test image
        if test_image_path and os.path.exists(test_image_path):
            image_path = test_image_path
            cleanup_image = False
        else:
            logger.info("Creating temporary test image...")
            image_path = self.create_test_image()
            cleanup_image = True
        
        try:
            results = {}
            
            # Test prediction
            logger.info("Testing single prediction...")
            pred_result = self.predict_image(image_path)
            results["prediction"] = "error" not in pred_result
            
            if results["prediction"]:
                logger.info(f"✅ Prediction successful!")
                if "class_id" in pred_result:
                    logger.info(f"   Class ID: {pred_result['class_id']}")
                if "confidence" in pred_result:
                    logger.info(f"   Confidence: {pred_result['confidence']:.3f}")
            else:
                logger.error(f"❌ Prediction failed: {pred_result.get('error', 'Unknown error')}")
            
            # Test performance (if prediction works)
            if results["prediction"]:
                logger.info("Testing performance...")
                perf_result = self.test_performance(image_path, num_requests=3)
                results["performance"] = "error" not in perf_result
                
                if results["performance"]:
                    avg_time = perf_result.get("avg_response_time", 0)
                    logger.info(f"✅ Performance test completed")
                    logger.info(f"   Average response time: {avg_time:.3f}s")
                else:
                    logger.error(f"❌ Performance test failed: {perf_result.get('error', 'Unknown error')}")
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("CEREBRIUM TEST SUMMARY")
            logger.info("="*60)
            
            passed_tests = sum(results.values())
            total_tests = len(results)
            
            for test_name, passed in results.items():
                status = "PASS" if passed else "FAIL"
                logger.info(f"{test_name:20s}: {status}")
            
            logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
            logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
            
            return all(results.values())
            
        finally:
            # Clean up temporary image
            if cleanup_image and os.path.exists(image_path):
                os.unlink(image_path)

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Test Cerebrium deployed model")
    parser.add_argument("base_url", help="Cerebrium endpoint URL")
    parser.add_argument("--image", help="Path to test image")
    parser.add_argument("--api-key", help="Cerebrium API key")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive test suite")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = CerebriumTester(args.base_url, args.api_key)
    
    if args.comprehensive:
        # Run comprehensive tests
        success = tester.run_comprehensive_tests(args.image)
        exit(0 if success else 1)
    
    elif args.performance and args.image:
        # Run performance tests
        result = tester.test_performance(args.image)
        if "error" not in result:
            print(f"Average response time: {result.get('avg_response_time', 0):.3f}s")
        else:
            print(f"Performance test failed: {result['error']}")
    
    elif args.image:
        # Single prediction
        result = tester.predict_image(args.image)
        if "error" not in result:
            class_id = result.get("class_id", -1)
            print(f"Predicted class ID: {class_id}")
            if "confidence" in result:
                print(f"Confidence: {result['confidence']:.3f}")
        else:
            print(f"Prediction failed: {result['error']}")
    
    else:
        print("Please specify --image, --comprehensive, or --performance")
        print("Use --help for more options")

if __name__ == "__main__":
    main()