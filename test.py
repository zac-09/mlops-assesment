#!/usr/bin/env python3
"""
Comprehensive test suite for the ONNX ImageNet classification model.
Tests model loading, preprocessing, inference, and performance.
"""

import unittest
import numpy as np
import os
import time
from PIL import Image
from model import ONNXImageClassifier, ImagePreprocessor, create_classifier
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestImagePreprocessor(unittest.TestCase):
    """Test cases for ImagePreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        self.temp_dir = tempfile.mkdtemp()
        
    def create_test_image(self, size=(256, 256), mode='RGB'):
        """Create a test image for testing."""
        image = Image.new(mode, size, color='red')
        image_path = os.path.join(self.temp_dir, f'test_{mode}_{size[0]}x{size[1]}.jpg')
        image.save(image_path)
        return image_path
    
    def test_preprocess_image_rgb(self):
        """Test preprocessing of RGB image."""
        image_path = self.create_test_image()
        result = self.preprocessor.preprocess_image(image_path)
        
        # Check output shape
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        
    def test_preprocess_image_grayscale(self):
        """Test preprocessing of grayscale image (should convert to RGB)."""
        image_path = self.create_test_image(mode='L')
        result = self.preprocessor.preprocess_image(image_path)
        
        # Should still output 3 channels after RGB conversion
        self.assertEqual(result.shape, (1, 3, 224, 224))
        
    def test_preprocess_different_sizes(self):
        """Test preprocessing of images with different sizes."""
        sizes = [(100, 100), (512, 512), (224, 224)]
        
        for size in sizes:
            with self.subTest(size=size):
                image_path = self.create_test_image(size=size)
                result = self.preprocessor.preprocess_image(image_path)
                
                # All should be resized to 224x224
                self.assertEqual(result.shape, (1, 3, 224, 224))
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing of numpy array."""
        # Create random image array
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess_numpy(image_array)
        
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
    
    def test_normalization_values(self):
        """Test that normalization is applied correctly."""
        # Create white image (should be normalized)
        image_array = np.ones((224, 224, 3), dtype=np.float32)
        result = self.preprocessor.preprocess_numpy(image_array)
        
        # Check that values are normalized (not 0 or 1)
        self.assertNotEqual(result.min(), 0.0)
        self.assertNotEqual(result.max(), 1.0)

class TestONNXImageClassifier(unittest.TestCase):
    """Test cases for ONNXImageClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "model.onnx"
        self.temp_dir = tempfile.mkdtemp()
        
        # Skip tests if model doesn't exist
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model not found. Run convert_to_onnx.py first.")
    
    def create_test_image(self, size=(224, 224)):
        """Create a test image for testing."""
        image = Image.new('RGB', size, color='blue')
        image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        image.save(image_path)
        return image_path
    
    def test_model_loading(self):
        """Test model loading and initialization."""
        classifier = ONNXImageClassifier(self.model_path)
        
        self.assertIsNotNone(classifier.session)
        self.assertIsNotNone(classifier.input_name)
        self.assertIsNotNone(classifier.output_name)
        self.assertEqual(len(classifier.class_names), 1000)
    
    def test_single_prediction(self):
        """Test single image prediction."""
        classifier = ONNXImageClassifier(self.model_path)
        image_path = self.create_test_image()
        
        class_id, confidence, class_name = classifier.predict(image_path)
        
        self.assertIsInstance(class_id, int)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(class_name, str)
        self.assertGreaterEqual(class_id, 0)
        self.assertLess(class_id, 1000)
        self.assertGreaterEqual(confidence, 0.0)
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        classifier = ONNXImageClassifier(self.model_path)
        
        # Create multiple test images
        image_paths = [self.create_test_image() for _ in range(3)]
        results = classifier.predict_batch(image_paths)
        
        self.assertEqual(len(results), 3)
        for class_id, confidence, class_name in results:
            self.assertIsInstance(class_id, int)
            self.assertIsInstance(confidence, float)
            self.assertIsInstance(class_name, str)
    
    def test_top_k_predictions(self):
        """Test top-k predictions."""
        classifier = ONNXImageClassifier(self.model_path)
        image_path = self.create_test_image()
        
        k = 5
        results = classifier.get_top_k_predictions(image_path, k=k)
        
        self.assertEqual(len(results), k)
        
        # Check that results are sorted by confidence (descending)
        confidences = [confidence for _, confidence, _ in results]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
    
    def test_factory_function(self):
        """Test factory function."""
        classifier = create_classifier(self.model_path)
        self.assertIsInstance(classifier, ONNXImageClassifier)

class TestPerformance(unittest.TestCase):
    """Test performance requirements."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "model.onnx"
        self.temp_dir = tempfile.mkdtemp()
        
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model not found. Run convert_to_onnx.py first.")
    
    def create_test_image(self):
        """Create a test image."""
        image = Image.new('RGB', (224, 224), color='green')
        image_path = os.path.join(self.temp_dir, 'perf_test.jpg')
        image.save(image_path)
        return image_path
    
    def test_inference_speed(self):
        """Test that inference meets speed requirements (2-3 seconds)."""
        classifier = ONNXImageClassifier(self.model_path)
        image_path = self.create_test_image()
        
        # Warm up
        classifier.predict(image_path)
        
        # Measure inference time
        start_time = time.time()
        classifier.predict(image_path)
        end_time = time.time()
        
        inference_time = end_time - start_time
        logger.info(f"Inference time: {inference_time:.3f} seconds")
        
        # Should be under 3 seconds (preferably under 1 second for ONNX)
        self.assertLess(inference_time, 3.0, "Inference time exceeds 3 seconds")
    
    def test_memory_usage(self):
        """Test basic memory usage (load model without errors)."""
        try:
            classifier = ONNXImageClassifier(self.model_path)
            image_path = self.create_test_image()
            
            # Should not raise memory errors
            for _ in range(10):
                classifier.predict(image_path)
                
        except MemoryError:
            self.fail("Memory error during repeated inference")

class TestRealImages(unittest.TestCase):
    """Test with provided real images."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_path = "model.onnx"
        
        if not os.path.exists(self.model_path):
            self.skipTest("ONNX model not found. Run convert_to_onnx.py first.")
    
    def test_tench_image(self):
        """Test with tench image (should predict class 0)."""
        image_path = "n01440764_tench"
        
        if not os.path.exists(image_path):
            self.skipTest(f"Test image not found: {image_path}")
        
        classifier = ONNXImageClassifier(self.model_path)
        class_id, confidence, class_name = classifier.predict(image_path)
        
        logger.info(f"Tench prediction: class_id={class_id}, confidence={confidence:.3f}")
        
        # Should predict class 0 for tench
        self.assertEqual(class_id, 0, f"Expected class 0 for tench, got {class_id}")
    
    def test_turtle_image(self):
        """Test with turtle image (should predict class 35)."""
        image_path = "n01667114_mud_turtle"
        
        if not os.path.exists(image_path):
            self.skipTest(f"Test image not found: {image_path}")
        
        classifier = ONNXImageClassifier(self.model_path)
        class_id, confidence, class_name = classifier.predict(image_path)
        
        logger.info(f"Turtle prediction: class_id={class_id}, confidence={confidence:.3f}")
        
        # Should predict class 35 for mud turtle
        self.assertEqual(class_id, 35, f"Expected class 35 for turtle, got {class_id}")

def run_comprehensive_tests():
    """Run all tests and generate report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestImagePreprocessor,
        TestONNXImageClassifier,
        TestPerformance,
        TestRealImages
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError: ' in traceback else 'Unknown failure'
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2] if len(traceback.split('\n')) > 1 else 'Unknown error'
            print(f"- {test}: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)