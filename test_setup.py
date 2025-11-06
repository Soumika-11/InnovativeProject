#!/usr/bin/env python3
"""
Test script to verify the face verification system setup
Run this before deploying to Jetson Nano
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
END = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*60}{END}")
    print(f"{BLUE}{text.center(60)}{END}")
    print(f"{BLUE}{'='*60}{END}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{END}")

def print_error(text):
    print(f"{RED}❌ {text}{END}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{END}")

def print_info(text):
    print(f"{BLUE}ℹ️  {text}{END}")

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print_success(f"{description}: Found ({size_mb:.2f} MB)")
        return True
    else:
        print_error(f"{description}: Not found")
        print_warning(f"   Expected at: {filepath}")
        return False

def check_directory(dirpath, description, min_files=1):
    """Check if a directory exists and has files"""
    if not os.path.exists(dirpath):
        print_error(f"{description}: Directory not found")
        print_warning(f"   Expected at: {dirpath}")
        return False
    
    files = [f for f in os.listdir(dirpath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(files) >= min_files:
        print_success(f"{description}: Found {len(files)} images")
        return True
    else:
        print_error(f"{description}: Only {len(files)} images found (need at least {min_files})")
        return False

def test_opencv():
    """Test OpenCV installation"""
    print_info("Testing OpenCV...")
    try:
        import cv2
        version = cv2.__version__
        print_success(f"OpenCV version: {version}")
        return True
    except ImportError:
        print_error("OpenCV not installed")
        return False

def test_tensorflow():
    """Test TensorFlow installation"""
    print_info("Testing TensorFlow...")
    try:
        import tensorflow as tf
        version = tf.__version__
        print_success(f"TensorFlow version: {version}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print_success(f"GPU(s) detected: {len(gpus)}")
        else:
            print_warning("No GPU detected (will use CPU)")
        return True
    except ImportError:
        print_error("TensorFlow not installed")
        return False

def test_model_loading(model_path):
    """Test if model can be loaded"""
    print_info("Testing model loading...")
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path, compile=False)
        print_success(f"Model loaded successfully")
        print_info(f"   Input shape: {model.input_shape}")
        print_info(f"   Output shape: {model.output_shape}")
        print_info(f"   Parameters: {model.count_params():,}")
        return True
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return False

def test_camera(camera_index=0):
    """Test if camera is accessible"""
    print_info(f"Testing camera (index {camera_index})...")
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print_error("Camera not accessible")
            return False
        
        ret, frame = cap.read()
        if not ret:
            print_error("Could not read frame from camera")
            cap.release()
            return False
        
        height, width = frame.shape[:2]
        print_success(f"Camera working: {width}x{height}")
        cap.release()
        return True
    except Exception as e:
        print_error(f"Camera test failed: {e}")
        return False

def test_haar_cascade():
    """Test Haar cascade face detection"""
    print_info("Testing Haar cascade...")
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if face_cascade.empty():
            print_error("Haar cascade failed to load")
            return False
        
        print_success("Haar cascade loaded successfully")
        return True
    except Exception as e:
        print_error(f"Haar cascade test failed: {e}")
        return False

def check_docker_files():
    """Check if all Docker-related files exist"""
    print_header("Checking Docker Files")
    
    files = {
        "Dockerfile": "Dockerfile",
        "docker-compose.yml": "Docker Compose configuration",
        "app.py": "Main application",
        "utils.py": "Utility functions",
        "requirements-jetson.txt": "Jetson requirements",
        "build_and_run_jetson.sh": "Build script",
        ".dockerignore": "Docker ignore file"
    }
    
    all_exist = True
    for filename, description in files.items():
        if not check_file(filename, description):
            all_exist = False
    
    return all_exist

def check_model_files():
    """Check if model files exist"""
    print_header("Checking Model Files")
    
    all_exist = True
    if not check_file("face_embedding_model_CLEAN.h5", "Model file"):
        all_exist = False
    if not check_file("model_architecture.json", "Model architecture"):
        print_warning("Model architecture JSON missing (optional)")
    
    return all_exist

def check_data_files():
    """Check if data files exist"""
    print_header("Checking Data Files")
    
    all_exist = True
    ref_dir = "data_extracted/ref/short_references_final"
    
    if not check_directory(ref_dir, "Reference images", min_files=2):
        all_exist = False
        print_warning("Need at least 2 reference images for gallery")
    
    return all_exist

def test_python_dependencies():
    """Test if Python dependencies are installed"""
    print_header("Checking Python Dependencies")
    
    dependencies = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "tensorflow": "tensorflow",
        "PIL": "Pillow",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn"
    }
    
    all_installed = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print_success(f"{package}")
        except ImportError:
            print_error(f"{package} not installed")
            all_installed = False
    
    return all_installed

def run_integration_test():
    """Run a full integration test"""
    print_header("Running Integration Test")
    
    try:
        # Import modules
        from tensorflow import keras
        from utils import (
            load_image,
            index_images_by_person,
            build_gallery_embeddings
        )
        
        # Load model
        print_info("Loading model...")
        model = keras.models.load_model("face_embedding_model_CLEAN.h5", compile=False)
        print_success("Model loaded")
        
        # Index images
        print_info("Indexing reference images...")
        ref_dir = "data_extracted/ref/short_references_final"
        reference_images = index_images_by_person(ref_dir)
        print_success(f"Found {len(reference_images)} persons")
        
        # Build gallery
        print_info("Building gallery embeddings...")
        gallery_embeddings = build_gallery_embeddings(model, reference_images, 128)
        print_success(f"Gallery built with {len(gallery_embeddings)} embeddings")
        
        print_success("Integration test passed!")
        return True
        
    except Exception as e:
        print_error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print_header("Face Verification System - Pre-Deployment Check")
    
    results = {
        "docker_files": False,
        "model_files": False,
        "data_files": False,
        "python_deps": False,
        "opencv": False,
        "tensorflow": False,
        "model_loading": False,
        "camera": False,
        "haar_cascade": False,
        "integration": False
    }
    
    # Run all checks
    results["docker_files"] = check_docker_files()
    results["model_files"] = check_model_files()
    results["data_files"] = check_data_files()
    
    print_header("Testing Python Dependencies")
    results["python_deps"] = test_python_dependencies()
    
    print_header("Testing System Components")
    results["opencv"] = test_opencv()
    results["tensorflow"] = test_tensorflow()
    
    if results["model_files"]:
        results["model_loading"] = test_model_loading("face_embedding_model_CLEAN.h5")
    
    results["camera"] = test_camera(0)
    results["haar_cascade"] = test_haar_cascade()
    
    # Integration test
    if results["model_files"] and results["data_files"] and results["python_deps"]:
        results["integration"] = run_integration_test()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = GREEN if result else RED
        print(f"{color}{status:6s}{END} - {test_name.replace('_', ' ').title()}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_success("All tests passed! ✨")
        print_info("System is ready for deployment to Jetson Nano")
        return 0
    else:
        print_error("Some tests failed")
        print_warning("Fix the issues before deploying to Jetson Nano")
        return 1

if __name__ == "__main__":
    sys.exit(main())
