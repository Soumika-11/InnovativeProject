"""
Face Verification System - Main Application for Jetson Nano
Real-time face verification using USB webcam in Docker container
"""

import os
import sys
import cv2
import numpy as np
from tensorflow import keras
import time
from datetime import datetime

# Import utility functions
from utils import (
    detect_and_crop_face,
    identify_face_from_frame,
    build_gallery_embeddings,
    index_images_by_person,
    draw_face_bbox
)


# Configuration from environment variables
BASE_DIR = os.getenv('APP_HOME', '/app')
CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', '0'))
VERIFICATION_THRESHOLD = float(os.getenv('VERIFICATION_THRESHOLD', '0.6'))
IMG_SIZE = int(os.getenv('IMG_SIZE', '128'))
EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '64'))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, 'face_embedding_model_CLEAN.h5')
REFERENCE_DIR = os.path.join(BASE_DIR, 'data_extracted', 'ref', 'short_references_final')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def load_model(model_path):
    """Load the pre-trained face embedding model"""
    print(f"\n🔄 Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"✅ Model loaded successfully!")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shape: {model.output_shape}")
        print(f"   - Total parameters: {model.count_params():,}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def initialize_camera(camera_index=0, max_retries=5):
    """Initialize USB webcam with retry logic"""
    print(f"\n📷 Initializing camera (index: {camera_index})...")
    
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            # Set camera properties for better performance on Jetson Nano
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ Camera initialized successfully!")
            print(f"   - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"   - FPS: {int(cap.get(cv2.CAP_PROP_FPS))}")
            return cap
        
        print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed. Retrying...")
        time.sleep(1)
    
    print(f"❌ Failed to initialize camera after {max_retries} attempts")
    print(f"💡 Tips:")
    print(f"   - Check if camera is connected via USB")
    print(f"   - Try different camera index (CAMERA_INDEX env var)")
    print(f"   - Verify camera permissions in Docker")
    print(f"   - Run: ls -l /dev/video*")
    sys.exit(1)


def run_verification_loop(cap, model, gallery_embeddings):
    """Main verification loop with webcam"""
    print("\n" + "="*60)
    print("🎥 REAL-TIME FACE VERIFICATION SYSTEM")
    print("="*60)
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save snapshot")
    print("  - Press 'r' to reset/reload gallery")
    print("="*60 + "\n")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Make a copy for processing
        display_frame = frame.copy()
        
        # Detect face
        cropped_face, bbox = detect_and_crop_face(frame)
        
        if cropped_face is not None and bbox is not None:
            # Identify face
            person_id, distance, status = identify_face_from_frame(
                cropped_face, model, gallery_embeddings, 
                VERIFICATION_THRESHOLD, IMG_SIZE
            )
            
            # Prepare label
            label = f"{person_id} ({distance:.3f})"
            
            # Color based on status
            color = (0, 255, 0) if status == "MATCH" else (0, 0, 255)
            
            # Draw bounding box and label
            display_frame = draw_face_bbox(display_frame, bbox, label, color)
            
            # Print verification result
            if status == "MATCH":
                print(f"✅ {person_id} verified (distance: {distance:.3f})")
            else:
                print(f"❌ Unknown person (distance: {distance:.3f})")
        
        # Draw FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw threshold info
        cv2.putText(display_frame, f"Threshold: {VERIFICATION_THRESHOLD:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Face Verification - Jetson Nano', display_frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('s'):
            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(OUTPUT_DIR, f"snapshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, display_frame)
            print(f"📸 Snapshot saved: {snapshot_path}")
        elif key == ord('r'):
            print("\n🔄 Reloading gallery...")
            gallery_embeddings = load_gallery(model)
            print("✅ Gallery reloaded!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def load_gallery(model):
    """Load and build gallery embeddings"""
    print(f"\n📂 Loading reference images from: {REFERENCE_DIR}")
    
    if not os.path.exists(REFERENCE_DIR):
        print(f"❌ Reference directory not found: {REFERENCE_DIR}")
        sys.exit(1)
    
    # Index reference images
    reference_images = index_images_by_person(REFERENCE_DIR, pattern='__')
    
    if not reference_images:
        print(f"⚠️ No reference images found!")
        print(f"💡 Make sure images are in: {REFERENCE_DIR}")
        sys.exit(1)
    
    print(f"✅ Found {len(reference_images)} persons in gallery")
    
    # Build gallery embeddings
    print(f"\n🔨 Building gallery embeddings...")
    gallery_embeddings = build_gallery_embeddings(model, reference_images, IMG_SIZE)
    
    print(f"✅ Gallery built with {len(gallery_embeddings)} persons:")
    for person_id in gallery_embeddings.keys():
        print(f"   - {person_id}")
    
    return gallery_embeddings


def main():
    """Main application entry point"""
    print("\n" + "="*60)
    print("🚀 FACE VERIFICATION SYSTEM - JETSON NANO")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - Camera Index: {CAMERA_INDEX}")
    print(f"  - Verification Threshold: {VERIFICATION_THRESHOLD}")
    print(f"  - Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Embedding Dim: {EMBEDDING_DIM}")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    model = load_model(MODEL_PATH)
    
    # Load gallery
    gallery_embeddings = load_gallery(model)
    
    # Initialize camera
    cap = initialize_camera(CAMERA_INDEX)
    
    try:
        # Run verification loop
        run_verification_loop(cap, model, gallery_embeddings)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Application terminated")


if __name__ == "__main__":
    main()
