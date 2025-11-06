"""
Utility functions for Face Verification System
Extracted from Jupyter notebook for Docker deployment
"""

import os
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf


def preprocess_image(img, img_size=128):
    """
    Preprocess image for model input
    
    Args:
        img: Input image (numpy array)
        img_size: Target image size (default: 128)
    
    Returns:
        Preprocessed image array
    """
    if img is None:
        return None
    
    # Resize image
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    return img_normalized


def load_image(img_path, img_size=128):
    """
    Load and preprocess an image from file path
    
    Args:
        img_path: Path to image file
        img_size: Target image size (default: 128)
    
    Returns:
        Preprocessed image array or None if error
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return preprocess_image(img, img_size)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None


def detect_and_crop_face(frame, face_cascade_path=None):
    """
    Detect and crop face from frame using Haar Cascade
    
    Args:
        frame: Input frame (BGR format)
        face_cascade_path: Path to Haar cascade XML (optional)
    
    Returns:
        cropped_face: Cropped face image (RGB) or None
        bbox: Bounding box (x, y, w, h) or None
    """
    if face_cascade_path is None:
        # Use OpenCV's built-in Haar cascade
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None, None
    
    # Get the largest face
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Crop face from frame
    face_img = frame[y:y+h, x:x+w]
    
    # Convert to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    return face_rgb, (x, y, w, h)


def euclidean_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Euclidean distance as float
    """
    return np.linalg.norm(embedding1 - embedding2)


def build_gallery_embeddings(model, image_paths_dict, img_size=128):
    """
    Build a gallery of face embeddings from reference images
    
    Args:
        model: Feature extractor model
        image_paths_dict: Dictionary {person_id: [image_paths]}
        img_size: Image size for preprocessing
    
    Returns:
        gallery_embeddings: Dictionary {person_id: embedding_array}
    """
    gallery_embeddings = {}
    
    for person_id, paths in image_paths_dict.items():
        embeddings = []
        
        for path in paths:
            img = load_image(path, img_size)
            if img is not None:
                # Expand dimensions for batch processing
                img_batch = np.expand_dims(img, axis=0)
                
                # Get embedding
                embedding = model.predict(img_batch, verbose=0)
                embeddings.append(embedding[0])
        
        if embeddings:
            # Average embeddings for this person
            avg_embedding = np.mean(embeddings, axis=0)
            gallery_embeddings[person_id] = avg_embedding
            print(f"✅ {person_id}: {len(embeddings)} embeddings averaged")
    
    return gallery_embeddings


def identify_face(query_embedding, gallery_embeddings, threshold=0.6):
    """
    Identify face by comparing query embedding with gallery
    
    Args:
        query_embedding: Query face embedding
        gallery_embeddings: Dictionary of reference embeddings
        threshold: Distance threshold for verification
    
    Returns:
        person_id: Identified person ID or "UNKNOWN"
        min_distance: Minimum distance found
        status: "MATCH" or "REJECT"
    """
    min_distance = float('inf')
    identified_person = "UNKNOWN"
    
    for person_id, ref_embedding in gallery_embeddings.items():
        distance = euclidean_distance(query_embedding, ref_embedding)
        
        if distance < min_distance:
            min_distance = distance
            identified_person = person_id
    
    # Determine status based on threshold
    status = "MATCH" if min_distance < threshold else "REJECT"
    
    if status == "REJECT":
        identified_person = "UNKNOWN"
    
    return identified_person, float(min_distance), status


def identify_face_from_frame(face_img, model, gallery_embeddings, threshold=0.6, img_size=128):
    """
    Identify face from a cropped face image
    
    Args:
        face_img: Cropped face image (RGB)
        model: Feature extractor model
        gallery_embeddings: Gallery embeddings dictionary
        threshold: Verification threshold
        img_size: Image size for preprocessing
    
    Returns:
        person_id: Identified person or "UNKNOWN"
        distance: Distance to closest match
        status: "MATCH" or "REJECT"
    """
    # Preprocess face
    face_processed = preprocess_image(face_img, img_size)
    if face_processed is None:
        return "UNKNOWN", float('inf'), "REJECT"
    
    # Get embedding
    face_batch = np.expand_dims(face_processed, axis=0)
    query_embedding = model.predict(face_batch, verbose=0)[0]
    
    # Identify
    person_id, distance, status = identify_face(query_embedding, gallery_embeddings, threshold)
    
    return person_id, distance, status


def index_images_by_person(image_dir, pattern='__'):
    """
    Index images by person ID from a directory
    
    Args:
        image_dir: Directory containing images
        pattern: Separator pattern in filename (default: '__')
    
    Returns:
        images_by_person: Dictionary {person_id: [image_paths]}
    """
    images_by_person = {}
    
    if not os.path.exists(image_dir):
        print(f"⚠️ Directory not found: {image_dir}")
        return images_by_person
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Extract person ID (before the pattern)
            if pattern in filename:
                person_id = filename.split(pattern)[0]
            else:
                person_id = os.path.splitext(filename)[0]
            
            full_path = os.path.join(image_dir, filename)
            
            if person_id not in images_by_person:
                images_by_person[person_id] = []
            images_by_person[person_id].append(full_path)
    
    return images_by_person


def draw_face_bbox(frame, bbox, label, color=(0, 255, 0), thickness=2):
    """
    Draw bounding box and label on frame
    
    Args:
        frame: Input frame (BGR)
        bbox: Bounding box (x, y, w, h)
        label: Label text
        color: Box color in BGR (default: green)
        thickness: Line thickness
    
    Returns:
        frame: Frame with drawn bbox and label
    """
    x, y, w, h = bbox
    
    # Draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame
