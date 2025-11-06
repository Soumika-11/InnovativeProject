# Face Verification System - Complete Implementation

## 🎯 Overview

This is a complete face verification system using Siamese Neural Networks, converted from Google Colab to run locally on your machine with USB webcam support.

## ✨ Features

- **Siamese Neural Network** for face embeddings
- **Contrastive Loss** for training
- **USB Webcam Support** via OpenCV (cv2.VideoCapture)
- **Real-time Face Detection** using Haar Cascades
- **Face Verification** against a gallery of known faces
- **Optional Gender Detection** using DeepFace
- **Comprehensive Jupyter Notebook** with all code in one place
- **🆕 Docker Support** for NVIDIA Jetson Nano deployment

## 📋 Requirements

- Python 3.8 or higher
- USB Webcam (for real-time verification)
- Dataset with reference and distorted images

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Structure

Your dataset should be organized as follows:

```
InnovativeProject/
├── data_extracted/
│   ├── ref/
│   │   └── short_references_final/
│   │       ├── person1.jpg
│   │       ├── person2.jpg
│   │       └── ...
│   └── distorted/
│       └── Short_distortion_final/
│           ├── person1__distortion1.jpg
│           ├── person1__distortion2.jpg
│           └── ...
├── face_verification_complete.ipynb
└── requirements.txt
```

### 3. Run the Notebook

```bash
jupyter notebook face_verification_complete.ipynb
```

## 📓 Notebook Structure

The notebook is organized into 15 sections:

1. **Import Libraries** - Load all required dependencies
2. **Configuration** - Set paths and hyperparameters
3. **Helper Functions** - Image preprocessing utilities
4. **Model Architecture** - Siamese CNN definition
5. **Loss & Generator** - Training components
6. **USB Webcam Functions** - Local camera capture (replaces Colab)
7. **Verification Functions** - Face identification logic
8. **Gender Detection** - Optional DeepFace integration
9. **Data Preparation** - Index images by person ID
10. **Train/Load Model** - Model training or loading
11. **Build Gallery** - Create face embedding database
12. **Test Random Pairs** - Validation on dataset
13. **Webcam Verification** - Real-time verification demo
14. **Multiple Tests** - Batch testing and accuracy
15. **Summary** - System overview and tips

## 🔧 Configuration

Edit these paths in the notebook's configuration section:

```python
BASE_DIR = '/Users/soumikadas/Documents/SourceCodes/InnovativeProject'
REFERENCE_DIR = os.path.join(OUTPUT_DIR, 'ref', 'short_references_final')
DISTORTION_DIR = os.path.join(OUTPUT_DIR, 'distorted', 'Short_distortion_final')
```

### Hyperparameters

```python
IMG_SIZE = 128              # Image size for model input
EMBEDDING_DIM = 64          # Dimension of face embeddings
VERIFICATION_THRESHOLD = 0.6 # Distance threshold for verification
EPOCHS = 10                 # Training epochs
BATCH_SIZE = 32             # Training batch size
```

## 📷 Using USB Webcam

The system uses OpenCV's `cv2.VideoCapture` for webcam access:

```python
# Capture a frame from USB camera
frame = capture_and_display_webcam(camera_index=0)

# Detect face in frame
cropped_face, bbox = detect_and_crop_face(frame)

# Verify against gallery
person_id, distance, status = identify_face_from_frame(
    cropped_face, 
    final_extractor_clean, 
    gallery_embeddings, 
    VERIFICATION_THRESHOLD
)
```

**Controls:**
- Press **'c'** to capture a frame
- Press **'q'** to quit without capturing

## 🎨 Key Changes from Colab Version

### Removed:
- ❌ `from google.colab import drive`
- ❌ `from google.colab.output import eval_js`
- ❌ `from IPython.display import Javascript`
- ❌ Browser-based webcam capture with JavaScript
- ❌ File download commands (`files.download()`)
- ❌ Shell commands (`!wget`, `!pip`)
- ❌ Zip file extraction code

### Added:
- ✅ `cv2.VideoCapture()` for USB webcam
- ✅ Local file path configuration
- ✅ Proper directory structure handling
- ✅ Cross-platform compatibility
- ✅ Comprehensive error handling
- ✅ Real-time face detection with bounding boxes

## 📊 Model Performance

The model learns face embeddings using:
- **Architecture**: Siamese CNN with 3 conv blocks
- **Loss**: Contrastive loss with margin=1.0
- **Training**: 50% positive pairs, 50% negative pairs
- **Embedding**: 64-dimensional vector per face

## 🔍 Face Verification Process

1. **Capture** image from webcam or load from file
2. **Detect** face using Haar Cascade
3. **Extract** face embedding using trained model
4. **Compare** with gallery embeddings using Euclidean distance
5. **Verify** if distance < threshold → MATCH, else → REJECT

## 🎯 Tips for Best Results

- Ensure good lighting when capturing webcam images
- Position face centered in the camera frame
- Use consistent image quality for training data
- Adjust `VERIFICATION_THRESHOLD` based on your dataset
- Add more training data for better accuracy

## 🐛 Troubleshooting

### Camera Not Found
```python
# Try different camera index
frame = capture_and_display_webcam(camera_index=1)
```

### Model Not Training
- Check if dataset paths are correct
- Ensure you have enough training data (at least 2 images per person)
- Verify image files are valid (not corrupted)

### Low Verification Accuracy
- Adjust `VERIFICATION_THRESHOLD` (try values between 0.5-0.8)
- Increase training epochs
- Add data augmentation
- Use better quality training images

## 📦 Optional Features

### Gender Detection
To enable gender detection, install DeepFace:

```bash
pip install deepface
```

Then use the `detect_gender()` function in the notebook.

## 🐳 Docker Deployment (Jetson Nano)

This project now includes full Docker support for deployment on **NVIDIA Jetson Nano** with USB webcam access!

### Quick Deploy to Jetson Nano

```bash
# On Jetson Nano
./build_and_run_jetson.sh
```

### Features
- ✅ Optimized for ARM64 architecture
- ✅ CUDA GPU acceleration
- ✅ USB webcam device passthrough
- ✅ Real-time face verification
- ✅ Automatic startup and monitoring

### Documentation
- **[QUICKSTART_JETSON.md](QUICKSTART_JETSON.md)** - Quick deployment guide
- **[README_DOCKER.md](README_DOCKER.md)** - Complete Docker documentation

### Files Included
- `Dockerfile` - Container definition for Jetson Nano
- `docker-compose.yml` - Container orchestration
- `app.py` - Standalone application for Docker
- `utils.py` - Helper functions
- `requirements-jetson.txt` - Jetson-specific dependencies
- `build_and_run_jetson.sh` - Automated build/run script
- `test_setup.py` - Pre-deployment verification script

### Before Deploying

Test your setup locally:

```bash
python3 test_setup.py
```

This will verify:
- ✓ All required files are present
- ✓ Model can be loaded
- ✓ Camera is accessible
- ✓ Dependencies are installed

## 🤝 Contributing

Feel free to enhance this project by:
- Adding more face detection methods (MTCNN, RetinaFace)
- Implementing data augmentation
- Adding age estimation
- Creating a web interface
- Optimizing for edge devices

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision utilities
- Original Siamese Network paper for the architecture concept

---

**Note**: This system has been fully converted from Google Colab to run locally. All cloud-specific dependencies have been removed and replaced with local alternatives.
