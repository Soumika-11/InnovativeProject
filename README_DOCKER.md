# Face Verification System - Docker Deployment for Jetson Nano

## 🎯 Overview

This Docker setup allows you to run the Face Verification System on NVIDIA Jetson Nano with USB webcam support. The container includes all necessary dependencies and is optimized for ARM64 architecture with CUDA acceleration.

## 📋 Prerequisites

### Hardware Requirements
- **NVIDIA Jetson Nano** (2GB or 4GB model)
- **USB Webcam** (compatible with V4L2)
- **MicroSD Card** (32GB+ recommended)
- **Power Supply** (5V 4A barrel jack recommended for GPU workloads)

### Software Requirements
- **JetPack 4.6** or later (includes L4T and NVIDIA Container Runtime)
- **Docker** installed and configured
- **docker-compose** installed

## 🚀 Quick Start

### 1. Install Docker and NVIDIA Container Runtime

If not already installed on your Jetson Nano:

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y docker.io

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install docker-compose
sudo apt-get install -y docker-compose

# Verify NVIDIA runtime
docker info | grep -i runtime
```

### 2. Prepare Your Files

Ensure you have the following files in your project directory:

```
InnovativeProject/
├── face_embedding_model_CLEAN.h5     # Trained model (required)
├── model_architecture.json            # Model architecture
├── data_extracted/
│   └── ref/
│       └── short_references_final/    # Reference images (required)
├── Dockerfile
├── docker-compose.yml
├── requirements-jetson.txt
├── app.py
├── utils.py
└── build_and_run_jetson.sh
```

### 3. Connect USB Webcam

```bash
# Verify camera is detected
ls -l /dev/video*

# Test camera (optional)
v4l2-ctl --device=/dev/video0 --info
```

### 4. Build and Run

Make the script executable and run:

```bash
chmod +x build_and_run_jetson.sh
./build_and_run_jetson.sh
```

Follow the interactive menu to:
1. Build the Docker image
2. Run the container
3. View logs

## 🔧 Manual Build and Run

### Build the Docker Image

```bash
docker build -t face-verification:jetson .
```

**Note:** Building may take 15-20 minutes on Jetson Nano.

### Run with docker-compose

```bash
# Allow X11 access for display
xhost +local:docker

# Start the container
docker-compose up -d

# View logs
docker logs -f face_verification_jetson
```

### Run with Docker Command

```bash
docker run -it --rm \
  --runtime nvidia \
  --privileged \
  --device /dev/video0:/dev/video0 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/data_extracted:/app/data_extracted \
  -v $(pwd)/face_embedding_model_CLEAN.h5:/app/face_embedding_model_CLEAN.h5 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --network host \
  --name face_verification_jetson \
  face-verification:jetson
```

## ⚙️ Configuration

### Environment Variables

You can customize the system using environment variables in `docker-compose.yml`:

```yaml
environment:
  - CAMERA_INDEX=0                    # USB camera index (0, 1, 2, ...)
  - VERIFICATION_THRESHOLD=0.6        # Distance threshold for matching
  - IMG_SIZE=128                      # Input image size
  - EMBEDDING_DIM=64                  # Embedding dimension
```

### Camera Selection

If you have multiple USB cameras:

```bash
# List all video devices
ls -l /dev/video*

# Set camera index in docker-compose.yml
CAMERA_INDEX=1  # Use /dev/video1
```

### Adjust Verification Threshold

Lower threshold = stricter matching
Higher threshold = more lenient matching

Recommended range: **0.4 - 0.8**

## 🎮 Usage

### Interactive Controls

When the application is running:

- **'q'** - Quit application
- **'s'** - Save snapshot to `/app/output/`
- **'r'** - Reload gallery embeddings

### Viewing Output

The container display window shows:
- Real-time video feed
- Face bounding boxes (green = match, red = unknown)
- Person ID and distance
- FPS counter
- Verification threshold

### Accessing Saved Snapshots

Snapshots are saved to the `output/` directory on your host:

```bash
ls -lh output/
```

## 🐛 Troubleshooting

### Camera Not Detected

```bash
# Check if camera is accessible
ls -l /dev/video*

# Verify permissions
sudo chmod 666 /dev/video0

# Check Docker has device access
docker exec -it face_verification_jetson ls -l /dev/video*
```

### Display Issues

```bash
# Allow X11 access
xhost +local:docker

# Check DISPLAY variable
echo $DISPLAY

# Try setting explicitly
export DISPLAY=:0
```

### Low FPS / Performance Issues

1. **Use barrel jack power supply** (not USB)
2. **Enable maximum performance mode**:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```
3. **Reduce camera resolution** in docker-compose.yml
4. **Close other applications** to free resources

### Out of Memory Errors

For 2GB Jetson Nano models:

```bash
# Reduce memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 1.5G  # Reduce from 3G
```

### Model Not Loading

```bash
# Verify model file exists
ls -lh face_embedding_model_CLEAN.h5

# Check model in container
docker exec -it face_verification_jetson ls -lh /app/face_embedding_model_CLEAN.h5
```

## 📊 Performance Optimization

### For Jetson Nano 2GB
- Use lower resolution (480p)
- Reduce batch processing
- Close unnecessary applications
- Swap file recommended (4GB+)

### For Jetson Nano 4GB
- Can handle 720p resolution
- Better performance with multiple faces
- More stable for long-running applications

### Enable Swap (if not already configured)

```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 🔐 Security Notes

- Container runs in **privileged mode** for camera access
- X11 access is opened with `xhost +local:docker`
- Consider restricting access in production environments

## 📦 Container Management

### View Running Containers

```bash
docker ps
```

### Stop Container

```bash
docker stop face_verification_jetson
```

### Remove Container

```bash
docker rm face_verification_jetson
```

### View Logs

```bash
docker logs face_verification_jetson
docker logs -f face_verification_jetson  # Follow mode
```

### Shell Access

```bash
docker exec -it face_verification_jetson /bin/bash
```

### Resource Usage

```bash
# Monitor container resources
docker stats face_verification_jetson
```

## 🔄 Updating

### Update Code Only

```bash
# Stop container
docker-compose down

# Update files (app.py, utils.py)
# Restart
docker-compose up -d
```

### Rebuild Image

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📁 Directory Structure in Container

```
/app/
├── face_embedding_model_CLEAN.h5
├── model_architecture.json
├── app.py
├── utils.py
├── data_extracted/
│   └── ref/
│       └── short_references_final/
└── output/                    # Mounted from host
```

## 🌐 Future Enhancements

Potential additions for production deployment:

1. **REST API** - Flask/FastAPI web interface
2. **MQTT** - IoT integration for remote monitoring
3. **Database** - Store verification logs
4. **Multi-camera** - Support multiple USB cameras
5. **Cloud sync** - Upload results to cloud storage

## 📖 Additional Resources

- [NVIDIA Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)
- [JetPack Documentation](https://docs.nvidia.com/jetson/)
- [L4T TensorFlow Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorflow)
- [Docker on Jetson](https://github.com/dusty-nv/jetson-containers)

## 🤝 Support

For issues specific to:
- **Jetson Nano** - Check NVIDIA Jetson forums
- **Docker** - Verify Docker and NVIDIA runtime installation
- **Model/Code** - Check main README.md

## 📝 License

This Docker setup is part of the Face Verification System project.

---

**Note:** This setup is optimized for NVIDIA Jetson Nano. For other platforms (x86_64, Raspberry Pi), modifications to the Dockerfile base image will be needed.
