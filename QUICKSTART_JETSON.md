# 🚀 QUICK START GUIDE - Jetson Nano Docker Deployment

## 📦 What's Included

Your project is now fully dockerized for NVIDIA Jetson Nano with:

✅ **Dockerfile** - Optimized for ARM64 with CUDA support
✅ **docker-compose.yml** - Easy container orchestration
✅ **app.py** - Main application with webcam support
✅ **utils.py** - Helper functions for face detection/verification
✅ **requirements-jetson.txt** - Jetson-specific dependencies
✅ **build_and_run_jetson.sh** - Automated build/run script
✅ **README_DOCKER.md** - Complete documentation

## 🎯 On Your Jetson Nano

### Step 1: Transfer Files

Copy your entire project directory to Jetson Nano:

```bash
# From your Mac (in project directory)
rsync -avz --progress /Users/soumikadas/Documents/SourceCodes/InnovativeProject/ \
    jetson@<JETSON_IP>:~/face_verification/
```

Or use USB drive / SD card

### Step 2: Connect USB Webcam

```bash
# On Jetson, verify camera
ls -l /dev/video*
# Should show /dev/video0
```

### Step 3: Run the Setup Script

```bash
cd ~/face_verification
./build_and_run_jetson.sh
```

Select option **1** (Build and run)

### Step 4: Use the System

Application will start with live camera feed showing:
- Green box = Person recognized
- Red box = Unknown person
- Person ID and distance score

**Controls:**
- Press **'q'** to quit
- Press **'s'** to save snapshot
- Press **'r'** to reload gallery

## 🔧 Manual Commands

### Build Image
```bash
docker build -t face-verification:jetson .
```

### Run Container
```bash
xhost +local:docker
docker-compose up -d
```

### View Logs
```bash
docker logs -f face_verification_jetson
```

### Stop Container
```bash
docker-compose down
```

## ⚙️ Configuration

Edit `docker-compose.yml` environment variables:

```yaml
- CAMERA_INDEX=0              # Change if using different USB port
- VERIFICATION_THRESHOLD=0.6  # Lower = stricter (0.4-0.8)
- IMG_SIZE=128               # Image size for processing
```

## 🎨 Key Features

1. **GPU Acceleration** - Uses Jetson's CUDA cores
2. **Real-time Processing** - Live webcam verification
3. **USB Webcam Support** - Automatic device detection
4. **Persistent Storage** - Snapshots saved to `output/` folder
5. **Resource Optimized** - Memory limits for Jetson Nano

## 📊 Expected Performance

- **FPS**: 15-25 on Jetson Nano 4GB
- **Latency**: ~50-100ms per frame
- **Memory**: ~1.5-2GB usage
- **Power**: Use 5V 4A barrel jack for stability

## 🐛 Quick Troubleshooting

### Camera Not Working
```bash
sudo chmod 666 /dev/video0
# Then restart container
```

### Display Not Showing
```bash
export DISPLAY=:0
xhost +local:docker
```

### Out of Memory
```bash
# Enable swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Slow Performance
```bash
# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 📁 Project Structure

```
InnovativeProject/
├── Dockerfile                          # Container definition
├── docker-compose.yml                  # Container orchestration
├── app.py                             # Main application
├── utils.py                           # Helper functions
├── requirements-jetson.txt            # Dependencies
├── build_and_run_jetson.sh           # Setup script
├── face_embedding_model_CLEAN.h5     # Your trained model
├── data_extracted/ref/               # Reference images
└── output/                           # Saved snapshots
```

## 🔄 Updating the System

### Update Code Only
```bash
docker-compose down
# Edit app.py or utils.py
docker-compose up -d
```

### Rebuild Everything
```bash
./build_and_run_jetson.sh
# Select option 2 (Build only)
# Then option 3 (Run only)
```

## 💡 Pro Tips

1. **Use barrel jack power** - USB power insufficient for GPU workloads
2. **Good lighting** - Improves face detection accuracy
3. **Position camera** - 1-2 meters from subject
4. **Test threshold** - Adjust based on your use case
5. **Monitor resources** - `docker stats face_verification_jetson`

## 🌐 What's Different from Notebook?

**Removed:**
- ❌ Jupyter notebook interface
- ❌ Manual cell execution
- ❌ Training code (model pre-loaded)
- ❌ Plotting/visualization

**Added:**
- ✅ Automatic startup
- ✅ Continuous operation
- ✅ Real-time display
- ✅ Snapshot saving
- ✅ Container isolation

## 📖 Full Documentation

See `README_DOCKER.md` for complete documentation including:
- Detailed installation
- Configuration options
- Performance optimization
- Security considerations
- Future enhancements

## 🆘 Need Help?

1. Check `docker logs face_verification_jetson`
2. Review README_DOCKER.md
3. Verify camera with `v4l2-ctl --device=/dev/video0 --info`
4. Test model loading manually in container shell

---

**Ready to deploy! 🎉**

Your face verification system is now containerized and ready for Jetson Nano!
