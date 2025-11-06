# Dockerfile for Face Verification System on Jetson Nano
# Based on NVIDIA L4T (Linux for Tegra) with TensorFlow support

FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create app directory
WORKDIR ${APP_HOME}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    libhdf5-dev \
    libhdf5-serial-dev \
    hdf5-tools \
    libatlas-base-dev \
    gfortran \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    v4l-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install Python dependencies
# Note: TensorFlow is already included in the base image
COPY requirements.txt ${APP_HOME}/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY face_embedding_model_CLEAN.h5 ${APP_HOME}/
COPY model_architecture.json ${APP_HOME}/
COPY data_extracted/ ${APP_HOME}/data_extracted/

# Create output directory for results
RUN mkdir -p ${APP_HOME}/output

# Copy the main application script
COPY app.py ${APP_HOME}/
COPY utils.py ${APP_HOME}/

# Set permissions
RUN chmod +x ${APP_HOME}/app.py

# Expose any ports if needed (for future web interface)
EXPOSE 5000

# Default command - run the face verification app
CMD ["python3", "app.py"]
