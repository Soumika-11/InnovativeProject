#!/bin/bash

# Build and Run Script for Face Verification System on Jetson Nano
# This script helps build and deploy the Docker container

set -e

echo "================================================"
echo "Face Verification System - Jetson Nano Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="face_verification_jetson"
IMAGE_NAME="face-verification:jetson"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson Nano
check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        print_warn "This script is designed for NVIDIA Jetson Nano"
        print_warn "Continuing anyway, but may encounter issues..."
    else
        print_info "Running on Jetson Nano"
        cat /etc/nv_tegra_release
    fi
}

# Check if required files exist
check_files() {
    print_info "Checking required files..."
    
    if [ ! -f "face_embedding_model_CLEAN.h5" ]; then
        print_error "Model file 'face_embedding_model_CLEAN.h5' not found!"
        exit 1
    fi
    
    if [ ! -d "data_extracted/ref/short_references_final" ]; then
        print_error "Reference images directory not found!"
        exit 1
    fi
    
    print_info "All required files present"
}

# Check USB camera
check_camera() {
    print_info "Checking for USB camera..."
    
    if [ -c "/dev/video0" ]; then
        print_info "Camera found at /dev/video0"
        v4l2-ctl --device=/dev/video0 --info 2>/dev/null || true
    else
        print_warn "No camera found at /dev/video0"
        print_warn "Please connect USB webcam before running"
    fi
}

# Setup X11 for display
setup_x11() {
    print_info "Setting up X11 for display..."
    
    # Allow Docker to access X11
    xhost +local:docker 2>/dev/null || print_warn "Could not run xhost (may not be needed)"
    
    # Export DISPLAY if not set
    if [ -z "$DISPLAY" ]; then
        export DISPLAY=:0
        print_warn "DISPLAY not set, using :0"
    fi
    
    print_info "Display: $DISPLAY"
}

# Stop existing container
stop_container() {
    print_info "Checking for existing container..."
    
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_info "Stopping and removing existing container..."
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    fi
}

# Build Docker image
build_image() {
    print_info "Building Docker image..."
    print_warn "This may take 10-20 minutes on Jetson Nano..."
    
    docker build -t ${IMAGE_NAME} .
    
    if [ $? -eq 0 ]; then
        print_info "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Run container
run_container() {
    print_info "Starting container..."
    
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_info "Container started successfully!"
        print_info "Container name: ${CONTAINER_NAME}"
    else
        print_error "Failed to start container"
        exit 1
    fi
}

# Show logs
show_logs() {
    print_info "Showing container logs (Ctrl+C to exit)..."
    sleep 2
    docker logs -f ${CONTAINER_NAME}
}

# Main menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo "1) Build and run (full setup)"
    echo "2) Build only"
    echo "3) Run only (use existing image)"
    echo "4) Stop container"
    echo "5) Show logs"
    echo "6) Shell into container"
    echo "7) Exit"
    echo ""
    read -p "Enter choice [1-7]: " choice
    
    case $choice in
        1)
            check_jetson
            check_files
            check_camera
            setup_x11
            stop_container
            build_image
            run_container
            show_logs
            ;;
        2)
            check_files
            build_image
            ;;
        3)
            check_camera
            setup_x11
            stop_container
            run_container
            show_logs
            ;;
        4)
            stop_container
            print_info "Container stopped"
            ;;
        5)
            show_logs
            ;;
        6)
            print_info "Opening shell in container..."
            docker exec -it ${CONTAINER_NAME} /bin/bash
            ;;
        7)
            print_info "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            show_menu
            ;;
    esac
}

# Main execution
main() {
    echo ""
    show_menu
}

main
