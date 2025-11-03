"""camera_utils.py
Utility helpers to capture images from a local USB webcam (OpenCV) or fall back to
Google Colab browser capture when running in Colab.

Usage in notebook:
    from camera_utils import capture_image
    live_path = capture_image("live_photo.jpg", camera_index=0, interactive=False)

This keeps the notebook unchanged but provides a single, tested place to call the local
camera. If you want, I can also patch the notebook to import and use this helper.
"""

import time
import cv2


def in_colab():
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def capture_image(filename: str = "photo.jpg", camera_index: int = 0, timeout: int = 5, interactive: bool = False, quality: float = 0.8) -> str:
    """Capture a single image and save to `filename`.

    - Prefers local USB webcam via OpenCV when available.
    - If running inside Google Colab, uses the browser-based capture method.

    Parameters:
        filename: output filename
        camera_index: integer index for cv2.VideoCapture
        timeout: seconds to wait for a valid frame (non-interactive)
        interactive: if True, shows an OpenCV preview window and waits for 'c' to capture
        quality: only used for Colab capture (JPEG quality)

    Returns:
        The filename that was written.

    Raises:
        Exception if capture fails or webcam cannot be opened.
    """
    # If in Colab, prefer the JS-based capture (keeps existing notebook behavior)
    if in_colab():
        try:
            from IPython.display import display, Javascript  # type: ignore
            from google.colab.output import eval_js  # type: ignore
            from base64 import b64decode
        except Exception as e:
            raise RuntimeError("Colab capture required but Colab APIs are not available: {}".format(e))

        js = Javascript('''
        async function takePhoto(quality) {
          const div = document.createElement('div');
          const capture = document.createElement('button');
          capture.textContent = 'Capture';
          div.appendChild(capture);

          const video = document.createElement('video');
          video.style.display = 'block';
          const stream = await navigator.mediaDevices.getUserMedia({video: true});
          document.body.appendChild(div);
          div.appendChild(video);
          video.srcObject = stream;
          await video.play();

          google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

          await new Promise((resolve) => capture.onclick = resolve);

          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          canvas.getContext('2d').drawImage(video, 0, 0);
          stream.getVideoTracks()[0].stop();
          return canvas.toDataURL('image/jpeg', quality);
        }
        ''')
        display(js)
        data = eval_js('takePhoto({})'.format(quality))
        binary = b64decode(data.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(binary)
        return filename

    # Local (non-Colab) path: use OpenCV VideoCapture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception(f"❌ Could not open webcam at index {camera_index}")

    frame = None
    start = time.time()
    while time.time() - start < timeout:
        ret, tmp = cap.read()
        if not ret:
            continue
        frame = tmp
        if interactive:
            cv2.imshow('Camera - press c to capture, q to cancel', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                break
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise Exception('Capture cancelled by user')
        else:
            # Non-interactive: take the first valid frame
            break

    cap.release()
    if interactive:
        cv2.destroyAllWindows()
    if frame is None:
        raise Exception('❌ Capture failed: no frame received from webcam')

    cv2.imwrite(filename, frame)
    return filename
