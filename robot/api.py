from flask import Flask, request, jsonify
import picamera
from io import BytesIO

app = Flask(__name__)

# Initialize camera object
camera = picamera.PiCamera()

@app.route('/camera/connect', methods=['GET'])
def connect_camera():
    # Wait for the camera to warm up
    camera.start_preview()
    return 'Camera connected successfully'

@app.route('/camera/capture', methods=['POST'])
def capture_image():
    # Capture an image and return it as a response
    img_io = BytesIO()
    camera.capture(img_io, 'jpeg')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route('/camera/disconnect', methods=['GET'])
def disconnect_camera():
    # Close the camera connection
    camera.close()
    return 'Camera disconnected successfully'

if __name__ == '__main__':
    app.run(debug=True)
