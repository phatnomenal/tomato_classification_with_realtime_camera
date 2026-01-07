from flask import Flask, render_template, Response, jsonify, send_file
from camera import VideoCamera
import os
import csv
import signal
import sys
import atexit

app = Flask(__name__)

# ================= GLOBAL CAMERA =================
camera = None


def get_camera():
    global camera
    if camera is None:
        camera = VideoCamera()
    return camera


# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')


def gen(camera):
    """
    MJPEG streaming generator
    """
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(
        gen(get_camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/start_cam')
def start_cam():
    cam = get_camera()
    cam.start()
    return jsonify({"status": "camera started"}), 200


@app.route('/stop_cam')
def stop_cam():
    global camera
    if camera:
        camera.stop()
    return jsonify({"status": "camera stopped"}), 200


@app.route('/api/logs')
def get_logs():
    log_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'detection_logs.csv'
    )

    logs = []

    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                logs = rows[-10:][::-1]  # newest first
        except Exception as e:
            print("Error reading logs:", e)

    return jsonify(logs)


@app.route('/download_csv')
def download_csv():
    log_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'detection_logs.csv'
    )

    if os.path.exists(log_file):
        return send_file(log_file, as_attachment=True)
    else:
        return jsonify({"error": "No logs found"}), 404


# ================= CLEANUP =================

def cleanup(signum=None, frame=None):
    global camera
    if camera:
        camera.stop()
        print("Camera released")

    if signum is not None:
        sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
atexit.register(cleanup)


# ================= MAIN =================

if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
        use_reloader=False
    )
