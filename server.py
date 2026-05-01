import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, jsonify

app = Flask(__name__)

@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

# --- Config ---
CAM_URLS = [
    "rtsp://admin:Admin%40123@192.168.1.133:554/unicaststream/4",   # CAM 1 — North
    "rtsp://admin:Admin%40123@192.168.1.132:554/unicaststream/4",   # CAM 2 — East
    "rtsp://admin:Admin%40123@192.168.1.134:554/unicaststream/4",   # CAM 3 — South
    "rtsp://admin:Admin%40123@192.168.1.135:554/unicaststream/4",   # CAM 4 — West
]
CAM_NAMES = ["North", "East", "South", "West"]
NUM_CAMS        = len(CAM_URLS)
STREAM_WIDTH    = 2560
STREAM_HEIGHT   = 480
STITCH_OVERLAP  = 80
FPS_TARGET      = 25

# --- Globals ---
panoramic_frame = None
frame_lock = threading.Lock()
stats = {"fps": 0, "stitched": False, "cam_ok": [False] * NUM_CAMS, "cam_names": CAM_NAMES}

# Pre-compute blend gradient (cached)
_alpha_cache = {}
def _get_alpha(overlap_px):
    if overlap_px not in _alpha_cache:
        _alpha_cache[overlap_px] = np.linspace(0, 1, overlap_px, dtype=np.float32).reshape(1, -1, 1)
    return _alpha_cache[overlap_px]


# ==============================
# THREADED CAMERA GRABBER
# ==============================

class CameraGrabber:
    """
    Dedicated thread per camera that continuously grabs frames.
    The main loop just reads the latest frame — never blocked by RTSP.
    """
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.frame = None
        self.ok = False
        self.lock = threading.Lock()
        self._cap = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self):
        try:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                print(f"[INFO] {self.name} camera connected")
                self._cap = cap
                return True
            else:
                cap.release()
                return False
        except Exception as e:
            print(f"[ERROR] {self.name} open failed: {e}")
            return False

    def _loop(self):
        reconnect_delay = 1.0
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._open():
                    with self.lock:
                        self.ok = False
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 1.5, 5.0)
                    continue
                reconnect_delay = 1.0

            try:
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.ok = True
                else:
                    print(f"[WARN] {self.name} frame drop, reconnecting...")
                    with self.lock:
                        self.ok = False
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    self._cap = None
                    time.sleep(0.5)
            except Exception as e:
                print(f"[ERROR] {self.name} read: {e}")
                with self.lock:
                    self.ok = False
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
                time.sleep(0.5)

    def get_frame(self):
        """Get latest frame and status (non-blocking)."""
        with self.lock:
            return self.ok, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._running = False


# ==============================
# STITCHING
# ==============================

def stitch_frames(frames, overlap_px):
    """
    Stitch N camera frames into one panoramic image with alpha-blended seams.
    Each frame is resized to (segment_w, STREAM_HEIGHT) and blended at overlaps.
    """
    n = len(frames)
    h = STREAM_HEIGHT
    segment_w = STREAM_WIDTH // n

    # Resize all frames to their segment size (with extra overlap margin)
    resized = []
    for f in frames:
        resized.append(cv2.resize(f, (segment_w + overlap_px, h)))

    canvas = np.empty((h, STREAM_WIDTH, 3), dtype=np.uint8)
    alpha = _get_alpha(overlap_px)

    for i in range(n):
        x_start = i * segment_w
        if i == 0:
            # First camera: full unique region, no left blend
            canvas[:, x_start:x_start + segment_w] = resized[i][:, :segment_w]
        else:
            # Overlap blend with previous camera
            left_strip = resized[i - 1][:, segment_w:segment_w + overlap_px].astype(np.float32)
            right_strip = resized[i][:, :overlap_px].astype(np.float32)
            actual_w = min(left_strip.shape[1], right_strip.shape[1], overlap_px)
            if actual_w > 0:
                a = alpha[:, :actual_w, :]
                blended = ((1.0 - a) * left_strip[:, :actual_w] +
                           a * right_strip[:, :actual_w]).astype(np.uint8)
                canvas[:, x_start:x_start + actual_w] = blended

            # Unique region for this camera (after overlap)
            unique_start = x_start + overlap_px
            unique_end = x_start + segment_w
            src_start = overlap_px
            src_end = src_start + (unique_end - unique_start)
            if unique_end <= STREAM_WIDTH and src_end <= resized[i].shape[1]:
                canvas[:, unique_start:unique_end] = resized[i][:, src_start:src_end]

    return canvas


# Demo colors for each camera slot
_DEMO_COLORS = [
    (0, 80, 160),   # CAM 1 - blue
    (0, 120, 40),   # CAM 2 - green
    (160, 80, 0),   # CAM 3 - orange
    (120, 0, 120),  # CAM 4 - purple
]

def generate_demo_frame(t, cam_index):
    """Generates a fake camera frame for demo."""
    w, h = STREAM_WIDTH // NUM_CAMS, STREAM_HEIGHT
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    color = _DEMO_COLORS[cam_index % len(_DEMO_COLORS)]
    frame[:] = color
    cx = int(w // 2 + np.sin(t * 0.8 + cam_index) * (w // 3))
    cy = int(h // 2 + np.cos(t * 0.6 + cam_index) * (h // 4))
    cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)
    label = f"CAM {cam_index + 1} — {CAM_NAMES[cam_index]}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


# ==============================
# STITCH LOOP (reads latest frames from grabber threads)
# ==============================

def stitch_loop(grabbers):
    global panoramic_frame, stats

    interval = 1.0 / FPS_TARGET
    fps_counter, fps_time, fps_val = 0, time.time(), 0
    t = 0
    segment_size = (STREAM_WIDTH // NUM_CAMS, STREAM_HEIGHT)

    while True:
        try:
            loop_start = time.time()

            # Non-blocking read from all grabber threads
            frames = []
            cam_ok = []
            for i, g in enumerate(grabbers):
                ok, frame = g.get_frame()
                cam_ok.append(ok)
                if frame is None:
                    frame = generate_demo_frame(t, i)
                    t += interval
                frames.append(frame)

            demo_mode = not any(cam_ok)

            # Resize all to segment size
            frames = [cv2.resize(f, segment_size) for f in frames]

            # Stitch
            try:
                stitched = stitch_frames(frames, STITCH_OVERLAP)
            except Exception as e:
                print(f"[WARN] Stitch error: {e}")
                stitched = np.hstack(frames)

            # FPS counter
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps_val = fps_counter
                fps_counter = 0
                fps_time = now

            mode_str = "DEMO" if demo_mode else "LIVE"
            cv2.putText(stitched, f"{fps_val} fps | {mode_str}",
                        (STREAM_WIDTH - 160, STREAM_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

            with frame_lock:
                panoramic_frame = stitched
                stats.update({"fps": fps_val, "cam_ok": cam_ok,
                              "stitched": True,
                              "demo_mode": demo_mode})

            elapsed = time.time() - loop_start
            time.sleep(max(0, interval - elapsed))

        except Exception as e:
            print(f"[ERROR] Stitch loop: {e}")
            time.sleep(0.3)


def gen_mjpeg():
    while True:
        with frame_lock:
            frame = panoramic_frame

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')

        time.sleep(0.03)


@app.route('/stream')
def stream():
    return Response(gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    with frame_lock:
        frame = panoramic_frame
    if frame is None:
        return "No frame yet", 503
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/status')
def status():
    return jsonify(stats)


@app.route('/')
def index():
    return "BRAHMA server running. Visit /stream for MJPEG feed."


if __name__ == '__main__':
    # Start dedicated grabber threads for each camera (1 thread per camera)
    grabbers = []
    for i, url in enumerate(CAM_URLS):
        g = CameraGrabber(url, f"CAM_{i + 1}_{CAM_NAMES[i]}")
        grabbers.append(g)

    # Start stitch loop in its own thread
    t = threading.Thread(target=stitch_loop, args=(grabbers,), daemon=True)
    t.start()

    print(f"BRAHMA Panoramic Camera Server ({NUM_CAMS} cameras)")
    print("Stream : http://localhost:5050/stream")
    print("Status : http://localhost:5050/status")
    app.run(host='0.0.0.0', port=5050, threaded=True)
