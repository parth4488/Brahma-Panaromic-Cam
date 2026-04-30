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
CAM_LEFT_INDEX  = "rtsp://admin:Admin%40123@192.168.1.132:554/unicaststream/1"
CAM_RIGHT_INDEX = "rtsp://admin:Admin%40123@192.168.1.133:554/unicaststream/1"

STREAM_WIDTH    = 1280
STREAM_HEIGHT   = 480
STITCH_OVERLAP  = 80  # initial fallback
FPS_TARGET      = 20

# --- Globals ---
panoramic_frame = None
frame_lock = threading.Lock()
stats = {
    "fps": 0,
    "left_ok": False,
    "right_ok": False,
    "stitched": False,
    "overlap": STITCH_OVERLAP
}

# ==============================
# 🔧 CAMERA HELPERS
# ==============================

def open_camera(url):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def safe_read(cap, url, side):
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] {side} camera reconnecting...")
        cap.release()
        time.sleep(0.5)
        cap = open_camera(url)
        return cap, False, None
    return cap, True, frame


# ==============================
# 🔥 AUTO OVERLAP DETECTION
# ==============================

def compute_overlap_offset(frame_l, frame_r, max_search=200):
    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(gray_l, None)
    kp2, des2 = orb.detectAndCompute(gray_r, None)

    if des1 is None or des2 is None:
        return STITCH_OVERLAP

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 10:
        return STITCH_OVERLAP

    shifts = []
    for m in matches:
        x1 = kp1[m.queryIdx].pt[0]
        x2 = kp2[m.trainIdx].pt[0]
        shift = int(x1 - x2)
        if abs(shift) < max_search:
            shifts.append(shift)

    if not shifts:
        return STITCH_OVERLAP

    median_shift = int(np.median(shifts))
    overlap = abs(median_shift)

    print(f"[INFO] Auto overlap detected: {overlap}px")
    return max(20, min(overlap, 200))


# ==============================
# 🎥 STITCHING
# ==============================

def alpha_blend_seam(left, right, overlap_px):
    h = min(left.shape[0], right.shape[0])
    left  = left[:h]
    right = right[:h]

    half_w = STREAM_WIDTH // 2

    left_crop  = left[:, :half_w + overlap_px]
    right_crop = right[:, max(0, right.shape[1] - half_w - overlap_px):]

    rw = STREAM_WIDTH - half_w
    right_resized = cv2.resize(right_crop, (rw + overlap_px, h))

    canvas = np.zeros((h, STREAM_WIDTH, 3), dtype=np.uint8)
    canvas[:, :half_w] = cv2.resize(left_crop[:, :half_w], (half_w, h))

    # Blend overlap
    for i in range(overlap_px):
        alpha = i / overlap_px
        col_l = half_w - overlap_px + i
        col_r = i

        if 0 <= col_l < STREAM_WIDTH and col_r < right_resized.shape[1]:
            canvas[:, col_l] = cv2.addWeighted(
                left_crop[:, col_l].reshape(h, 1, 3)
                if col_l < left_crop.shape[1]
                else np.zeros((h, 1, 3), np.uint8),
                1 - alpha,
                right_resized[:, col_r].reshape(h, 1, 3),
                alpha, 0
            ).reshape(h, 3)

    right_fill = cv2.resize(right_resized[:, overlap_px:], (rw, h))
    canvas[:, half_w:] = right_fill

    return canvas


# ==============================
# 🎥 CAMERA LOOP
# ==============================

def camera_loop():
    global panoramic_frame, stats, STITCH_OVERLAP

    cap_l = open_camera(CAM_LEFT_INDEX)
    cap_r = open_camera(CAM_RIGHT_INDEX)

    overlap_initialized = False

    fps_counter = 0
    last_fps_time = time.time()
    fps_val = 0

    while True:
        cap_l, ok_l, frame_l = safe_read(cap_l, CAM_LEFT_INDEX, "LEFT")
        cap_r, ok_r, frame_r = safe_read(cap_r, CAM_RIGHT_INDEX, "RIGHT")

        if not ok_l or not ok_r:
            continue

        # Resize early
        half = (STREAM_WIDTH // 2, STREAM_HEIGHT)
        frame_l = cv2.resize(frame_l, half)
        frame_r = cv2.resize(frame_r, half)

        # 🔥 Auto overlap (run once)
        if not overlap_initialized:
            STITCH_OVERLAP = compute_overlap_offset(frame_l, frame_r)
            overlap_initialized = True

        stitched = alpha_blend_seam(frame_l, frame_r, STITCH_OVERLAP)

        # FPS
        fps_counter += 1
        if time.time() - last_fps_time >= 1:
            fps_val = fps_counter
            fps_counter = 0
            last_fps_time = time.time()

        # Overlay
        ts = time.strftime("%H:%M:%S")
        cv2.putText(stitched, f"BRAHMA {ts}",
                    (10, STREAM_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 255, 200), 1)

        cv2.putText(stitched, f"{fps_val} FPS",
                    (STREAM_WIDTH - 100, STREAM_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 255, 200), 1)

        with frame_lock:
            panoramic_frame = stitched
            stats.update({
                "fps": fps_val,
                "left_ok": ok_l,
                "right_ok": ok_r,
                "stitched": True,
                "overlap": STITCH_OVERLAP
            })


# ==============================
# 🌐 MJPEG STREAM
# ==============================

def gen_mjpeg():
    while True:
        with frame_lock:
            frame = panoramic_frame

        if frame is None:
            continue

        _, buf = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')


@app.route('/stream')
def stream():
    return Response(gen_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    with frame_lock:
        frame = panoramic_frame

    if frame is None:
        return "No frame yet", 503

    _, buf = cv2.imencode('.jpg', frame,
                          [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(buf.tobytes(), mimetype='image/jpeg')


@app.route('/status')
def status():
    return jsonify(stats)


@app.route('/')
def index():
    return "BRAHMA server running. Visit /stream"


# ==============================
# 🚀 MAIN
# ==============================

if __name__ == '__main__':
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("BRAHMA Panoramic Camera Server")
    print("Stream : http://localhost:5050/stream")
    print("Status : http://localhost:5050/status")

    app.run(host='0.0.0.0', port=5050, threaded=True)