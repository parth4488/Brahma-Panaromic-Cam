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
CAM_LEFT_INDEX  = "rtsp://admin:Admin%40123@192.168.154.66:554/unicaststream/1"   # change to your left camera index
CAM_RIGHT_INDEX = "rtsp://admin:Admin%40123@192.168.154.67:554/unicaststream/1"   # change to your right camera index
STREAM_WIDTH    = 1280
STREAM_HEIGHT   = 480
STITCH_OVERLAP  = 80  # px overlap for blending seam
FPS_TARGET      = 20

# --- Globals ---
panoramic_frame = None
frame_lock = threading.Lock()
stats = {"fps": 0, "left_ok": False, "right_ok": False, "stitched": False}


def alpha_blend_seam(left, right, overlap_px):
    """Simple linear alpha blend at the seam between two frames."""
    h = min(left.shape[0], right.shape[0])
    left  = left[:h]
    right = right[:h]

    half_w = STREAM_WIDTH // 2

    # Crop each frame to half width + overlap
    left_crop  = left[:, :half_w + overlap_px]
    right_crop = right[:, max(0, right.shape[1] - half_w - overlap_px):]

    # Resize right crop to fill its half
    rw = STREAM_WIDTH - half_w
    right_resized = cv2.resize(right_crop, (rw + overlap_px, h))

    # Build canvas
    canvas = np.zeros((h, STREAM_WIDTH, 3), dtype=np.uint8)
    canvas[:, :half_w] = cv2.resize(left_crop[:, :half_w], (half_w, h))

    # Blend overlap zone
    for i in range(overlap_px):
        alpha = i / overlap_px
        col_l = half_w - overlap_px + i
        col_r = i
        if 0 <= col_l < STREAM_WIDTH and col_r < right_resized.shape[1]:
            canvas[:, col_l] = cv2.addWeighted(
                left_crop[:, col_l].reshape(h, 1, 3) if col_l < left_crop.shape[1] else np.zeros((h, 1, 3), np.uint8),
                1 - alpha,
                right_resized[:, col_r].reshape(h, 1, 3),
                alpha, 0
            ).reshape(h, 3)

    # Fill right half
    right_fill = cv2.resize(right_resized[:, overlap_px:], (rw, h))
    canvas[:, half_w:] = right_fill

    return canvas


def generate_demo_frame(t, side="left"):
    """Generates a fake camera frame for demo (when no real cameras attached)."""
    w, h = STREAM_WIDTH // 2, STREAM_HEIGHT
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    color = (0, 80, 160) if side == "left" else (0, 120, 40)
    frame[:] = color

    # Animated circle
    cx = int(w // 2 + np.sin(t * 0.8) * (w // 3))
    cy = int(h // 2 + np.cos(t * 0.6) * (h // 4))
    cv2.circle(frame, (cx, cy), 40, (255, 255, 255), -1)

    label = "CAM L" if side == "left" else "CAM R"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"t={t:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def camera_loop():
    global panoramic_frame, stats

    cap_l = cv2.VideoCapture(CAM_LEFT_INDEX)
    cap_r = cv2.VideoCapture(CAM_RIGHT_INDEX)

    left_ok  = cap_l.isOpened()
    right_ok = cap_r.isOpened()
    demo_mode = not (left_ok and right_ok)

    if left_ok:
        cap_l.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH // 2)
        cap_l.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    if right_ok:
        cap_r.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_WIDTH // 2)
        cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)

    interval = 1.0 / FPS_TARGET
    fps_counter, fps_time, fps_val = 0, time.time(), 0
    t = 0

    while True:
        loop_start = time.time()

        if demo_mode:
            frame_l = generate_demo_frame(t, "left")
            frame_r = generate_demo_frame(t, "right")
            t += interval
            left_ok = right_ok = True
        else:
            ret_l, frame_l = cap_l.read()
            ret_r, frame_r = cap_r.read()
            left_ok  = ret_l
            right_ok = ret_r

            if not ret_l:
                frame_l = generate_demo_frame(t, "left")
            if not ret_r:
                frame_r = generate_demo_frame(t, "right")

        # Resize both to half width
        half = (STREAM_WIDTH // 2, STREAM_HEIGHT)
        frame_l = cv2.resize(frame_l, half)
        frame_r = cv2.resize(frame_r, half)

        # Stitch
        stitched = alpha_blend_seam(frame_l, frame_r, STITCH_OVERLAP)

        # Timestamp overlay
        ts = time.strftime("%H:%M:%S")
        cv2.putText(stitched, f"BRAHMA  {ts}  {'DEMO' if demo_mode else 'LIVE'}",
                    (10, STREAM_HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 220, 255), 1, cv2.LINE_AA)

        # FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps_val = fps_counter
            fps_counter = 0
            fps_time = time.time()

        cv2.putText(stitched, f"{fps_val} fps",
                    (STREAM_WIDTH - 70, STREAM_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)

        with frame_lock:
            panoramic_frame = stitched.copy()
            stats.update({"fps": fps_val, "left_ok": left_ok,
                          "right_ok": right_ok, "stitched": True,
                          "demo_mode": demo_mode})

        elapsed = time.time() - loop_start
        time.sleep(max(0, interval - elapsed))


def gen_mjpeg():
    while True:
        with frame_lock:
            frame = panoramic_frame
        if frame is None:
            time.sleep(0.02)
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(1.0 / FPS_TARGET)


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
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("BRAHMA Panoramic Camera Server")
    print("Stream : http://localhost:5050/stream")
    print("Status : http://localhost:5050/status")
    app.run(host='0.0.0.0', port=5050, threaded=True)
