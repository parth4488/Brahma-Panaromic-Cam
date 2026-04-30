# 🎥 BRAHMA — Panoramic Camera System

> Multi-camera panoramic live streaming system that merges two camera feeds into a single wide-angle view, accessible via web browser.

---

## 📌 Problem Statement

Most cameras have a limited field of view (FOV), showing only one direction at a time. Even PTZ cameras, while able to rotate, still display only a single view at any moment — making real-time wide-area monitoring difficult.

## 💡 Proposed Solution

BRAHMA uses **multiple camera sensors**, each capturing a different direction. A central controller collects these inputs, **stitches the frames**, and outputs a single live panoramic stream — creating a wide, continuous view and overcoming the limitations of traditional cameras.

---

## 🏗️ System Architecture

```
[Camera LEFT]  ──RTSP──┐
                        ├──► server.py (Flask) ──MJPEG──► dashboard.html (Browser)
[Camera RIGHT] ──RTSP──┘
```

---

## 📁 Project Structure

```
Brahma-Panaromic-Cam/
│
├── server.py        # Backend — captures, stitches & streams camera feeds
├── dashboard.html   # Frontend — live panoramic viewer in browser
└── README.md
```

---

## ⚙️ Requirements

- Python 3.9+
- OpenCV
- Flask
- NumPy
- Two IP cameras with RTSP support

Install dependencies:
```bash
pip install flask opencv-python numpy
```

---

## 🚀 Setup & Run

### Step 1 — Configure cameras in `server.py`

```python
CAM_LEFT_INDEX  = "rtsp://admin:Admin%40123@192.168.x.x:554/unicaststream/1"
CAM_RIGHT_INDEX = "rtsp://admin:Admin%40123@192.168.x.y:554/unicaststream/1"
```

> **Note:** Special characters in password must be URL-encoded. `@` → `%40`

### Step 2 — Start the server

```bash
python3 server.py
```

Server will start at:
```
http://0.0.0.0:5050
```

### Step 3 — Open Dashboard

Open `dashboard.html` in browser and enter your server IP:
```
http://<your-server-ip>:5050
```
Click **Connect** — panoramic stream will appear live! ✅

---

## 🌐 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /stream` | Live MJPEG panoramic stream |
| `GET /snapshot` | Single JPEG frame capture |
| `GET /status` | Camera status & FPS (JSON) |

---

## 🔧 Configuration Options

| Parameter | Default | Description |
|---|---|---|
| `STREAM_WIDTH` | 1280 | Output panoramic width (px) |
| `STREAM_HEIGHT` | 480 | Output panoramic height (px) |
| `STITCH_OVERLAP` | 80 | Seam blend overlap zone (px) |
| `FPS_TARGET` | 20 | Target frames per second |

---

## 🎯 Features

- ✅ Dual camera RTSP stream capture
- ✅ Real-time alpha-blend stitching at seam
- ✅ Live MJPEG stream over HTTP
- ✅ Auto demo mode (if cameras unavailable)
- ✅ Snapshot capture
- ✅ Live FPS counter & camera status
- ✅ Sleek dark-mode web dashboard
- ✅ Seam line toggle for debugging

---

## 🌍 Public Access (Optional)

For access outside local network, use **ngrok**:

```bash
# Download from https://ngrok.com
ngrok http 5050
```

This generates a public URL like:
```
https://abc123.ngrok.io
```

Enter this URL in the dashboard's server field to access from anywhere.

---

## 👥 Team

**Project BRAHMA** — Built for Hackathon

---

## 📄 License

MIT License — Free to use and modify.
