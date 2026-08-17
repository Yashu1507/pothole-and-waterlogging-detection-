# 🌧️ AI Waterlogging + Road Hazard Detection System

An AI-powered system that uses drone/CCTV footage to automatically detect **waterlogging** and **potholes** on roads, score their severity, and push real-time alerts to a live dashboard for faster civic response.

Built for ELCIA 2026 Hackathon. — addresses the official challenge track on **waterlogging, potholes, drainage overflow, and road risks.**

---

## 🚨 The Problem

Cities rely on manual reporting or infrequent inspections to catch road hazards like waterlogging and potholes. By the time issues are reported, roads may already be unsafe — causing accidents, traffic jams, and vehicle damage. There's no automated, real-time way to detect and prioritize these hazards at scale.

## ✅ Our Solution

A drone/camera feed is processed by two AI detection models (waterlogging + pothole), which classify severity, tag the location/zone, capture evidence frames, and instantly alert a live dashboard — so city teams know exactly where to send an inspection crew, and how urgently.

---

## 🏗️ Architecture

```
Drone / CCTV feed
       │
       ▼
 Frame extraction (sampled frames + GPS tag)
       │
  ┌────┴─────┐
  ▼           ▼
Waterlogging  Pothole
detection     detection
  └────┬─────┘
       ▼
Severity + location scoring
       │
       ▼
Evidence capture (frame + timestamp)
       │
       ▼
Backend API (FastAPI) ──stores──▶ Database
       │
       └──broadcasts (WebSocket)──▶ Frontend dashboard
```

See `/docs/architecture.png` for the full diagram.

---

## ✨ Features

- 🎯 **Dual hazard detection** — waterlogging (segmentation) + potholes (object detection)
- 📊 **Automatic severity scoring** — LOW / MEDIUM / HIGH based on detected area/coverage
- 📍 **Zone-based location tagging** — maps GPS/pixel coordinates to city zones
- 📸 **Evidence capture** — every alert is backed by a timestamped frame
- ⚡ **Real-time dashboard** — live alert feed over WebSocket, no manual refresh
- 🗺️ **Map view** — see hazards plotted geographically
- ⚠️ **Recommended actions** — auto-suggested response (e.g. "Road inspection", "Monitor")

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Detection models | YOLOv8 / Roboflow (pothole), segmentation model (waterlogging) |
| Backend | FastAPI, SQLAlchemy, WebSockets |
| Database | SQLite (dev) |
| Frontend | React, Tailwind CSS, react-leaflet |
| Model hosting | Roboflow serverless inference |

---

## 📁 Repo Structure

```
.
├── backend/
│   ├── main.py            # FastAPI app, API + WebSocket endpoints
│   ├── models.py          # DB schema
│   ├── database.py        # DB connection setup
│   ├── inference.py       # Calls detection model, builds hazard events
│   ├── requirements.txt
│   └── evidence/          # saved evidence frames
├── frontend/
│   ├── src/
│   │   ├── components/    # AlertCard, AlertFeed, ZoneMap
│   │   ├── hooks/         # useAlertSocket.js
│   │   ├── App.jsx
│   │   └── api.js
│   ├── package.json
│   └── ...
├── models/
│   └── training_notebooks/  # Colab/Jupyter notebooks used for training
├── docs/
│   └── architecture.png
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Roboflow account + API key (or local YOLO weights)

### 1. Clone the repo
```bash
git clone https://github.com/Yashu1507/pothole-and-waterlogging-detection-.git
cd <repo-name>
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env       # fill in your API keys
uvicorn main:app --reload --port 8000
```
Backend runs at `http://localhost:8000` — API docs at `http://localhost:8000/docs`.

### 3. Frontend setup
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173` (or whatever Vite/CRA assigns).

### 4. Run detection
```bash
cd backend
python inference.py --source path/to/video.mp4
```
This will detect hazards frame-by-frame and post events to the backend, which the dashboard picks up live.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events` | Submit a new hazard detection event |
| `GET` | `/events` | Get recent events (default limit 50) |
| `GET` | `/events/{zone}` | Get events for a specific zone |
| `WS` | `/ws/alerts` | Live event stream for the dashboard |

Example event payload:
```json
{
  "type": "pothole",
  "severity": "HIGH",
  "zone": "Zone B",
  "confidence": 0.91,
  "evidence_path": "evidence/frame_00123.jpg",
  "recommended_action": "Road inspection",
  "timestamp": "2026-08-11T16:32:00"
}
```

---

## 📸 Demo / Screenshots

# 🌧️ AI Waterlogging + Road Hazard Detection System

An AI-powered system that uses drone/CCTV footage to automatically detect **waterlogging** and **potholes** on roads, score their severity, and push real-time alerts to a live dashboard for faster civic response.

Built for [Hackathon Name] — addresses the official challenge track on **waterlogging, potholes, drainage overflow, and road risks.**

---

## 🚨 The Problem

Cities rely on manual reporting or infrequent inspections to catch road hazards like waterlogging and potholes. By the time issues are reported, roads may already be unsafe — causing accidents, traffic jams, and vehicle damage. There's no automated, real-time way to detect and prioritize these hazards at scale.

## ✅ Our Solution

A drone/camera feed is processed by two AI detection models (waterlogging + pothole), which classify severity, tag the location/zone, capture evidence frames, and instantly alert a live dashboard — so city teams know exactly where to send an inspection crew, and how urgently.

---

## 🏗️ Architecture

```
Drone / CCTV feed
       │
       ▼
 Frame extraction (sampled frames + GPS tag)
       │
  ┌────┴─────┐
  ▼           ▼
Waterlogging  Pothole
detection     detection
  └────┬─────┘
       ▼
Severity + location scoring
       │
       ▼
Evidence capture (frame + timestamp)
       │
       ▼
Backend API (FastAPI) ──stores──▶ Database
       │
       └──broadcasts (WebSocket)──▶ Frontend dashboard
```

See `/docs/architecture.png` for the full diagram.

---

## ✨ Features

- 🎯 **Dual hazard detection** — waterlogging (segmentation) + potholes (object detection)
- 📊 **Automatic severity scoring** — LOW / MEDIUM / HIGH based on detected area/coverage
- 📍 **Zone-based location tagging** — maps GPS/pixel coordinates to city zones
- 📸 **Evidence capture** — every alert is backed by a timestamped frame
- ⚡ **Real-time dashboard** — live alert feed over WebSocket, no manual refresh
- 🗺️ **Map view** — see hazards plotted geographically
- ⚠️ **Recommended actions** — auto-suggested response (e.g. "Road inspection", "Monitor")

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| Detection models | YOLOv8 / Roboflow (pothole), segmentation model (waterlogging) |
| Backend | FastAPI, SQLAlchemy, WebSockets |
| Database | SQLite (dev) |
| Frontend | React, Tailwind CSS, react-leaflet |
| Model hosting | Roboflow serverless inference |

---

## 📁 Repo Structure

```
.
├── backend/
│   ├── main.py            # FastAPI app, API + WebSocket endpoints
│   ├── models.py          # DB schema
│   ├── database.py        # DB connection setup
│   ├── inference.py       # Calls detection model, builds hazard events
│   ├── requirements.txt
│   └── evidence/          # saved evidence frames
├── frontend/
│   ├── src/
│   │   ├── components/    # AlertCard, AlertFeed, ZoneMap
│   │   ├── hooks/         # useAlertSocket.js
│   │   ├── App.jsx
│   │   └── api.js
│   ├── package.json
│   └── ...
├── models/
│   └── training_notebooks/  # Colab/Jupyter notebooks used for training
├── docs/
│   └── architecture.png
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- A Roboflow account + API key (or local YOLO weights)

### 1. Clone the repo
```bash
git clone https://github.com/Yashu1507/pothole-and-waterlogging-detection-.git
cd <repo-name>
```

### 2. Backend setup
```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env       # fill in your API keys
uvicorn main:app --reload --port 8000
```
Backend runs at `http://localhost:8000` — API docs at `http://localhost:8000/docs`.

### 3. Frontend setup
```bash
cd frontend
npm install
npm run dev
```
Frontend runs at `http://localhost:5173` (or whatever Vite/CRA assigns).

### 4. Run detection
```bash
cd backend
python inference.py --source path/to/video.mp4
```
This will detect hazards frame-by-frame and post events to the backend, which the dashboard picks up live.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/events` | Submit a new hazard detection event |
| `GET` | `/events` | Get recent events (default limit 50) |
| `GET` | `/events/{zone}` | Get events for a specific zone |
| `WS` | `/ws/alerts` | Live event stream for the dashboard |

Example event payload:
```json
{
  "type": "pothole",
  "severity": "HIGH",
  "zone": "Zone B",
  "confidence": 0.91,
  "evidence_path": "evidence/frame_00123.jpg",
  "recommended_action": "Road inspection",
  "timestamp": "2026-08-11T16:32:00"
}
```

---

## 📸 Demo / Screenshots

<img width="1070" height="682" alt="WhatsApp Image 2026-08-11 at 10 35 53 PM" src="https://github.com/user-attachments/assets/d7743131-9b24-4e4f-b444-d789c6ea5cc3" />



---

## 🗺️ Roadmap

- [ ] Add depth-based severity estimation (stereo camera / LiDAR)
- [ ] SMS/WhatsApp alerts for high-severity zones
- [ ] Historical trend dashboard (which zones flood most often)
- [ ] Edge deployment on Jetson Nano for onboard drone inference

---

## 👥 Team

- Yashaswini.M.N - Dashboard/backend lead 
- Rohini.H.G - AI/ML lead 

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.


---

## 🗺️ Roadmap

- [ ] Add depth-based severity estimation (stereo camera / LiDAR)
- [ ] SMS/WhatsApp alerts for high-severity zones
- [ ] Historical trend dashboard (which zones flood most often)
- [ ] Edge deployment on Jetson Nano for onboard drone inference

---

## 👥 Team

- Yashaswini.M.N - Dashboard/backend lead 
- Rohini.H.G - AI/ML lead 

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
