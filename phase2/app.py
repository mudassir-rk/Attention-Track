# import time
# import threading
# import collections
# import sqlite3
# import json
# import os
# from datetime import datetime

# import cv2
# import mediapipe as mp
# import numpy as np
# from flask import Flask, render_template, jsonify
# from flask_socketio import SocketIO

# EAR_SECONDS_THRESHOLD = 2.0      # eyes closed
# YAW_SECONDS_THRESHOLD = 3.0      # head turned sideways
# PITCH_SECONDS_THRESHOLD = 1.0     # head tilted down (phone in lap) — kept short since this should be sensitive
# NO_FACE_SECONDS_THRESHOLD = 3.0  # face missing entirely
# CALIBRATION_SECONDS = 4
# DEBOUNCE_WINDOW = 30

# # MediaPipe Face Mesh landmark indices used for each measurement
# LEFT_EYE = [33, 160, 158, 133, 153, 144]
# RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# NOSE_TIP = 1
# LEFT_CHEEK = 234
# RIGHT_CHEEK = 454
# FOREHEAD = 10   # top of forehead, used as the "up" reference point for pitch
# CHIN = 152      # bottom of chin, used as the "down" reference point for pitch

# mp_face_mesh = mp.solutions.face_mesh

# DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.db")

# app = Flask(__name__)
# app.config["SECRET_KEY"] = "distraction-detector-dev"
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# # Shared state between the detection thread and the Flask routes
# state_lock = threading.Lock()
# shared_state = {
#     "status": "starting",
#     "reason": "",
#     "focus_score": 100.0,
#     "session_log": [],
# }
# session_start_time = None
# detection_thread_started = False


# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS sessions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             start_time TEXT NOT NULL,
#             end_time TEXT NOT NULL,
#             duration_seconds REAL NOT NULL,
#             final_score REAL NOT NULL,
#             session_log TEXT NOT NULL
#         )
#     """)
#     conn.commit()
#     conn.close()


# def save_session(start_iso, end_iso, duration_seconds, score, log):
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         "INSERT INTO sessions (start_time, end_time, duration_seconds, final_score, session_log) VALUES (?, ?, ?, ?, ?)",
#         (start_iso, end_iso, duration_seconds, score, json.dumps(log)),
#     )
#     conn.commit()
#     conn.close()


# def get_all_sessions():
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     rows = conn.execute("SELECT * FROM sessions ORDER BY end_time DESC").fetchall()
#     conn.close()
#     return [dict(row) for row in rows]


# def get_session(session_id):
#     conn = sqlite3.connect(DB_PATH)
#     conn.row_factory = sqlite3.Row
#     row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
#     conn.close()
#     return dict(row) if row else None


# def euclidean(p1, p2):
#     return np.linalg.norm(np.array(p1) - np.array(p2))


# def eye_aspect_ratio(landmarks, eye_indices, w, h):
#     pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
#     p1, p2, p3, p4, p5, p6 = pts
#     vertical_1 = euclidean(p2, p6)
#     vertical_2 = euclidean(p3, p5)
#     horizontal = euclidean(p1, p4)
#     if horizontal == 0:
#         return 0.3
#     return (vertical_1 + vertical_2) / (2.0 * horizontal)


# def yaw_offset(landmarks, w, h):
#     """Side-to-side head turn estimate: nose tip position relative to the
#     midpoint of the two cheeks, normalized by face width."""
#     nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
#     left = np.array([landmarks[LEFT_CHEEK].x * w, landmarks[LEFT_CHEEK].y * h])
#     right = np.array([landmarks[RIGHT_CHEEK].x * w, landmarks[RIGHT_CHEEK].y * h])
#     mid = (left + right) / 2.0
#     face_width = euclidean(left, right)
#     if face_width == 0:
#         return 0.0
#     return (nose[0] - mid[0]) / face_width


# def pitch_offset(landmarks, w, h):
#     """Up-down head tilt estimate (PHASE 3): nose tip's vertical position
#     relative to the midpoint between forehead and chin, normalized by
#     face height. When the head tilts DOWN (looking at a phone in the
#     lap), the nose tip moves further below this midpoint.
#     This is a 2D proxy, same style/simplicity as yaw_offset — not full 3D
#     pose estimation, kept explainable and lightweight on purpose."""
#     nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
#     top = np.array([landmarks[FOREHEAD].x * w, landmarks[FOREHEAD].y * h])
#     bottom = np.array([landmarks[CHIN].x * w, landmarks[CHIN].y * h])
#     mid = (top + bottom) / 2.0
#     face_height = euclidean(top, bottom)
#     if face_height == 0:
#         return 0.0
#     return (nose[1] - mid[1]) / face_height


# def collect_samples(cap, face_mesh, seconds):
#     """Silent version of Phase 1's calibration sampler (no cv2.imshow,
#     since this runs headless inside the Flask background thread).
#     Collects EAR, yaw, AND pitch samples on every call — the caller just
#     uses whichever list is relevant for that calibration step."""
#     ears, yaws, pitches = [], [], []
#     start = time.time()
#     while time.time() - start < seconds:
#         ok, frame = cap.read()
#         if not ok:
#             continue
#         frame = cv2.flip(frame, 1)
#         h, w = frame.shape[:2]
#         results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         if results.multi_face_landmarks:
#             lm = results.multi_face_landmarks[0].landmark
#             left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
#             right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
#             ears.append((left_ear + right_ear) / 2.0)
#             yaws.append(yaw_offset(lm, w, h))
#             pitches.append(pitch_offset(lm, w, h))
#     return ears, yaws, pitches


# def run_calibration(cap, face_mesh):
#     """Four-step calibration: straight, away (yaw), closed (EAR), and now
#     down (pitch) — each step derives a personalized threshold instead of
#     relying on generic hardcoded values."""
#     socketio.emit("calibration_step", {"step": "straight", "seconds": CALIBRATION_SECONDS})
#     straight_ears, straight_yaws, straight_pitches = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

#     socketio.emit("calibration_step", {"step": "away", "seconds": CALIBRATION_SECONDS})
#     _, away_yaws, _ = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

#     socketio.emit("calibration_step", {"step": "closed", "seconds": CALIBRATION_SECONDS})
#     closed_ears, _, _ = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

#     socketio.emit("calibration_step", {"step": "down", "seconds": CALIBRATION_SECONDS})
#     _, _, down_pitches = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

#     baseline_ear = float(np.mean(straight_ears)) if straight_ears else 0.28
#     closed_ear = float(np.mean(closed_ears)) if closed_ears else baseline_ear * 0.6
#     ear_threshold = (baseline_ear + closed_ear) / 2.0

#     baseline_yaw = float(np.mean(np.abs(straight_yaws))) if straight_yaws else 0.05
#     away_yaw = float(np.mean(np.abs(away_yaws))) if away_yaws else 0.25
#     yaw_threshold = (baseline_yaw + away_yaw) / 2.0

#     baseline_pitch = float(np.mean(straight_pitches)) if straight_pitches else 0.0
#     down_pitch = float(np.mean(down_pitches)) if down_pitches else baseline_pitch + 0.15
#     # Weighted 30% of the way from baseline toward the "down" sample
#     # (instead of the 50/50 midpoint used for eyes/yaw) so a SMALLER head
#     # tilt is enough to trigger this — pitch is intentionally the most
#     # sensitive of the three checks, since a quick glance down at a phone
#     # should still count as distraction.
#     pitch_threshold = baseline_pitch + 0.3 * (down_pitch - baseline_pitch)
#     pitch_direction = 1 if down_pitch >= baseline_pitch else -1

#     return ear_threshold, yaw_threshold, pitch_threshold, pitch_direction


# def detection_loop():
#     """Runs forever in a background thread. Reads webcam, runs face mesh,
#     and emits a WebSocket event ONLY when the focus status changes
#     (not every frame) — that's what the browser reacts to."""
#     global session_start_time

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         socketio.emit("error", {"message": "Could not open webcam."})
#         return

#     with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#     ) as face_mesh:

#         with state_lock:
#             shared_state["status"] = "calibrating"
#         ear_threshold, yaw_threshold, pitch_threshold, pitch_direction = run_calibration(cap, face_mesh)

#         session_start_time = time.time()
#         with state_lock:
#             shared_state["status"] = "focused"
#             shared_state["session_log"] = []
#         socketio.emit("status_change", {"status": "focused", "reason": ""})

#         eye_closed_since = None    # timestamp when eyes FIRST became closed, or None
#         yaw_away_since = None      # timestamp when head FIRST turned sideways, or None
#         pitch_down_since = None    # timestamp when head FIRST tilted down, or None
#         no_face_since = None       # timestamp when face FIRST disappeared, or None
#         focus_history = collections.deque(maxlen=DEBOUNCE_WINDOW)
#         last_status = "focused"

#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 continue
#             frame = cv2.flip(frame, 1)
#             h, w = frame.shape[:2]
#             results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#             now = time.time()
#             is_focused = True
#             reason = ""

#             if not results.multi_face_landmarks:
#                 if no_face_since is None:
#                     no_face_since = now
#                 eye_closed_since = None
#                 yaw_away_since = None
#                 pitch_down_since = None
#                 if now - no_face_since >= NO_FACE_SECONDS_THRESHOLD:
#                     is_focused = False
#                     reason = "face not visible"
#             else:
#                 no_face_since = None
#                 lm = results.multi_face_landmarks[0].landmark

#                 left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
#                 right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
#                 avg_ear = (left_ear + right_ear) / 2.0
#                 yaw = yaw_offset(lm, w, h)
#                 pitch = pitch_offset(lm, w, h)

#                 # Time-based tracking: record WHEN the condition first
#                 # started, then check elapsed real seconds — this works
#                 # consistently regardless of how fast this specific
#                 # machine processes frames.
#                 if avg_ear < ear_threshold:
#                     if eye_closed_since is None:
#                         eye_closed_since = now
#                 else:
#                     eye_closed_since = None

#                 if abs(yaw) > yaw_threshold:
#                     if yaw_away_since is None:
#                         yaw_away_since = now
#                 else:
#                     yaw_away_since = None

#                 if pitch_direction * pitch > pitch_direction * pitch_threshold:
#                     if pitch_down_since is None:
#                         pitch_down_since = now
#                 else:
#                     pitch_down_since = None

#                 # Priority order: pitch (looking down) is checked BEFORE
#                 # eyes-closed. This matters because tilting your head down
#                 # naturally makes your eyes look more "closed" to the
#                 # webcam too (eyelid occlusion from that angle) — without
#                 # this order, a genuine "looking down" would incorrectly
#                 # get relabeled "eyes closed" once both conditions overlap.
#                 if pitch_down_since is not None and now - pitch_down_since >= PITCH_SECONDS_THRESHOLD:
#                     is_focused = False
#                     reason = "looking down (phone?)"
#                 elif eye_closed_since is not None and now - eye_closed_since >= EAR_SECONDS_THRESHOLD:
#                     is_focused = False
#                     reason = "eyes closed"
#                 elif yaw_away_since is not None and now - yaw_away_since >= YAW_SECONDS_THRESHOLD:
#                     is_focused = False
#                     reason = "looking away"

#             focus_history.append(1 if is_focused else 0)
#             focus_score = 100.0 * sum(focus_history) / len(focus_history)
#             current_status = "focused" if is_focused else "distracted"
#             elapsed = now - session_start_time

#             with state_lock:
#                 shared_state["status"] = current_status
#                 shared_state["reason"] = reason
#                 shared_state["focus_score"] = focus_score
#                 shared_state["session_log"].append((round(elapsed, 1), is_focused))

#             # Only emit on CHANGE, not every frame — this is what keeps
#             # the WebSocket traffic light and the video reaction crisp.
#             if current_status != last_status:
#                 socketio.emit("status_change", {"status": current_status, "reason": reason})
#                 last_status = current_status

#             # Lightweight periodic update for the live focus-score readout
#             socketio.emit("focus_update", {"focus_score": round(focus_score, 1)})

#             socketio.sleep(0.05)  # ~20 checks/sec is plenty for this use case


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/history")
# def history():
#     sessions = get_all_sessions()
#     return render_template("history.html", sessions=sessions)


# @app.route("/api/session/<int:session_id>")
# def api_session(session_id):
#     session = get_session(session_id)
#     if session is None:
#         return jsonify({"error": "Session not found"}), 404
#     return jsonify({
#         "log": json.loads(session["session_log"]),
#         "final_score": session["final_score"],
#         "start_time": session["start_time"],
#         "duration": session["duration_seconds"],
#     })


# @socketio.on("connect")
# def handle_connect():
#     global detection_thread_started
#     if not detection_thread_started:
#         detection_thread_started = True
#         socketio.start_background_task(detection_loop)


# @socketio.on("request_report")
# def handle_report_request():
#     """Called when the video ends — saves the session to SQLite, then sends
#     the full session log so the frontend can render the Focus Score +
#     Distraction Timeline chart."""
#     with state_lock:
#         log = list(shared_state["session_log"])
#         score = shared_state["focus_score"]

#     if session_start_time is not None:
#         end_time = time.time()
#         start_dt = datetime.fromtimestamp(session_start_time).isoformat()
#         end_dt = datetime.fromtimestamp(end_time).isoformat()
#         duration = end_time - session_start_time
#         save_session(start_dt, end_dt, duration, score, log)

#     socketio.emit("session_report", {"log": log, "final_score": round(score, 1)})


# if __name__ == "__main__":
#     init_db()
#     print("Starting Flask-SocketIO server on http://127.0.0.1:5000 ...", flush=True)
#     socketio.run(app, host="127.0.0.1", port=5000, debug=False, use_reloader=False)

import time
import threading
import collections
import sqlite3
import json
import os
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO

EAR_SECONDS_THRESHOLD = 2.0      # eyes closed
YAW_SECONDS_THRESHOLD = 3.0      # head turned sideways
PITCH_SECONDS_THRESHOLD = 1.0     # head tilted down (phone in lap) — kept short since this should be sensitive
NO_FACE_SECONDS_THRESHOLD = 3.0  # face missing entirely
CALIBRATION_SECONDS = 4
DEBOUNCE_WINDOW = 30

# MediaPipe Face Mesh landmark indices used for each measurement
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10   # top of forehead, used as the "up" reference point for pitch
CHIN = 152      # bottom of chin, used as the "down" reference point for pitch

mp_face_mesh = mp.solutions.face_mesh

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = "distraction-detector-dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Shared state between the detection thread and the Flask routes
state_lock = threading.Lock()
shared_state = {
    "status": "starting",
    "reason": "",
    "focus_score": 100.0,
    "session_log": [],
}
session_start_time = None
detection_thread_started = False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            final_score REAL NOT NULL,
            session_log TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_session(start_iso, end_iso, duration_seconds, score, log):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (start_time, end_time, duration_seconds, final_score, session_log) VALUES (?, ?, ?, ?, ?)",
        (start_iso, end_iso, duration_seconds, score, json.dumps(log)),
    )
    conn.commit()
    conn.close()


def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM sessions ORDER BY end_time DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_session(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.3
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def yaw_offset(landmarks, w, h):
    """Side-to-side head turn estimate: nose tip position relative to the
    midpoint of the two cheeks, normalized by face width."""
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    left = np.array([landmarks[LEFT_CHEEK].x * w, landmarks[LEFT_CHEEK].y * h])
    right = np.array([landmarks[RIGHT_CHEEK].x * w, landmarks[RIGHT_CHEEK].y * h])
    mid = (left + right) / 2.0
    face_width = euclidean(left, right)
    if face_width == 0:
        return 0.0
    return (nose[0] - mid[0]) / face_width


def pitch_offset(landmarks, w, h):
    """Up-down head tilt estimate (PHASE 3): nose tip's vertical position
    relative to the midpoint between forehead and chin, normalized by
    face height. When the head tilts DOWN (looking at a phone in the
    lap), the nose tip moves further below this midpoint.
    This is a 2D proxy, same style/simplicity as yaw_offset — not full 3D
    pose estimation, kept explainable and lightweight on purpose."""
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    top = np.array([landmarks[FOREHEAD].x * w, landmarks[FOREHEAD].y * h])
    bottom = np.array([landmarks[CHIN].x * w, landmarks[CHIN].y * h])
    mid = (top + bottom) / 2.0
    face_height = euclidean(top, bottom)
    if face_height == 0:
        return 0.0
    return (nose[1] - mid[1]) / face_height


def collect_samples(cap, face_mesh, seconds):
    """Silent version of Phase 1's calibration sampler (no cv2.imshow,
    since this runs headless inside the Flask background thread).
    Collects EAR, yaw, AND pitch samples on every call — the caller just
    uses whichever list is relevant for that calibration step."""
    ears, yaws, pitches = [], [], []
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            ears.append((left_ear + right_ear) / 2.0)
            yaws.append(yaw_offset(lm, w, h))
            pitches.append(pitch_offset(lm, w, h))
    return ears, yaws, pitches


def run_calibration(cap, face_mesh):
    """Four-step calibration: straight, away (yaw), closed (EAR), and now
    down (pitch) — each step derives a personalized threshold instead of
    relying on generic hardcoded values."""
    socketio.emit("calibration_step", {"step": "straight", "seconds": CALIBRATION_SECONDS})
    straight_ears, straight_yaws, straight_pitches = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    socketio.emit("calibration_step", {"step": "away", "seconds": CALIBRATION_SECONDS})
    _, away_yaws, _ = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    socketio.emit("calibration_step", {"step": "closed", "seconds": CALIBRATION_SECONDS})
    closed_ears, _, _ = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    socketio.emit("calibration_step", {"step": "down", "seconds": CALIBRATION_SECONDS})
    _, _, down_pitches = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    baseline_ear = float(np.mean(straight_ears)) if straight_ears else 0.28
    closed_ear = float(np.mean(closed_ears)) if closed_ears else baseline_ear * 0.6
    ear_threshold = (baseline_ear + closed_ear) / 2.0

    baseline_yaw = float(np.mean(np.abs(straight_yaws))) if straight_yaws else 0.05
    away_yaw = float(np.mean(np.abs(away_yaws))) if away_yaws else 0.25
    yaw_threshold = (baseline_yaw + away_yaw) / 2.0

    baseline_pitch = float(np.mean(straight_pitches)) if straight_pitches else 0.0
    down_pitch = float(np.mean(down_pitches)) if down_pitches else baseline_pitch + 0.15
    # Weighted 30% of the way from baseline toward the "down" sample
    # (instead of the 50/50 midpoint used for eyes/yaw) so a SMALLER head
    # tilt is enough to trigger this — pitch is intentionally the most
    # sensitive of the three checks, since a quick glance down at a phone
    # should still count as distraction.
    pitch_threshold = baseline_pitch + 0.3 * (down_pitch - baseline_pitch)
    pitch_direction = 1 if down_pitch >= baseline_pitch else -1

    return ear_threshold, yaw_threshold, pitch_threshold, pitch_direction


def detection_loop():
    """Runs forever in a background thread. Reads webcam, runs face mesh,
    and emits a WebSocket event ONLY when the focus status changes
    (not every frame) — that's what the browser reacts to."""
    global session_start_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        socketio.emit("error", {"message": "Could not open webcam."})
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        with state_lock:
            shared_state["status"] = "calibrating"
        ear_threshold, yaw_threshold, pitch_threshold, pitch_direction = run_calibration(cap, face_mesh)

        session_start_time = time.time()
        with state_lock:
            shared_state["status"] = "focused"
            shared_state["session_log"] = []
        socketio.emit("status_change", {"status": "focused", "reason": ""})

        eye_closed_since = None    # timestamp when eyes FIRST became closed, or None
        yaw_away_since = None      # timestamp when head FIRST turned sideways, or None
        pitch_down_since = None    # timestamp when head FIRST tilted down, or None
        no_face_since = None       # timestamp when face FIRST disappeared, or None
        focus_history = collections.deque(maxlen=DEBOUNCE_WINDOW)
        last_status = "focused"

        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            now = time.time()
            is_focused = True
            reason = ""

            if not results.multi_face_landmarks:
                if no_face_since is None:
                    no_face_since = now
                eye_closed_since = None
                yaw_away_since = None
                pitch_down_since = None
                if now - no_face_since >= NO_FACE_SECONDS_THRESHOLD:
                    is_focused = False
                    reason = "face not visible"
            else:
                no_face_since = None
                lm = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                yaw = yaw_offset(lm, w, h)
                pitch = pitch_offset(lm, w, h)

                # Time-based tracking: record WHEN the condition first
                # started, then check elapsed real seconds — this works
                # consistently regardless of how fast this specific
                # machine processes frames.
                if avg_ear < ear_threshold:
                    if eye_closed_since is None:
                        eye_closed_since = now
                else:
                    eye_closed_since = None

                if abs(yaw) > yaw_threshold:
                    if yaw_away_since is None:
                        yaw_away_since = now
                else:
                    yaw_away_since = None

                if pitch_direction * pitch > pitch_direction * pitch_threshold:
                    if pitch_down_since is None:
                        pitch_down_since = now
                else:
                    pitch_down_since = None

                # Priority order: pitch (looking down) is checked BEFORE
                # eyes-closed. This matters because tilting your head down
                # naturally makes your eyes look more "closed" to the
                # webcam too (eyelid occlusion from that angle) — without
                # this order, a genuine "looking down" would incorrectly
                # get relabeled "eyes closed" once both conditions overlap.
                if pitch_down_since is not None and now - pitch_down_since >= PITCH_SECONDS_THRESHOLD:
                    is_focused = False
                    reason = "looking down (phone?)"
                elif eye_closed_since is not None and now - eye_closed_since >= EAR_SECONDS_THRESHOLD:
                    is_focused = False
                    reason = "eyes closed"
                elif yaw_away_since is not None and now - yaw_away_since >= YAW_SECONDS_THRESHOLD:
                    is_focused = False
                    reason = "looking away"

            focus_history.append(1 if is_focused else 0)
            focus_score = 100.0 * sum(focus_history) / len(focus_history)
            current_status = "focused" if is_focused else "distracted"
            elapsed = now - session_start_time

            with state_lock:
                shared_state["status"] = current_status
                shared_state["reason"] = reason
                shared_state["focus_score"] = focus_score
                shared_state["session_log"].append((round(elapsed, 1), is_focused))

            # Only emit on CHANGE, not every frame — this is what keeps
            # the WebSocket traffic light and the video reaction crisp.
            if current_status != last_status:
                socketio.emit("status_change", {"status": current_status, "reason": reason})
                last_status = current_status

            # Lightweight periodic update for the live focus-score readout
            socketio.emit("focus_update", {"focus_score": round(focus_score, 1)})

            socketio.sleep(0.05)  # ~20 checks/sec is plenty for this use case


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history")
def history():
    sessions = get_all_sessions()
    return render_template("history.html", sessions=sessions)


@app.route("/api/session/<int:session_id>")
def api_session(session_id):
    session = get_session(session_id)
    if session is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({
        "log": json.loads(session["session_log"]),
        "final_score": session["final_score"],
        "start_time": session["start_time"],
        "duration": session["duration_seconds"],
    })


@socketio.on("connect")
def handle_connect():
    global detection_thread_started
    if not detection_thread_started:
        detection_thread_started = True
        socketio.start_background_task(detection_loop)


@socketio.on("request_report")
def handle_report_request():
    """Called when the video ends — saves the session to SQLite, then sends
    the full session log so the frontend can render the Focus Score +
    Distraction Timeline chart."""
    with state_lock:
        log = list(shared_state["session_log"])
        score = shared_state["focus_score"]

    if session_start_time is not None:
        end_time = time.time()
        start_dt = datetime.fromtimestamp(session_start_time).isoformat()
        end_dt = datetime.fromtimestamp(end_time).isoformat()
        duration = end_time - session_start_time
        save_session(start_dt, end_dt, duration, score, log)

    socketio.emit("session_report", {"log": log, "final_score": round(score, 1)})


if __name__ == "__main__":
    init_db()
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000 ...", flush=True)
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, use_reloader=False)