import time
import threading
import collections

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

EAR_SECONDS_THRESHOLD = 2.0    
YAW_SECONDS_THRESHOLD = 3.0     
NO_FACE_SECONDS_THRESHOLD = 3.0 
CALIBRATION_SECONDS = 4
DEBOUNCE_WINDOW = 30

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

mp_face_mesh = mp.solutions.face_mesh

app = Flask(__name__)
app.config["SECRET_KEY"] = "distraction-detector-dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


state_lock = threading.Lock()
shared_state = {
    "status": "starting",       
    "reason": "",
    "focus_score": 100.0,
    "session_log": [],          
}
session_start_time = None
detection_thread_started = False


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
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    left = np.array([landmarks[LEFT_CHEEK].x * w, landmarks[LEFT_CHEEK].y * h])
    right = np.array([landmarks[RIGHT_CHEEK].x * w, landmarks[RIGHT_CHEEK].y * h])
    mid = (left + right) / 2.0
    face_width = euclidean(left, right)
    if face_width == 0:
        return 0.0
    return (nose[0] - mid[0]) / face_width


def collect_samples(cap, face_mesh, seconds):
    """Silent version of Phase 1's calibration sampler (no cv2.imshow,
    since this runs headless inside the Flask background thread)."""
    ears, yaws = [], []
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
    return ears, yaws


def run_calibration(cap, face_mesh):
    socketio.emit("calibration_step", {"step": "straight", "seconds": CALIBRATION_SECONDS})
    straight_ears, straight_yaws = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    socketio.emit("calibration_step", {"step": "away", "seconds": CALIBRATION_SECONDS})
    _, away_yaws = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    socketio.emit("calibration_step", {"step": "closed", "seconds": CALIBRATION_SECONDS})
    closed_ears, _ = collect_samples(cap, face_mesh, CALIBRATION_SECONDS)

    baseline_ear = float(np.mean(straight_ears)) if straight_ears else 0.28
    closed_ear = float(np.mean(closed_ears)) if closed_ears else baseline_ear * 0.6
    ear_threshold = (baseline_ear + closed_ear) / 2.0

    baseline_yaw = float(np.mean(np.abs(straight_yaws))) if straight_yaws else 0.05
    away_yaw = float(np.mean(np.abs(away_yaws))) if away_yaws else 0.25
    yaw_threshold = (baseline_yaw + away_yaw) / 2.0

    return ear_threshold, yaw_threshold


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
        ear_threshold, yaw_threshold = run_calibration(cap, face_mesh)

        session_start_time = time.time()
        with state_lock:
            shared_state["status"] = "focused"
            shared_state["session_log"] = []
        socketio.emit("status_change", {"status": "focused", "reason": ""})

        eye_closed_since = None 
        yaw_away_since = None    
        no_face_since = None      
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

                if eye_closed_since is not None and now - eye_closed_since >= EAR_SECONDS_THRESHOLD:
                    is_focused = False
                    reason = "eyes closed / looking down"
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
            if current_status != last_status:
                socketio.emit("status_change", {"status": current_status, "reason": reason})
                last_status = current_status

            socketio.emit("focus_update", {"focus_score": round(focus_score, 1)})

            socketio.sleep(0.05) 
@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    global detection_thread_started
    if not detection_thread_started:
        detection_thread_started = True
        socketio.start_background_task(detection_loop)


@socketio.on("request_report")
def handle_report_request():
    """Called when the video ends — sends the full session log so the
    frontend can render the Focus Score + Distraction Timeline chart."""
    with state_lock:
        log = list(shared_state["session_log"])
        score = shared_state["focus_score"]
    socketio.emit("session_report", {"log": log, "final_score": round(score, 1)})


if __name__ == "__main__":
    print("Starting Flask-SocketIO server on http://127.0.0.1:5000 ...", flush=True)
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, use_reloader=False)