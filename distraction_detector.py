import time
import collections
import cv2
import mediapipe as mp
import numpy as np

EAR_CONSEC_FRAMES = 15        
YAW_CONSEC_FRAMES = 20        
NO_FACE_CONSEC_FRAMES = 20    
CALIBRATION_SECONDS = 4      
DEBOUNCE_WINDOW = 30        

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

mp_face_mesh = mp.solutions.face_mesh


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Classic EAR formula: vertical eye distances / horizontal eye distance.
    Lower value = eye more closed."""
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.3  # neutral fallback, avoids divide-by-zero
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def yaw_offset(landmarks, w, h):
    """Very simple proxy for head yaw: how far the nose tip sits from the
    midpoint of the two cheeks, normalized by face width.
    ~0 = facing camera. Larger magnitude = turned to one side.
    This is NOT full 3D pose estimation (that would use solvePnP with a
    3D face model + camera matrix) — it's a lightweight beginner-friendly
    approximation that's good enough for "looking away" detection."""
    nose = np.array([landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h])
    left = np.array([landmarks[LEFT_CHEEK].x * w, landmarks[LEFT_CHEEK].y * h])
    right = np.array([landmarks[RIGHT_CHEEK].x * w, landmarks[RIGHT_CHEEK].y * h])
    mid = (left + right) / 2.0
    face_width = euclidean(left, right)
    if face_width == 0:
        return 0.0
    return (nose[0] - mid[0]) / face_width  


def run_calibration(cap, face_mesh):
    """Ask the user to look straight, then look away, to derive personalized
    thresholds instead of hardcoding generic tutorial values."""
    print("\n--- CALIBRATION ---")
    print(f"Step 1: Look STRAIGHT at the screen for {CALIBRATION_SECONDS} seconds...")
    straight_ears, straight_yaws = collect_samples(cap, face_mesh, CALIBRATION_SECONDS, "Look straight")

    print(f"Step 2: Turn your head AWAY (either side) for {CALIBRATION_SECONDS} seconds...")
    _, away_yaws = collect_samples(cap, face_mesh, CALIBRATION_SECONDS, "Look away")

    print(f"Step 3: CLOSE your eyes gently for {CALIBRATION_SECONDS} seconds...")
    closed_ears, _ = collect_samples(cap, face_mesh, CALIBRATION_SECONDS, "Close eyes")

    baseline_ear = float(np.mean(straight_ears)) if straight_ears else 0.28
    closed_ear = float(np.mean(closed_ears)) if closed_ears else baseline_ear * 0.6
    ear_threshold = (baseline_ear + closed_ear) / 2.0

    baseline_yaw = float(np.mean(np.abs(straight_yaws))) if straight_yaws else 0.05
    away_yaw = float(np.mean(np.abs(away_yaws))) if away_yaws else 0.25
    yaw_threshold = (baseline_yaw + away_yaw) / 2.0

    print("--- CALIBRATION COMPLETE ---")
    print(f"EAR threshold set to:  {ear_threshold:.3f}")
    print(f"Yaw threshold set to:  {yaw_threshold:.3f}\n")
    return ear_threshold, yaw_threshold


def collect_samples(cap, face_mesh, seconds, label):
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
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return ears, yaws

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        ear_threshold, yaw_threshold = run_calibration(cap, face_mesh)
        cv2.destroyWindow("Calibration")

        eye_closed_counter = 0
        yaw_away_counter = 0
        no_face_counter = 0
        focus_history = collections.deque(maxlen=DEBOUNCE_WINDOW)

        print("Starting live detection. Press 'q' to quit.\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            status = "FOCUSED"
            is_focused = True

            if not results.multi_face_landmarks:
                no_face_counter += 1
                eye_closed_counter = 0
                yaw_away_counter = 0
                if no_face_counter >= NO_FACE_CONSEC_FRAMES:
                    status = "DISTRACTED: face not visible"
                    is_focused = False
            else:
                no_face_counter = 0
                lm = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
                right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                yaw = yaw_offset(lm, w, h)

                if avg_ear < ear_threshold:
                    eye_closed_counter += 1
                else:
                    eye_closed_counter = 0

                if abs(yaw) > yaw_threshold:
                    yaw_away_counter += 1
                else:
                    yaw_away_counter = 0

                if eye_closed_counter >= EAR_CONSEC_FRAMES:
                    status = "DISTRACTED: eyes closed / looking down"
                    is_focused = False
                elif yaw_away_counter >= YAW_CONSEC_FRAMES:
                    status = "DISTRACTED: looking away"
                    is_focused = False

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            focus_history.append(1 if is_focused else 0)
            focus_score = 100.0 * sum(focus_history) / len(focus_history)

            color = (0, 200, 0) if is_focused else (0, 0, 255)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
            cv2.putText(frame, f"Focus score: {focus_score:.0f}%", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Distraction Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()