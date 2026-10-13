
import cv2
import numpy as np
import lgpio
import time
import signal
import sys

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "/home/srinivas/MoodVision/model/facial_expression_mobilefacenet_int8.onnx"
CASCADE_PATH = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

LABELS = [
    "ANGRY", "DISGUST", "FEARFUL",
    "HAPPY", "NEUTRAL", "SAD", "SURPRISED"
]

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
INFERENCE_INTERVAL = 5
HISTORY_SIZE = 10

GPIO_PIN = 17
BUZZER_SECONDS = 15

# ============================================================
# GPIO SETUP
# ============================================================

chip = None
cap = None
cleaned_up = False

def buzzer_off():
    if chip is not None:
        try:
            lgpio.gpio_write(chip, GPIO_PIN, 0)
        except Exception:
            pass

def cleanup():
    global cleaned_up

    if cleaned_up:
        return

    cleaned_up = True

    # Turn buzzer OFF before releasing GPIO
    buzzer_off()

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if chip is not None:
        try:
            lgpio.gpio_free(chip, GPIO_PIN)
        except Exception:
            pass

        try:
            lgpio.gpiochip_close(chip)
        except Exception:
            pass

    print("Program stopped; GPIO17 commanded LOW.")

def stop_handler(signum, frame):
    cleanup()
    raise SystemExit(0)

try:
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(chip, GPIO_PIN, 0)
    buzzer_off()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    # ========================================================
    # LOAD MODELS
    # ========================================================

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if face_cascade.empty():
        raise RuntimeError("Could not load face detector XML.")

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open USB camera.")

    print("MoodVision started. Press Q to quit.")

    # ========================================================
    # STATE
    # ========================================================

    frame_count = 0
    prediction_history = []
    stable_label = "WAITING"

    buzzer_active = False
    buzzer_start = 0.0

    # Require several consecutive SAD results before triggering.
    sad_streak = 0
    SAD_CONFIRMATIONS = 3

    # Prevent retriggering until a non-SAD expression is stable.
    sad_event_locked = False
    non_sad_streak = 0
    NON_SAD_RESET_COUNT = 3

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame error.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            stable_label = "NO FACE"
            prediction_history.clear()
            sad_streak = 0
            non_sad_streak = 0

        else:
            x, y, w, h = max(
                faces,
                key=lambda r: r[2] * r[3]
            )

            margin_x = int(w * 0.15)
            margin_y = int(h * 0.20)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                crop_h, crop_w = face_crop.shape[:2]

                tight_face = face_crop[
                    int(crop_h * 0.10):int(crop_h * 0.90),
                    int(crop_w * 0.10):int(crop_w * 0.90)
                ]

                if tight_face.size > 0:
                    # 2x zoom
                    zoomed_face = cv2.resize(
                        tight_face,
                        (448, 448),
                        interpolation=cv2.INTER_CUBIC
                    )

                    # Second face scan
                    zoom_gray = cv2.cvtColor(
                        zoomed_face,
                        cv2.COLOR_BGR2GRAY
                    )

                    second_faces = face_cascade.detectMultiScale(
                        zoom_gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(80, 80)
                    )

                    refined_face = zoomed_face

                    if len(second_faces) > 0:
                        rx, ry, rw, rh = max(
                            second_faces,
                            key=lambda r: r[2] * r[3]
                        )

                        mx = int(rw * 0.08)
                        my = int(rh * 0.10)

                        rx1 = max(0, rx - mx)
                        ry1 = max(0, ry - my)
                        rx2 = min(448, rx + rw + mx)
                        ry2 = min(448, ry + rh + my)

                        candidate = zoomed_face[ry1:ry2, rx1:rx2]

                        if candidate.size > 0:
                            refined_face = candidate

                    if frame_count % INFERENCE_INTERVAL == 0:
                        model_face = cv2.resize(
                            refined_face,
                            (112, 112),
                            interpolation=cv2.INTER_AREA
                        )

                        model_face = (
                            model_face.astype(np.float32) / 255.0
                        )
                        model_face = (model_face - 0.5) / 0.5

                        blob = cv2.dnn.blobFromImage(
                            model_face,
                            scalefactor=1.0,
                            size=(112, 112),
                            swapRB=False,
                            crop=False
                        )

                        net.setInput(blob, "data")
                        scores = net.forward("label")[0]

                        prediction_history.append(scores.copy())

                        if len(prediction_history) > HISTORY_SIZE:
                            prediction_history.pop(0)

                        avg_scores = np.mean(
                            prediction_history,
                            axis=0
                        )

                        class_index = int(np.argmax(avg_scores))
                        predicted_label = LABELS[class_index]

                        print("Prediction:", predicted_label)

                        if predicted_label == "SAD":
                            sad_streak += 1
                            non_sad_streak = 0
                        else:
                            sad_streak = 0
                            non_sad_streak += 1

                        # Unlock only after stable non-SAD predictions.
                        if (
                            sad_event_locked
                            and non_sad_streak >= NON_SAD_RESET_COUNT
                        ):
                            sad_event_locked = False

                        # Trigger one buzzer event for a confirmed SAD event.
                        if (
                            sad_streak >= SAD_CONFIRMATIONS
                            and not sad_event_locked
                        ):
                            lgpio.gpio_write(chip, GPIO_PIN, 1)
                            buzzer_active = True
                            buzzer_start = time.monotonic()
                            sad_event_locked = True

                            print("SAD confirmed: buzzer ON for 15 seconds.")

                        stable_label = predicted_label

                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 1
                )

                cv2.putText(
                    frame,
                    stable_label,
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

        # Timer runs independently of model predictions.
        if buzzer_active:
            if time.monotonic() - buzzer_start >= BUZZER_SECONDS:
                buzzer_off()
                buzzer_active = False
                print("15 seconds elapsed: buzzer OFF.")

        display_frame = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_LINEAR)
         cv2.imshow("MoodVision - HAPPY / SAD", display_frame)
        frame_count += 1

        # Q closes the program.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Q pressed. Exiting.")
            break

except KeyboardInterrupt:
    print("Ctrl+C received.")

except Exception as exc:
    print("Program error:", exc)

finally:
    buzzer_off()

######################################################





#####################################################