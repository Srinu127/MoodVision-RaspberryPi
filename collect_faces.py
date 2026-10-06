import cv2
import os
import time

# ============================================================
# SETTINGS
# ============================================================

CAMERA_ID = 0
WIDTH = 640
HEIGHT = 480
FACE_SIZE = 128

BASE_DIR = "/home/srinivas/MoodVision/dataset"

CASCADE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"


# ============================================================
# SELECT EXPRESSION
# ============================================================

print()
print("========================================")
print(" CUSTOM EMOTION DATASET COLLECTION")
print("========================================")
print()
print("1 = NEUTRAL")
print("2 = HAPPY")
print("3 = SAD")
print("4 = ANGRY")
print("5 = SURPRISE")
print()

choice = input("Select expression (1-5): ").strip()

labels = {
    "1": "NEUTRAL",
    "2": "HAPPY",
    "3": "SAD",
    "4": "ANGRY",
    "5": "SURPRISE"
}

if choice not in labels:
    print("Invalid selection")
    exit()

expression = labels[choice]

SAVE_DIR = os.path.join(BASE_DIR, expression)
os.makedirs(SAVE_DIR, exist_ok=True)

print()
print("Selected:", expression)
print("Saving to:", SAVE_DIR)
print()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(CASCADE)

if face_detector.empty():
    print("ERROR: Haar cascade could not be loaded")
    exit()


# ============================================================
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_ID)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()


# ============================================================
# VARIABLES
# ============================================================

image_count = len(os.listdir(SAVE_DIR))
last_save_time = 0
SAVE_INTERVAL = 0.4

print()
print("Camera started.")
print()
print("Position your face inside the camera.")
print()
print("Press SPACE to start capturing.")
print("Press Q to quit.")
print()

capturing = False


# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera frame error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    if len(faces) > 0:

        x, y, w, h = max(
            faces,
            key=lambda r: r[2] * r[3]
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # CAPTURE FACE
        # ----------------------------------------------------

        current_time = time.time()

        if (
            capturing
            and current_time - last_save_time >= SAVE_INTERVAL
        ):

            face = frame[y:y + h, x:x + w]

            face = cv2.resize(
                face,
                (FACE_SIZE, FACE_SIZE),
                interpolation=cv2.INTER_AREA
            )

            filename = os.path.join(
                SAVE_DIR,
                f"{expression}_{image_count:04d}.jpg"
            )

            cv2.imwrite(filename, face)

            image_count += 1
            last_save_time = current_time

            print(f"Captured: {image_count}")

    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (420, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Expression: {expression}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Images: {image_count}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    if capturing:

        cv2.putText(
            frame,
            "CAPTURING",
            (280, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Press SPACE",
            (280, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "Emotion Dataset Collection",
        frame
    )

    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):

        capturing = not capturing

        if capturing:
            print(">>> CAPTURE STARTED")
        else:
            print(">>> CAPTURE PAUSED")

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print()
print("Collection stopped.")
print("Expression:", expression)
print("Total images:", image_count)
