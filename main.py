import cv2
import numpy as np
from collections import deque
from sklearn.cluster import KMeans
from ultralytics import YOLO

INPUT_VIDEO = "match.mp4"
OUTPUT_VIDEO = "output.mp4"
MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.35
SAMPLE_INTERVAL = 5
MAX_SAMPLES = 400
MIN_SAMPLES = 30

REFEREE_HSV = np.array([30, 220, 220], dtype=np.float32)
REFEREE_THRESHOLD = 35.0

TEAM_COLORS = [
    (255, 100, 0),
    (0, 0, 255)
]

REFEREE_COLOR = (0, 255, 255)
UNKNOWN_COLOR = (180, 180, 180)

model = YOLO(MODEL_NAME)


def extract_shirt_color(frame, box):
    x1, y1, x2, y2 = map(int, box)

    frame_height, frame_width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    person = frame[y1:y2, x1:x2]

    height, width = person.shape[:2]

    if height < 20 or width < 10:
        return None

    shirt = person[
        int(height * 0.15):int(height * 0.50),
        int(width * 0.25):int(width * 0.75)
    ]

    if shirt.size == 0:
        return None

    hsv = cv2.cvtColor(shirt, cv2.COLOR_BGR2HSV)

    pixels = hsv.reshape(-1, 3).astype(np.float32)

    mask = (
        (pixels[:, 1] > 40) &
        (pixels[:, 2] > 40)
    )

    selected = pixels[mask]

    if len(selected) < 10:
        selected = pixels

    if len(selected) == 0:
        return None

    color = np.median(selected, axis=0)

    return color


def color_distance(color_a, color_b):
    hue_difference = abs(
        float(color_a[0]) - float(color_b[0])
    )

    hue_difference = min(
        hue_difference,
        180.0 - hue_difference
    )

    saturation_difference = (
        float(color_a[1]) - float(color_b[1])
    )

    value_difference = (
        float(color_a[2]) - float(color_b[2])
    )

    return float(np.sqrt(
        (2.0 * hue_difference) ** 2 +
        (0.15 * saturation_difference) ** 2 +
        (0.10 * value_difference) ** 2
    ))


def is_referee(color):
    return (
        color_distance(color, REFEREE_HSV)
        < REFEREE_THRESHOLD
    )


def detect_people(frame):
    results = model.predict(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            coordinates = (
                box.xyxy[0].cpu().numpy()
            )

            shirt_color = extract_shirt_color(
                frame,
                coordinates
            )

            if shirt_color is None:
                continue

            detections.append(
                (coordinates, shirt_color)
            )

    return detections


def collect_team_samples(video):
    samples = []

    frame_index = 0

    while len(samples) < MAX_SAMPLES:
        success, frame = video.read()

        if not success:
            break

        frame_index += 1

        if frame_index % SAMPLE_INTERVAL != 0:
            continue

        detections = detect_people(frame)

        for _, color in detections:
            if is_referee(color):
                continue

            samples.append(color)

            if len(samples) >= MAX_SAMPLES:
                break

    return np.array(samples, dtype=np.float32)


def train_team_classifier(samples):
    if len(samples) < MIN_SAMPLES:
        raise RuntimeError(
            "Not enough shirt samples to identify teams."
        )

    classifier = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    classifier.fit(samples)

    return classifier


def classify_person(color, classifier):
    if is_referee(color):
        return "Referee", REFEREE_COLOR

    team_id = int(
        classifier.predict(
            color.reshape(1, -1)
        )[0]
    )

    return (
        f"Team {chr(65 + team_id)}",
        TEAM_COLORS[team_id]
    )


def draw_detection(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def process_video():
    video = cv2.VideoCapture(INPUT_VIDEO)
    print("Video opened:", video.isOpened())
    print("Video path:", INPUT_VIDEO)
    print("FPS:", video.get(cv2.CAP_PROP_FPS))
    print("Frames:", video.get(cv2.CAP_PROP_FRAME_COUNT))

    if not video.isOpened():
        raise RuntimeError(
            f"Cannot open video: {INPUT_VIDEO}"
        )

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        video.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print("Collecting shirt samples...")

    samples = collect_team_samples(video)
    print("Collected samples:", len(samples))

    print("Training team classifier...")

    classifier = train_team_classifier(samples)

    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        video.release()
        raise RuntimeError(
            "Cannot create output video."
        )

    frame_number = 0

    try:
        while True:
            success, frame = video.read()

            if not success:
                break

            frame_number += 1

            detections = detect_people(frame)

            counts = {
                "Team A": 0,
                "Team B": 0,
                "Referee": 0
            }

            for box, shirt_color in detections:
                label, color = classify_person(
                    shirt_color,
                    classifier
                )

                counts[label] += 1

                draw_detection(
                    frame,
                    box,
                    label,
                    color
                )

            summary = (
                f"Team A: {counts['Team A']}  "
                f"Team B: {counts['Team B']}  "
                f"Referees: {counts['Referee']}"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (min(width, 650), 45),
                (20, 20, 20),
                -1
            )

            cv2.putText(
                frame,
                summary,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            writer.write(frame)

            cv2.imshow(
                "Football ML Analysis",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_number % 100 == 0:
                print(
                    f"Processed {frame_number} frames"
                )

    finally:
        video.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Output saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    process_video()