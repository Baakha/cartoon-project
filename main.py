from pathlib import Path
from collections import Counter, deque
from threading import Event, Lock, Thread
from zipfile import BadZipFile, ZipFile

import cv2
import numpy as np

project_dir = Path(__file__).resolve().parent
character_image_dir = project_dir / "characters"

DATABASE = {
    "Shrek": "shrek.jpg",
    "Harry Potter": "harry.jpg",
}

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MAX_TEMPLATE_SIZE = 800
MAX_FRAME_SIZE = 640

orb = cv2.ORB_create(
    nfeatures=500,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=10,
)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def resize_for_recognition(image, max_size):
    height, width = image.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image, scale


def load_templates():
    image_sources = {}

    for name, filename in DATABASE.items():
        path = project_dir / filename
        image_sources.setdefault(name, []).append((path.name, path))

    if character_image_dir.exists():
        for path in sorted(character_image_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            relative_path = path.relative_to(character_image_dir)
            name = relative_path.parts[0] if len(relative_path.parts) > 1 else path.stem
            image_sources.setdefault(name, []).append((path.name, path))

    archive_paths = list(project_dir.glob("*.zip"))
    if character_image_dir.exists():
        archive_paths.extend(character_image_dir.rglob("*.zip"))

    for path in sorted(archive_paths):
        name = path.stem
        image_sources.setdefault(name, []).append((path.name, path))

    templates = {}
    for name, sources in image_sources.items():
        for source_name, source in sources:
            if source.suffix.lower() == ".zip":
                try:
                    with ZipFile(source) as archive:
                        for member in archive.infolist():
                            if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_EXTENSIONS:
                                continue

                            image_data = np.frombuffer(archive.read(member), dtype=np.uint8)
                            color_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                            if color_image is None:
                                print(f"[WARN] {name}: could not read {member.filename}")
                                continue
                            add_template(templates, name, color_image, member.filename)
                except BadZipFile:
                    print(f"[WARN] {name}: invalid ZIP archive - {source.name}")
                continue

            if not source.exists():
                print(f"[WARN] {name}: file not found - {source.name}")
                continue

            color_image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if color_image is None:
                print(f"[WARN] {name}: could not read {source.name}")
                continue
            add_template(templates, name, color_image, source_name)

    if not templates:
        raise FileNotFoundError(
            "No usable character images found. Add images to characters/<Character name>/ "
            "or characters/<Character name>.<jpg|png|...>, or add character ZIP files "
            "to the project folder or characters folder."
        )

    print(f"[INFO] Loaded templates for {len(templates)} characters.")
    return templates


def add_template(templates, name, color_image, source_name):
    color_image, _ = resize_for_recognition(color_image, MAX_TEMPLATE_SIZE)
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    image = clahe.apply(image)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None or len(keypoints) < 12:
        print(f"[WARN] {name}: not enough image details in {source_name}")
        return

    templates.setdefault(name, []).append((image.shape, keypoints, descriptors))
    print(f"[OK] {name}: image template loaded from {source_name}")


def recognize_image(frame, templates):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)
    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray, None)
    if frame_descriptors is None:
        return None, 0, None

    best_name = None
    best_inliers = 0
    best_ratio = 0.0
    best_polygon = None

    for name, character_templates in templates.items():
        for template_shape, template_keypoints, template_descriptors in character_templates:
            pairs = matcher.knnMatch(template_descriptors, frame_descriptors, k=2)
            good_matches = [
                first
                for pair in pairs
                if len(pair) == 2
                for first, second in [pair]
                if first.distance < 0.78 * second.distance
            ]
            if len(good_matches) < 12:
                continue

            template_points = [
                template_keypoints[match.queryIdx].pt for match in good_matches
            ]
            frame_points = [frame_keypoints[match.trainIdx].pt for match in good_matches]
            homography, mask = cv2.findHomography(
                np.float32(template_points),
                np.float32(frame_points),
                cv2.RANSAC,
                4.0,
            )
            if homography is None or mask is None:
                continue

            inlier_count = int(mask.ravel().sum())
            inlier_ratio = inlier_count / len(good_matches)

            height, width = template_shape
            corners = cv2.perspectiveTransform(
                np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]),
                homography,
            )[0]
            frame_height, frame_width = gray.shape
            polygon_area = abs(cv2.contourArea(corners))
            if (
                inlier_count < 12
                or inlier_ratio < 0.25
                or not cv2.isContourConvex(corners.astype("float32"))
                or polygon_area < frame_width * frame_height * 0.003
                or polygon_area > frame_width * frame_height * 1.5
            ):
                continue

            if inlier_count < best_inliers or (
                inlier_count == best_inliers and inlier_ratio <= best_ratio
            ):
                continue

            best_name = name
            best_inliers = inlier_count
            best_ratio = inlier_ratio
            best_polygon = corners.astype("int32").reshape(1, 4, 2)

    if best_name is None:
        return None, best_inliers, None
    return best_name, best_inliers, best_polygon


def recognition_worker(templates, state, state_lock, frame_ready, stop_event):
    last_frame_id = 0
    while not stop_event.is_set():
        frame_ready.wait()
        frame_ready.clear()
        if stop_event.is_set():
            break

        with state_lock:
            frame_id = state["frame_id"]
            if frame_id == last_frame_id:
                continue
            frame = state["frame"].copy()

        try:
            recognition_frame, scale = resize_for_recognition(frame, MAX_FRAME_SIZE)
            name, inliers, polygon = recognize_image(recognition_frame, templates)
            if polygon is not None and scale < 1.0:
                polygon = np.rint(polygon / scale).astype("int32")
        except Exception as error:
            with state_lock:
                state["error"] = error
            stop_event.set()
            return

        with state_lock:
            state["result"] = (frame_id, name, inliers, polygon)
        last_frame_id = frame_id


def main():
    templates = load_templates()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check that it is connected and not in use.")

    print(
        "[INFO] Show a reference image to the camera. Add photos to "
        "characters/<Character name>/ or character ZIP files to the characters folder "
        "to recognize more characters. Press 'q' to quit."
    )
    recent_detections = deque(maxlen=5)
    state = {"frame_id": 0, "frame": None, "result": None, "error": None}
    state_lock = Lock()
    frame_ready = Event()
    stop_event = Event()
    worker = Thread(
        target=recognition_worker,
        args=(templates, state, state_lock, frame_ready, stop_event),
        daemon=True,
    )
    worker.start()
    last_result_id = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            with state_lock:
                state["frame_id"] += 1
                state["frame"] = frame.copy()
            frame_ready.set()

            with state_lock:
                error = state["error"]
                result = state["result"]
            if error is not None:
                raise RuntimeError("Character recognition failed.") from error
            if result is not None and result[0] != last_result_id:
                last_result_id, name, inliers, polygon = result
                recent_detections.append((name, inliers, polygon))

            frame = cv2.flip(frame, 1)
            found_names = [
                detection[0]
                for detection in recent_detections
                if detection[0] is not None
            ]
            stable_name, votes = (
                Counter(found_names).most_common(1)[0] if found_names else (None, 0)
            )

            if stable_name is None or votes < 3:
                cv2.putText(
                    frame,
                    "Show a known image to the camera",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                _, display_inliers, display_polygon = next(
                    detection
                    for detection in reversed(recent_detections)
                    if detection[0] == stable_name
                )
                if display_polygon is not None:
                    display_polygon = display_polygon.copy()
                    display_polygon[:, :, 0] = frame.shape[1] - 1 - display_polygon[:, :, 0]
                    cv2.polylines(frame, display_polygon, True, (0, 255, 0), 2)
                    x, y = display_polygon[0][0]
                    cv2.putText(
                        frame,
                        f"{stable_name} ({display_inliers} matches)",
                        (max(10, int(x)), max(30, int(y) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            cv2.putText(
                frame,
                "Press 'q' to quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Character Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop_event.set()
        frame_ready.set()
        worker.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
