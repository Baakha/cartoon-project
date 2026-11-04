from pathlib import Path
from collections import Counter, deque

import cv2
import numpy as np

DATABASE = {
    "Shrek": "shrek.jpg",
    "Harry Potter": "harry.jpg",
}

project_dir = Path(__file__).resolve().parent
orb = cv2.ORB_create(
    nfeatures=3000,
    nlevels=10,
    edgeThreshold=15,
    fastThreshold=10,
)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_templates():
    templates = {}

    for name, filename in DATABASE.items():
        path = project_dir / filename
        if not path.exists():
            print(f"[WARN] {name}: file not found - {path.name}")
            continue

        color_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color_image is None:
            print(f"[WARN] {name}: could not read {path.name}")
            continue

        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        image = clahe.apply(image)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is None or len(keypoints) < 12:
            print(f"[WARN] {name}: not enough image details in {path.name}")
            continue

        templates[name] = (image.shape, keypoints, descriptors)
        print(f"[OK] {name}: image template loaded")

    if not templates:
        raise FileNotFoundError("No usable character images found in the project folder.")

    return templates


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

    for name, (template_shape, template_keypoints, template_descriptors) in templates.items():
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


def main():
    templates = load_templates()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check that it is connected and not in use.")

    print("[INFO] Show a character image to the camera. Press 'q' to quit.")
    recent_detections = deque(maxlen=5)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")

            name, inliers, polygon = recognize_image(frame, templates)
            frame = cv2.flip(frame, 1)
            if polygon is not None:
                polygon = polygon.copy()
                polygon[:, :, 0] = frame.shape[1] - 1 - polygon[:, :, 0]
            recent_detections.append((name, inliers, polygon))
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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
