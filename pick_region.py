import argparse
import json
from pathlib import Path

import cv2


def parse_time(value):
    """Accept SS, MM:SS, or HH:MM:SS."""
    parts = [float(p) for p in value.split(":")]

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    raise ValueError("Use SS, MM:SS, or HH:MM:SS")


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="AllFindingsInOne.mp4")
parser.add_argument("--time", default="00:01:00")
parser.add_argument("--output", default="regions_verified.json")
args = parser.parse_args()

video = cv2.VideoCapture(args.input)

if not video.isOpened():
    raise SystemExit(f"Cannot open video: {args.input}")

timestamp = parse_time(args.time)
video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
ok, original = video.read()
video.release()

if not ok:
    raise SystemExit("Could not extract a frame at that timestamp.")

height, width = original.shape[:2]

# Fit the display to your monitor without changing source coordinates.
max_width = 1450
max_height = 780
scale = min(max_width / width, max_height / height, 1.0)

display_width = round(width * scale)
display_height = round(height * scale)

display_original = cv2.resize(
    original,
    (display_width, display_height),
    interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
)

start_point = None
end_point = None
drawing = False


def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        end_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False


window = "Select webpage bounds | Drag mouse | S=Save | R=Reset | Q=Quit"

cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(window, mouse_callback)

print(f"Original frame: {width} x {height}")
print(f"Frame timestamp requested: {timestamp:.3f} seconds")
print("Drag a rectangle around the webpage content.")
print("Press S to save, R to reset, or Q to quit.")

while True:
    display = display_original.copy()

    if start_point is not None and end_point is not None:
        cv2.rectangle(
            display,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    cv2.imshow(window, display)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        start_point = None
        end_point = None

    elif key == ord("q") or key == 27:
        break

    elif key == ord("s"):
        if start_point is None or end_point is None:
            print("Select a rectangle first.")
            continue

        # Convert displayed coordinates to original video pixels.
        x1 = round(min(start_point[0], end_point[0]) / scale)
        y1 = round(min(start_point[1], end_point[1]) / scale)
        x2 = round((max(start_point[0], end_point[0]) + 1) / scale)
        y2 = round((max(start_point[1], end_point[1]) + 1) / scale)

        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        box_width = x2 - x1
        box_height = y2 - y1

        if box_width <= 0 or box_height <= 0:
            print("Invalid selection. Try again.")
            continue

        print("\nExact source-frame coordinates:")
        print(f"x = {x1}")
        print(f"y = {y1}")
        print(f"width = {box_width}")
        print(f"height = {box_height}")

        print("\nNormalized coordinates:")
        print(f"x = {x1 / width:.6f}")
        print(f"y = {y1 / height:.6f}")
        print(f"w = {box_width / width:.6f}")
        print(f"h = {box_height / height:.6f}")

        # At 30 FPS, leave boundary frames inside the excluded intervals.
        regions = [
            {
                "start": 0.0,
                "end": 158.966667,
                "x": x1 / width,
                "y": y1 / height,
                "w": box_width / width,
                "h": box_height / height,
            },
            {
                "start": 209.033333,
                "end": 347.966667,
                "x": x1 / width,
                "y": y1 / height,
                "w": box_width / width,
                "h": box_height / height,
            },
        ]

        output = Path(args.output)
        output.write_text(
            json.dumps({"regions": regions}, indent=2),
            encoding="utf-8",
        )

        print(f"\nSaved: {output.resolve()}")
        print("Review the rectangle before using these coordinates.")
        break

cv2.destroyAllWindows()