from ModelYOLO import BalloonYOLO
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]

    model = BalloonYOLO(dataset_dir="")
    model.load_model("balloon_detection/yolo_balloon/weights/best.pt")

    area_percent = model.predict_and_draw(image_path)
    print(f"Balloon area (%): {area_percent:.2f}")