from ModelUNET import BallonSegmentator_Unet
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]

    model_unet = BallonSegmentator_Unet()
    model_unet.load_model("unet_balloon.pth")

    area_percent = model_unet.predict_area_percent(image_path)
    print(f"Balloon area (%): {area_percent:.2f}")
