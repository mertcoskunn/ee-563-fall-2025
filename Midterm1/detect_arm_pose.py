import cv2
import mediapipe as mp
import numpy as np
import sys


def classify_arm_rotated(image_path):
    mp_pose = mp.solutions.pose

    image = cv2.imread(image_path)
    if image is None:
        print("Could not read image:", image_path)
        return "None"

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return "None"

        landmarks = results.pose_landmarks.landmark

        # MediaPipe landmark indices
        LEFT_SHOULDER = 11
        LEFT_WRIST = 15
        RIGHT_SHOULDER = 12
        RIGHT_WRIST = 16

        # Get shoulder and wrist positions in pixels
        h, w, _ = image.shape
        l_sh = np.array([landmarks[LEFT_SHOULDER].x * w, landmarks[LEFT_SHOULDER].y * h])
        r_sh = np.array([landmarks[RIGHT_SHOULDER].x * w, landmarks[RIGHT_SHOULDER].y * h])
        l_wr = np.array([landmarks[LEFT_WRIST].x * w, landmarks[LEFT_WRIST].y * h])
        r_wr = np.array([landmarks[RIGHT_WRIST].x * w, landmarks[RIGHT_WRIST].y * h])

        # Compute shoulder line vector
        sh_vec = r_sh - l_sh
        sh_vec_norm = sh_vec / np.linalg.norm(sh_vec)

        # Compute perpendicular vector (normal)
        normal = np.array([-sh_vec_norm[1], sh_vec_norm[0]])

        # Project wrists onto normal (distance from shoulder line)
        l_dist = np.dot(l_wr - l_sh, normal)
        r_dist = np.dot(r_wr - l_sh, normal)

        # If distance > 0, wrist is "above" shoulder line (depending on normal direction)
        left_up = l_dist > 0
        right_up = r_dist > 0

        if left_up and right_up:
            return "both"
        elif left_up:
            return "left"
        elif right_up:
            return "right"
        else:
            return "None"

if __name__ == "__main__":
    # Get image path from command line 
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please enter image path")
        sys.exit(1)

    result = classify_arm_rotated(image_path)
    print("#######################")
    print("Result: " + str(result))
