import cv2
import mediapipe as mp
import numpy as np
import sys

mp_face_mesh = mp.solutions.face_mesh

def align_face(image, landmarks):
    """
    Rotate the image so that the eyes are horizontal.
    Returns the rotated image and the rotation matrix.
    """
    h, w = image.shape[:2]

    # Get eye landmarks
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # Convert normalized coordinates to pixels
    left_eye_coord = np.array([left_eye.x * w, left_eye.y * h])
    right_eye_coord = np.array([right_eye.x * w, right_eye.y * h])

    # Compute angle between eyes
    dy = right_eye_coord[1] - left_eye_coord[1]
    dx = right_eye_coord[0] - left_eye_coord[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get rotation matrix around image center
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

    # Rotate the image
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def detect_face_direction(image):
    """
    Detect face looking direction.
    Returns:
      screen_direction: left/right/straight relative to the image
      person_direction: left/right/straight relative to the person's view
    """
    h, w, _ = image.shape

    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        # Process the image with MediaPipe
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Align face so eyes are horizontal
            aligned_img, M = align_face(image, landmarks)

            # Recompute landmark positions after alignment
            aligned_landmarks = []
            for lm in landmarks:
                x_px = lm.x * w
                y_px = lm.y * h
                coord = np.array([x_px, y_px, 1])
                new_coord = M @ coord  # apply rotation
                aligned_landmarks.append(new_coord)
            aligned_landmarks = np.array(aligned_landmarks)

            # Get eye center and nose position
            left_eye_coord = aligned_landmarks[33]
            right_eye_coord = aligned_landmarks[263]
            nose_coord = aligned_landmarks[1]

            eye_center_x = (left_eye_coord[0] + right_eye_coord[0]) / 2
            nose_rel_x = nose_coord[0] - eye_center_x

            # Threshold in pixels (~depends on image width)
            threshold = w * 0.03

            # Screen-relative direction
            if nose_rel_x < -threshold:
                screen_direction = "left"
            elif nose_rel_x > threshold:
                screen_direction = "right"
            else:
                screen_direction = "straight"

            # Person-relative direction (mirror)
            if screen_direction == "left":
                person_direction = "right"
            elif screen_direction == "right":
                person_direction = "left"
            else:
                person_direction = "straight"

            return screen_direction, person_direction

        else:
            # No face detected
            return "No face detected", "No face detected"


if __name__ == "__main__":
    # Get image path from command line 
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Please enter image path")

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found:", image_path)
        sys.exit(1)

    # Detect face direction
    screen_dir, person_dir = detect_face_direction(img)

    # Print results
    print("#######################")
    print(f"Screen: {screen_dir}, Person: {person_dir}")
