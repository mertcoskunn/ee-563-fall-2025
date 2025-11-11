## Requirements

You need to install the following Python packages before running the code:

```bash
pip install mediapipe opencv-python
```

# Arm Up Detection

This project uses **MediaPipe Pose** to detect body landmarks in a given image and classify which arm is raised.

### Usage
Run the script with an image path as an argument:

```bash
python detect_arm_pose.py IMG_PATH
```
## Algorithm Overview

1. **Pose Detection:**  
   The image is processed with MediaPipe Pose to obtain key landmarks, including shoulders and wrists.

2. **Shoulder Line Calculation:**  
   A line is defined between the left and right shoulders to represent the torso axis.

3. **Projection Method:**  
   Each wrist is projected onto a vector perpendicular to the shoulder line.

4. **Classification:**  
   - If the projected wrist position is above the shoulder line, the arm is considered **“up”**.  
   - Based on both arms’ positions, the result is classified as `left`, `right`, `both`, or `None`.

This approach ensures that the detection works correctly even if the image is rotated, flipped, or the person is at an angle.

## 2. Face Direction Detection

This project uses **MediaPipe Face Mesh** to detect facial landmarks and classify the looking direction of a face in a given image.

### Usage
Run the script with an image path as an argument:

```bash
python detect_face_direction.py IMG_PATH
```

### Algorithm Overview

1. **Face Landmark Detection:**  
   The image is processed with MediaPipe Face Mesh to obtain key facial landmarks, including eyes and nose.

2. **Face Alignment:**  
   The image is rotated so that the line between the eyes is horizontal. This ensures consistent detection regardless of image rotation.

3. **Nose Projection:**  
   - The nose tip position is compared to the horizontal center of the eyes.  
   - The horizontal difference determines the face's looking direction relative to the image.

4. **Classification:**  
   - **Screen-relative direction:** Left/right/straight relative to the image.  
   - **Person-relative direction:** Left/right/straight relative to the person's view (mirrored from screen-relative).

5. **Thresholding:**  
   A small threshold is applied to handle minor variations and prevent false classification.

This approach ensures correct face direction detection even if the image is rotated, flipped, or the person is at an angle.

