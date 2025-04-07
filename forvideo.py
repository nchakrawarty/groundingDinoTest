import cv2
import torch
from groundingdino.util.inference import load_model, predict, annotate
from groundingdino.util.misc import nested_tensor_from_tensor_list
import os

# Load the Grounding DINO model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(torch.__version__)           # Should indicate a CUDA-enabled version, e.g., 2.5.1+cu121
print(torch.cuda.is_available())   # Should return True
print(torch.version.cuda)          # Should match your CUDA version or nearby compatibility
print(torch.cuda.device_count())  # Should be > 0
print(device)

# Configuration
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = (
    "plastic, bottle, paper, PET, LDPE, HDPE, plastic bag, plastic bottle, "
    "electronic waste, e-waste, metal cans, aluminum can, tin can, cardboard, "
    "biological waste, food waste, compostable, glass, green glass, brown glass, white glass, "
    "metal, battery, clothes, fabric, shoes, trash, general waste, "
    "recyclable, non-recyclable, mixed waste, sanitary items"
)

# Input and output video paths
input_video_path = ".asset/WhatsApp Video 2024-11-18 at 13.49.44_dce814e7.mp4"
output_video_path = "output/output_video.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Process each frame
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Processing frame {frame_number}/{frame_count}")

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert frame to PyTorch tensor
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0  # Normalize to [0, 1]
    frame_tensor = frame_tensor.permute(2, 0, 1).to(device)  # Shape: (3, H, W)

    # Create a nested tensor
    # samples = nested_tensor_from_tensor_list([frame_tensor])

    # Run prediction
    try:
        # Run prediction (pass the frame tensor directly)
        boxes, logits, phrases = predict(
            model=model,
            image=frame_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

    except Exception as e:
        print(f"Error during prediction: {e}")
        break

    # Annotate the frame
    annotated_frame = annotate(image_source=frame_rgb, boxes=boxes, logits=logits, phrases=phrases)

    # Convert RGB back to BGR for OpenCV
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Write the annotated frame to the output video
    out.write(annotated_frame_bgr)

    frame_number += 1

# Release the video objects
cap.release()
out.release()

print("Video processing complete. Output saved to", output_video_path)
