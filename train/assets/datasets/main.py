import json
import os
from PIL import Image, ImageDraw

# Load the JSON file
with open("labels_my-project-name_2024-11-27-07-41-07.json", "r") as file:
    data = json.load(file)

output_dir = "annotated_images"
os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the JSON file
for filename, details in data.items():
    img_path = filename
    regions = details.get("regions", {})

    try:
        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)

            for region_id, region in regions.items():
                shape_attributes = region["shape_attributes"]
                label = region["region_attributes"].get("label", "Unknown")
                points_x = shape_attributes["all_points_x"]
                points_y = shape_attributes["all_points_y"]

                # Draw the polygon
                polygon_points = list(zip(points_x, points_y))
                draw.polygon(polygon_points, outline="red", width=3)

                # Draw bounding box around the polygon
                min_x, max_x = min(points_x), max(points_x)
                min_y, max_y = min(points_y), max(points_y)
                draw.rectangle([min_x, min_y, max_x, max_y], outline="blue", width=2)

                # Add label
                draw.text((min_x, min_y - 10), label, fill="white")

            output_path = os.path.join(output_dir, os.path.basename(filename))
            img.save(output_path)
            print(f"Saved annotated image: {output_path}")
    except FileNotFoundError:
        print(f"Image not found: {filename}")

print("Annotation process completed.")
