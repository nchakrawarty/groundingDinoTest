from flask import Flask, request, jsonify, send_file, render_template_string, render_template
import cv2
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os

app = Flask(__name__)

# Load the model once during startup
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# Configuration for thresholds
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# Create directories if they don't exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('output', exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the uploaded image
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        # Process the image
        image_source, image = load_image(image_path)
        # TEXT_PROMPT = "plastic . bottle . paper . PET . LDPE . HDPE ."  # Customize as needed
        TEXT_PROMPT = (
                        "plastic, bottle, paper, PET, LDPE, HDPE, plastic bag, plastic bottle, "
                        "electronic waste, e-waste, metal cans, aluminum can, tin can, cardboard, "
                        "biological waste, food waste, compostable, glass, green glass, brown glass, white glass, "
                        "metal, battery, clothes, fabric, shoes, trash, general waste, "
                        "recyclable, non-recyclable, mixed waste, sanitary items"
                    )

        
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        # Annotate the image
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

         # Count the number of detected items for each category
        item_counts = {}
        for phrase in phrases:
            if phrase not in item_counts:
                item_counts[phrase] = 1
            else:
                item_counts[phrase] += 1
                
        
        # Save the annotated image
        output_path = os.path.join('output', f'annotated_{file.filename}')
        cv2.imwrite(output_path, annotated_frame)

        # Display the annotated image in the browser
        # return render_template_string('''
        # <h1>Annotated Image</h1>
        # <img src="{{ url_for('output_image', filename='annotated_' + filename) }}" alt="Annotated Image">
        # <br><a href="/">Upload another image</a>
        # ''', filename=file.filename)
        # Return the URL of the annotated image
        # return jsonify({"url": f"/output/annotated_{file.filename}"})
        # Return the URL of the annotated image and the counts of detected items
        return jsonify({
            "url": f"/output/annotated_{file.filename}",
            "item_counts": item_counts
        })

@app.route('/output/<filename>')
def output_image(filename):
    return send_file(os.path.join('output', filename), mimetype='image/jpeg')

@app.route('/')
def index():
    return render_template('index.html')
    # return '''
    # <html>
    #     <body>
    #         <h1>Upload Image for Annotation</h1>
    #         <form action="/upload" method="post" enctype="multipart/form-data">
    #             <input type="file" name="file" />
    #             <input type="submit" value="Upload Image" />
    #         </form>
    #     </body>
    # </html>
    # '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
