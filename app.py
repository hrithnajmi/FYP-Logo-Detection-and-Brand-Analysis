from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.parse

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Adjusted to use current working directory
RESULT_FOLDER = os.path.join(os.getcwd(), 'results')  # Adjusted to use current working directory
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
CUSTOM_MODELS_FOLDER = os.path.join(os.getcwd(), 'yolov5', 'custom_models')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        model_name = request.form['model']
        model_path = os.path.join(CUSTOM_MODELS_FOLDER, f"{model_name}.pt")
        processed_video_relative_path, csv_path = process_video(filepath, model_path)

        print(f"processedVideoRelPth: {processed_video_relative_path}")
        print(f"csv is at: {csv_path}")

        try:
            os.startfile(processed_video_relative_path)
        except FileNotFoundError:
            return jsonify({'error': 'Processed video not found.'}), 404

        # Redirect to charts with csv_path as query parameter
        encoded_csv_path = urllib.parse.quote_plus(csv_path)
        return jsonify({
            'message': 'Video processed and opened in Media Player.',
            'model': model_name,
            'path': processed_video_relative_path,
            'csv_path': csv_path,
            'redirect_url': f'/charts?csv_path={encoded_csv_path}'  # Send encoded csv_path to charts
        }), 200

def process_video(video_path, model_path):
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    processed_video_folder = f"processed_{os.path.splitext(os.path.basename(video_path))[0]}"
    processed_video_name = os.path.basename(video_path)
    
    processed_video_dir = os.path.join(app.config['RESULT_FOLDER'], processed_video_folder)
    processed_video_path = os.path.join(processed_video_dir, processed_video_name)
    csv_path = os.path.join(processed_video_dir, "predictions.csv")

    os.makedirs(processed_video_dir, exist_ok=True)
    
    detect_script_path = os.path.join(app.root_path, 'yolov5', 'detect.py')
    subprocess.run([
        'python', detect_script_path, 
        '--weights', model_path, 
        '--source', video_path, 
        '--project', app.config['RESULT_FOLDER'], 
        '--name', processed_video_folder,
        '--save-csv',
        '--exist-ok'
    ])

    return processed_video_path.replace('\\', '/'), csv_path.replace('\\', '/')

@app.route('/charts')
def charts():
    csv_path_encoded = request.args.get('csv_path')
    csv_path = urllib.parse.unquote_plus(csv_path_encoded)
    print(f"this is csv in charts app.py: {csv_path}")

    if not csv_path or not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found"}), 404
    
    # Construct csv_dir relative to RESULT_FOLDER
    csv_dir = os.path.relpath(os.path.dirname(csv_path), app.config['RESULT_FOLDER'])

    df = pd.read_csv(csv_path)

    # Calculate total frames, total duration
    total_frames = df.shape[0]
    total_duration_ms = df["Duration (ms)"].sum()
    total_duration_seconds = total_duration_ms / 1000

    # Initialize lists and variables
    class_names = df["Prediction"].unique()
    class_names = [name for name in class_names if name != "no detections"]
    total_detection_duration = {class_name: 0 for class_name in class_names}
    total_detection_count = {class_name: 0 for class_name in class_names}
    max_confidence = -np.inf
    min_confidence = np.inf
    all_confidences = []

    # Calculate statistics
    for index, row in df.iterrows():
        prediction = row["Prediction"]
        confidence = row["Confidence"]
        duration = row["Duration (ms)"]

        # Calculate total detection duration per class
        if prediction in class_names:
            total_detection_duration[prediction] += duration
            total_detection_count[prediction] += 1

        # Update max and min confidence
        if not pd.isna(confidence):
            all_confidences.append(confidence)
            if confidence > max_confidence:
                max_confidence = confidence
            if confidence < min_confidence:
                min_confidence = confidence

    # Calculate averages and percentages
    average_confidences = {class_name: df[df["Prediction"] == class_name]["Confidence"].mean() for class_name in class_names}
    percentage_duration_with_detection = {class_name: (total_detection_duration[class_name] / total_duration_ms) * 100 for class_name in class_names}

    # Plotting graphs
    # 1. Distribution of Confidence Values
    plt.figure(figsize=(10, 6))
    plt.hist([confidence for confidence in all_confidences if confidence > 0], bins=10, alpha=0.7, color='orange')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Distribution of Confidence Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(app.config['RESULT_FOLDER'],csv_dir ,'confidence_distribution.png'))

    # 2. Duration Analysis: Box plot
    duration_data = [df[df["Prediction"] == class_name]["Duration (ms)"] for class_name in class_names]
    plt.figure(figsize=(10, 6))
    plt.boxplot(duration_data, labels=class_names)
    plt.xlabel('Logo Class')
    plt.ylabel('Duration (ms)')
    plt.title('Duration Analysis')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(app.config['RESULT_FOLDER'],csv_dir , 'duration_analysis.png'))

    # 3. Percentage of Duration for Detected Logos: Pie Chart
    plt.figure(figsize=(8, 8))
    plt.pie([percentage_duration_with_detection[class_name] for class_name in class_names], labels=class_names, autopct='%1.1f%%', startangle=140)
    plt.title('Percentage of Duration for Detected Logos')
    plt.tight_layout()
    plt.savefig(os.path.join(app.config['RESULT_FOLDER'],csv_dir , 'percentage_duration.png'))

    # 4. Average Confidence per Class: Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, [average_confidences[class_name] for class_name in class_names], color='green')
    plt.xlabel('Logo Class')
    plt.ylabel('Average Confidence')
    plt.title('Average Confidence per Class')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(app.config['RESULT_FOLDER'],csv_dir , 'average_confidence.png'))

    return render_template('charts.html', csv_dir=csv_dir)

@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
