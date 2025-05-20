import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time
import json
import glob
import shutil
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import scipy.io as sio
from scipy.io import loadmat

from flask import Flask, render_template, jsonify, request, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename
from flask_cors import CORS
from torch.utils.data import DataLoader

from model import get_model
from utils import get_transform
from evaluate import plot_predictions
from dataset import DogHeartTestDataset
from PIL import Image
import re
from flask import jsonify
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_session import Session


app = Flask(__name__)

## Train Images
# IMAGE_FOLDER ="static/new_dataset/Images"
# BACKUP_MAT_FOLDER = 'static/new_dataset/Truth_Backup'
# GROUND_TRUTH_FOLDER = 'static/new_dataset/Labels'
# PREDICT_FOLDER = 'static/new_dataset/Train/Predict'

## Valid Images
# IMAGE_FOLDER ="static/new_dataset/Valid/Images"
# GROUND_TRUTH_FOLDER = 'static/new_dataset/Valid/Labels'
# BACKUP_MAT_FOLDER = 'static/new_dataset/Valid_Backup'

## Test Images
IMAGE_FOLDER ="static/new_dataset/Test_Images/Images"
GROUND_TRUTH_FOLDER = 'static/new_dataset/Test_Images/Labels'
BACKUP_MAT_FOLDER = 'static/new_dataset/Test_Backup'
PREDICT_FOLDER = 'static/new_dataset/Test_Images/Predict'


OVERLAY_FOLDER = 'static/new_dataset/Test/overlayed_images'

app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
app.config['PREDICT_FOLDER'] = PREDICT_FOLDER
app.config['OVERLAY_FOLDER'] = OVERLAY_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mat'}

# Ensure overlay folder exists
os.makedirs(OVERLAY_FOLDER, exist_ok=True)
os.makedirs(GROUND_TRUTH_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)

CORS(app, resources={r"/*": {"origins": "*"}})

# 🔹 MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/vhs_db"
mongo = PyMongo(app)

# 🔹 Flask Session Configuration
app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = "scr23#234dfQfsw"
Session(app)

# 🔹 Password Encryption
bcrypt = Bcrypt(app)


#region "Methods"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clear_folder(folder_name):
    files = glob.glob(os.path.join(folder_name, '*'))
    for f in files:
        os.remove(f)

def calculate_vhs(image_name,points):
    # Convert points to NumPy array (in case it's a list)
    points = np.array(points, dtype=np.float32)

    # --- Load image and get original size ---
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    img = Image.open(img_path)
    width, height = img.size  # (W, H)

    # --- Resize the points to match new image size (512x512) ---
    target_size = 512
    points[:, 0] = (points[:, 0] / width) * target_size
    points[:, 1] = (points[:, 1] / height) * target_size

    # Convert input to torch tensors
    pts = [torch.tensor(p, dtype=torch.float32) for p in points]

    A, B, C, D, E, F = pts

    # Calculate segment lengths using Euclidean norm
    AB = torch.norm(A - B, p=2, dim=-1)
    CD = torch.norm(C - D, p=2, dim=-1)
    EF = torch.norm(E - F, p=2, dim=-1)    
    vhs = 6 * (AB + CD) / EF 

    return {
        "Points": [p.tolist() for p in pts],
        "AB": round(AB.item(), 8),
        "CD": round(CD.item(), 8),
        "EF": round(EF.item(), 8),
        "VHS": round(vhs.item(), 8)
    }


def calc_vhs(x: torch.Tensor):
    """Compute VHS value based on six predicted points."""
    A, B = x[..., 0:2], x[..., 2:4]
    C, D = x[..., 4:6], x[..., 6:8]
    E, F = x[..., 8:10], x[..., 10:12]

    AB = torch.norm(A - B, p=2, dim=-1)
    CD = torch.norm(C - D, p=2, dim=-1)
    EF = torch.norm(E - F, p=2, dim=-1)

    vhs = 6 * (AB + CD) / EF
    return vhs

def calculateVHS(A,B,C,D,E,F):
    # Calculate distances using Euclidean formula
    AB = np.linalg.norm(B - A)  
    CD = np.linalg.norm(D - C)  
    EF = np.linalg.norm(F - E) 

    # Calculate VHS
    VHS = 6 * (AB + CD) / EF
    return VHS

def save_predictions_to_mat(image_name, predicted_points, vhs_value):
    """Save predicted points and VHS in MATLAB-compatible .mat format."""
    mat_filename = os.path.splitext(image_name)[0] + ".mat"
    mat_filepath = os.path.join(PREDICT_FOLDER, mat_filename)

    # Convert points to correct MATLAB format
    mat_data = {
        "six_points": np.array(predicted_points),  # Store 6 points in an array
        "VHS": np.array([[vhs_value]])  # Store VHS as a 2D array
    }

    sio.savemat(mat_filepath, mat_data)
    print(f"✅ Predictions saved in MATLAB format: {mat_filepath}")

#endregion

#region "HTML Template APIs"

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/vhs')
@app.route('/dashboard')
def vhs():
    if "user" not in session:  # 🔹 Check if user is NOT logged in
        return redirect(url_for("login"))  # 🔹 Redirect to login page
    
    return render_template('vhs.html')  # ✅ If logged in, show the page

# 🔹 Home Route (Redirect logged-in users)
@app.route("/")
@app.route("/login")
def login_page():
    if "user" in session:  # ✅ Check if user is already logged in
        return redirect(url_for("vhs"))  # Redirect to dashboard
    return render_template("login.html")  # Show login page if not logged in

# 🔹 Register Route (Redirect logged-in users)
@app.route("/register")
def register_page():
    if "user" in session:
        return redirect(url_for("vhs"))  # Redirect to dashboard
    return render_template("register.html")  # Show register page if not logged in


# User Registration
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()  # ✅ Get JSON data instead of form data
    if not data:
        return jsonify({"success": False, "message": "Invalid request data"}), 400
    
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"success": False, "message": "Missing username or password"}), 400

    if mongo.db.users.find_one({"username": username}):
        return jsonify({"success": False, "message": "Username already exists!"}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    mongo.db.users.insert_one({"username": username, "password": hashed_pw})

    return jsonify({"success": True, "message": "Registration successful! Redirecting..."}), 200

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "Invalid request data"}), 400

        username = data.get("username")
        password = data.get("password")

        user = mongo.db.users.find_one({"username": username})
        
        if user and bcrypt.check_password_hash(user["password"], password):
            session["user"] = username
            return jsonify({"success": True, "message": "Login successful! Redirecting..."}), 200
        else:
            return jsonify({"success": False, "message": "Invalid Credentials"}), 401
    
    except Exception as e:
        return jsonify({"success": False, "message": "Internal Server Error", "error": str(e)}), 500

#endregion

#region "API"

@app.route('/upload_images', methods=['POST'])
def upload_images():
    # Check if mode is provided in the request
    mode = request.form.get('mode')
    if not mode or mode not in ['images', 'truth']:
        return jsonify({'error': 'Invalid or missing mode'}), 400

    # Determine the folder based on mode
    target_folder = IMAGE_FOLDER if mode == 'images' else GROUND_TRUTH_FOLDER

    # Check if files are provided
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    
    if not files:
        return jsonify({'error': 'No selected file'}), 400

    # Clear the target folder before saving new files (optional)
    clear_folder(target_folder)

    uploaded_files = []
    
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            
            # Validate file type based on mode
            if mode == 'images' and not allowed_file(filename):  # Ensure it's an image for "images"
                continue
            
            if mode == 'truth' and not filename.endswith('.mat'):  # Ensure it's a .mat file for "truth"
                continue
            
            # Save the file to the target folder
            file.save(os.path.join(target_folder, filename))
            uploaded_files.append(filename)
    
    if uploaded_files:
        return jsonify({
            'message': f'Files uploaded successfully to {mode} folder',
            'files': uploaded_files
        }), 200
    else:
        return jsonify({'error': f'No valid {mode} files uploaded'}), 400

def natural_sort_key(s):
    # Extract numeric parts from the filename for natural sorting
    return [float(c) if c.replace('.', '', 1).isdigit() else c.lower() for c in re.split(r'(\d+\.\d+|\d+)', s)]

@app.route('/get_image/<int:index>')
def get_image(index):
    # Get list of images and sort them naturally
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    images.sort(key=natural_sort_key)  # Sort images naturally based on numeric parts
    
    if 0 <= index < len(images):
        image_name = images[index]
        image_path = os.path.join(app.config['IMAGE_FOLDER'], image_name)

        # Get image dimensions using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Could not read image"}), 500
        
        height, width, _ = image.shape  # Extract height and width

        return jsonify({
            "url": f"{IMAGE_FOLDER}\{image_name}",
            "index": index,
            "width": width,
            "height": height,
            "image_name":image_name
        })
    
    return jsonify({"error": "Image not found"}), 404

@app.route('/get_total_images')
def get_total_images():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return jsonify({"total": len(images)})
    
@app.route('/save_coordinates/<image_name>', methods=['POST'])
def save_coordinates(image_name):
    """
    Save six predicted points into a .mat file, calculate VHS, and store it in GROUND_TRUTH_FOLDER.
    """
    coordinates_data = request.json

    try:
        
        # Convert input to a valid NumPy array (fixing dict issues)
        six_points = np.round(np.array([list(map(float, p)) for p in coordinates_data.get("six_points")], dtype=np.float32), 5)


        # Ensure six valid points
        if six_points.shape != (6, 2):
            return jsonify({"error": "Invalid points format. Expected shape (6,2)."}), 400
        
        # Calculate VHS on the server side
        vhs_data = calculate_vhs(image_name,six_points)

        if vhs_data is None:
            return jsonify({"error": "Invalid VHS calculation (possible division by zero)."}), 400

        # Extract only VHS value for storage
        vhs_value = vhs_data["VHS"]

        # Define `.mat` file path (same name as image but with .mat extension)
        mat_filename = os.path.join(GROUND_TRUTH_FOLDER, f"{os.path.splitext(image_name)[0]}.mat")

        # Save data to .mat file
        mat_data = {
            "six_points": six_points,
            "VHS": np.array([[vhs_value]], dtype=np.float32)
        }

        backup_mat_file = os.path.join(BACKUP_MAT_FOLDER, f"{os.path.splitext(image_name)[0]}.mat")
        sio.savemat(mat_filename, mat_data)
        sio.savemat(backup_mat_file, mat_data)

        return jsonify({
            "message": "Coordinates saved successfully",
            "file": mat_filename,
            "VHS": vhs_value
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/calculate_vhs', methods=['POST'])
def vhs_endpoint():
    data = request.json
    points = data.get('points')
    image_name = data.get('imageName')

    if not points or len(points) != 6:
        return jsonify({"error": "Invalid input. Please provide 6 points."}), 400

    try:
        result = calculate_vhs(image_name,points)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict_points/<image_name>', methods=['GET'])
def predict_points(image_name):
    mat_filename = os.path.splitext(image_name)[0] + '.mat'  # Match the image name with .mat file
    mat_filepath = os.path.join(app.config['PREDICT_FOLDER'], mat_filename)
    image_filepath = os.path.join(app.config['IMAGE_FOLDER'], image_name)
    
    if not os.path.exists(mat_filepath):
        return jsonify({"error": "Prediction file not found"}), 404

    if not os.path.exists(image_filepath):
        return jsonify({"error": "Image file not found"}), 404

    try:
        # Load image
        image = cv2.imread(image_filepath)
        if image is None:
            return jsonify({"error": "Could not read image"}), 500

        # Load .mat file
        mat_data = loadmat(mat_filepath)

        # Ensure 'six_points' key exists
        if 'six_points' not in mat_data:
            return jsonify({"error": "'six_points' key not found in .mat file"}), 500

        # Extract the six points
        points_array = mat_data['six_points']  # Shape (6,2)
        labeled_points = {
            "PA": points_array[0],
            "PB": points_array[1],
            "PC": points_array[2],
            "PD": points_array[3],
            "PE": points_array[4],
            "PF": points_array[5],
        }

        # Extract VHS value if available
        vhs_value = float(mat_data['VHS'][0, 0]) if 'VHS' in mat_data else None
        if vhs_value is not None:
            vhs_value = round(vhs_value, 8)

        # Define color and thickness for drawing
        line_color = (0, 0, 255)  # Red color in BGR
        point_color = (255, 0, 0)  # Blue color for points
        thickness = 1

        # Draw lines between pairs
        line_pairs = [("PA", "PB"), ("PC", "PD"), ("PE", "PF")]
        for p1, p2 in line_pairs:
            pt1 = tuple(labeled_points[p1].astype(int))
            pt2 = tuple(labeled_points[p2].astype(int))
            cv2.line(image, pt1, pt2, line_color, thickness)

        # # Draw points
        # for label, point in labeled_points.items():
        #     pt = tuple(point.astype(int))
        #     #cv2.circle(image, pt, 5, point_color, -1)
        #     #cv2.putText(image, label, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)

        # Save overlayed image
        overlayed_filename = f"{image_name}"
        overlayed_filepath = os.path.join(app.config['OVERLAY_FOLDER'], overlayed_filename)
        cv2.imwrite(overlayed_filepath, image)

        return jsonify({
            "image": image_name,
            "overlayed_image": f"/static/overlayed_images/{overlayed_filename}",
            "labeled_points": {k: v.tolist() for k, v in labeled_points.items()},
            "VHS": vhs_value
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate_predictions', methods=['GET'])
def generate_predictions():
    clear_folder(PREDICT_FOLDER)
    
    def generate():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(device)
        checkpoint_path = f'/static/model/bm_15.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        img_size = 512

        test_dataset = DogHeartTestDataset(IMAGE_FOLDER, transforms=get_transform(img_size))
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        total_images = len(test_loader.dataset)
        processed_images = 0

        model.eval()

        with torch.no_grad():
            for images, names in test_loader:
                images = images.to(device)
                outputs = model(images)
                outputs = outputs.cpu().numpy()
                outputs = outputs.reshape(outputs.shape[0], 6, 2)

                for i, points in enumerate(outputs):
                    img = Image.open(f'{IMAGE_FOLDER}/{names[i]}')
                    
                    # Get original image size and return predicted points back to original points size
                    w, h = img.size
                    points = points.reshape(-1, 2)
                    points = points * img_size
                    
                    points[:, 0] = w / img_size * points[:, 0]
                    points[:, 1] = h / img_size * points[:, 1]
                    
                    vhs = calculateVHS(points[0], points[1], points[2], points[3], points[4], points[5])

                    save_predictions_to_mat(names[i], points, vhs)

                    processed_images += 1
                    progress = round((processed_images * 100) / total_images, 0)
                    
                    # ✅ Ensure correct SSE format with double newline
                    yield f"data: {json.dumps({'progress': progress})}\n\n"

        # ✅ Keep the connection alive for a few seconds to ensure proper closure
        yield "data: {\"progress\": 100, \"status\": \"done\"}\n\n"

    return Response(generate(), content_type="text/event-stream; charset=utf-8")

#################### Predict Direct Sub Method ################
def process_image_logic(mode, image_name):
    """Processes the image based on the specified mode and returns the relevant data."""
    # Locate corresponding image
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    if not os.path.exists(image_path):
        return {"error": f"Corresponding image {image_name} not found!"}, 404

    # Load image (keep original size)
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape  # (height, width, channels)

    # Match the image name with its corresponding .mat file
    mat_filename = os.path.splitext(image_name)[0] + '.mat'

    overlayed_image_url = f"{IMAGE_FOLDER}/{image_name}"

    six_points = None
    vhs_value = None

    # Handle prediction mode (always available)
    if mode == "predict" or mode == "both":
        mat_filepath = os.path.join(PREDICT_FOLDER, mat_filename)

        if not os.path.exists(mat_filepath):
            return {"error": f"{mat_filepath} not found!"}, 404

        # Load .mat file for prediction
        mat_data = sio.loadmat(mat_filepath)
        if "six_points" not in mat_data or "VHS" not in mat_data:
            return {"error": "Missing 'six_points' or 'VHS' in .mat file"}, 400

        six_points = mat_data["six_points"].astype(float)
        vhs_value = calculateVHS(six_points[0], six_points[1], six_points[2], six_points[3], six_points[4], six_points[5])
        #vhs_value = round(float(mat_data["VHS"][0, 0]), 8)
        vhs_value = round(vhs_value, 2)
        
        # Overlay prediction points on the image
        for (x, y) in six_points:
            #cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(image, (int(round(x)), int(round(y))), radius=1, color=(0, 0, 255), thickness=-1)

            label = f"({x:.2f}, {y:.2f})"
            #cv2.putText(image, label, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)



        # Draw bold red lines between specific pairs of points
        line_pairs = [(0, 1), (2, 3), (4, 5)]
        for p1, p2 in line_pairs:
            x1, y1 = six_points[p1]
            x2, y2 = six_points[p2]
            cv2.line(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
           )

        # Add a legend for the prediction (Red dot with 'Prediction' label)
        legend_x, legend_y = image.shape[1] - 150, 20
        cv2.circle(image, (legend_x, legend_y), radius=4, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            image,
            f"Prediction({vhs_value})",
            (legend_x + 15, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,                    # 🔠 Bigger text
            (255, 255, 255),        # ⚪ White color
            2,                      # 🖍️ Thicker stroke
            cv2.LINE_AA             # 🔍 Smooth edges
        )

        # Save the overlayed image
        overlayed_path = os.path.join(OVERLAY_FOLDER, image_name)
        cv2.imwrite(overlayed_path, image)

    # Check if ground truth exists, otherwise skip truth processing
    if mode == "truth" or mode == "both":
        mat_filepath = os.path.join(GROUND_TRUTH_FOLDER, mat_filename)

        if not os.path.exists(mat_filepath):
            if mode=="both":
            # No ground truth found, return error and fallback to predict mode
                return process_image_logic("predict", image_name)  # Fallback to 'predict' mode
            else:
                return {"error": "NO_GT_FOUND"}, 404

        # Load .mat file for ground truth
        mat_data = sio.loadmat(mat_filepath)
        if "six_points" not in mat_data or "VHS" not in mat_data:
            return {"error": "Missing 'six_points' or 'VHS' in .mat file"}, 400

        six_points = mat_data["six_points"].astype(float)
        vhs_value = calculateVHS(six_points[0], six_points[1], six_points[2], six_points[3], six_points[4], six_points[5])
        #vhs_value = round(float(mat_data["VHS"][0, 0]), 8)
        vhs_value = round(vhs_value, 2)

        if mode == "both":
            overlayed_image_url = f"{OVERLAY_FOLDER}/{image_name}"

    # Create a dictionary to store the points
    points_dict = {
        "PA": six_points[0].tolist(),
        "PB": six_points[1].tolist(),
        "PC": six_points[2].tolist(),
        "PD": six_points[3].tolist(),
        "PE": six_points[4].tolist(),
        "PF": six_points[5].tolist()
    }

    return {
        "image": image_name,
        "overlayed_image": overlayed_image_url,
        "six_points": points_dict,  # Include the points dictionary
        "VHS": vhs_value,
        "width": img_width,
        "height": img_height
    }, 200

@app.route('/predict_direct/<image_name>', methods=['GET'])
def plot_predictions_direct(image_name):
    """Handles the request, processes the image based on mode, and returns the response."""
    # Get the mode from request arguments (defaults to 'predict')
    mode = request.args.get('mode', 'predict').lower()

    response_data, status_code = process_image_logic(mode, image_name)
    
    return jsonify(response_data), status_code
        
#endregion

@app.after_request
def disable_ngrok_buffering(response):
    response.headers["X-Accel-Buffering"] = "no"
    return response

if __name__ == '__main__':
    app.run(debug=True)