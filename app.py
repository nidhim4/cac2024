import os
from PIL import Image, ImageOps
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename

from edge_detection import HED
from style_transfer import style_transfer
from inpainting import inpainting

app = Flask(__name__)

UPLOAD_FOLDER_1 = 'static/uploads/folder1'
UPLOAD_FOLDER_2 = 'static/uploads/folder2'
OUTPUT_FOLDER = 'static/outputs'
STYLE_FOLDER = 'static/images/styles'

app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
app.config['UPLOAD_FOLDER_2'] = UPLOAD_FOLDER_2
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def is_valid_color(rgb):
    return not (rgb == (0, 0, 0) or rgb == (255, 255, 255))

def give_most_hex(file_path, code='hex'):
    my_image = Image.open(file_path).convert('RGB')
    size = my_image.size

    if size[0] >= 1200 or size[1] >= 1200:
        my_image = ImageOps.scale(my_image, factor=0.6)
    elif size[0] >= 800 or size[1] >= 800:
        my_image = ImageOps.scale(my_image, factor=0.5)
    elif size[0] >= 600 or size[1] >= 600:
        my_image = ImageOps.scale(my_image, factor=0.4)
    elif size[0] >= 400 or size[1] >= 400:
        my_image = ImageOps.scale(my_image, factor=0.2)

    my_image = ImageOps.posterize(my_image, 2)
    image_array = np.array(my_image)

    unique_colors = {}
    for column in image_array:
        for rgb in column:
            t_rgb = tuple(rgb)
            if t_rgb not in unique_colors:
                unique_colors[t_rgb] = 1
            else:
                unique_colors[t_rgb] += 1

    sorted_unique_colors = sorted(unique_colors.items(), key=lambda x: x[1], reverse=True)
    filtered_colors = [color for color, count in sorted_unique_colors if is_valid_color(color)]
    top_6 = filtered_colors[0:6]

    if code == 'hex':
        hex_list = [rgb_to_hex(rgb) for rgb in top_6]
        return hex_list
    else:
        return top_6
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/extract_colors', methods=['POST'])
def extract_colors():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        hex_colors = give_most_hex(filepath)

        return jsonify({'success': True, 'hex_colors': hex_colors})
    else:
        return jsonify({'success': False, 'message': 'Invalid file type'})

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file_key = 'file-1' if 'file-1' in request.files else 'file-2'
        folder_key = 'UPLOAD_FOLDER_1' if file_key == 'file-1' else 'UPLOAD_FOLDER_2'
        file = request.files.get(file_key)

        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config[folder_key], filename)
            file.save(filepath)

            folder_name = os.path.basename(app.config[folder_key])
            file_url = url_for('static', filename=f'uploads/{folder_name}/{filename}')

            return jsonify({'success': True, 'filePath': file_url})
        else:
            return jsonify({'success': False, 'message': 'Invalid file type'})


@app.route('/perform_edge_detection', methods=['POST'])
def perform_edge_detection():
    data = request.json
    image_path = data.get('image_path')

    if image_path:
        image_file_path = image_path.replace("/static", "static")
        edge_output_path = 'static/outputs/edgedetection_output.png'

        HED(image_file_path, edge_output_path)

        return jsonify({'success': True, 'edge_image_path': edge_output_path})
    else:
        return jsonify({'success': False, 'error': 'No image path provided for edge detection'})


@app.route('/apply_theme', methods=['POST'])
def apply_theme():
    selected_theme = request.form.get('theme')
    original_image_path = request.form.get('image_path')

    theme_style_map = {
        'Abstract': 'abstract.png',
        'Geometric': 'geometric.png',
        'Pop art': 'popart.png',
        'Watercolor': 'watercolor1.png',
        'Oil Painting': 'oilpainting.png',
        'None': None
    }

    if selected_theme and original_image_path:
        original_image_file_path = original_image_path.replace('/static', 'static')
        theme_filename = theme_style_map.get(selected_theme)

        if theme_filename:
            theme_path = os.path.join(app.config['STYLE_FOLDER'], theme_filename)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'style_output.png')

            style_transfer(original_image_file_path, theme_path, output_path)

            return jsonify(
                {'success': True, 'styled_image_path': url_for('static', filename=f'outputs/style_output.png')})
        else:
            return jsonify(
                {'success': True, 'styled_image_path': original_image_path})
    else:
        return jsonify({'success': False, 'error': 'No theme or image path provided'})

@app.route('/merge_images', methods=['POST'])
def merge_images():
    original_image_path = request.form.get('original_image_path')
    second_image_path = request.form.get('second_image_path')

    if original_image_path and second_image_path:
        original_image_file_path = original_image_path.replace('/static', 'static')
        second_image_file_path = second_image_path.replace('/static', 'static')

        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'style_output2.png')

        style_transfer(original_image_file_path, second_image_file_path, output_path)

        return jsonify({'success': True, 'merged_image_path': url_for('static', filename=f'outputs/style_output2.png')})
    else:
        return jsonify({'success': False, 'error': 'Missing image paths'})

@app.route('/inpaint', methods=['POST'])
def perform_inpainting():
    data = request.json
    image_path = data.get('image_path').replace("/static", "static")
    prompt = data.get('prompt')
    x = int(data.get('x'))
    y = int(data.get('y'))
    width = int(data.get('width'))
    height = int(data.get('height'))
    if image_path and prompt:
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'inpainted_output.png')
        inpainting(image_path, prompt, x, y, width, height, output_path)
        return jsonify({'success': True, 'processed_image_path': url_for('static', filename='outputs/inpainted_output.png')})
    else:
        return jsonify({'success': False, 'error': 'Invalid input for inpainting'})


if __name__ == '__main__':
    app.run(debug=True)



