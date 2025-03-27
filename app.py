import os
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf

# Initialize Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TensorFlow Lite model
try:
    interpreter = tf.lite.Interpreter(model_path='model/mnist.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("✅ TensorFlow Lite model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    interpreter = None

def preprocess_image(file_stream):
    """Process image directly from file stream"""
    try:
        img = Image.open(file_stream).convert('L').resize((28, 28))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array.reshape(1, 28, 28)
    except Exception as e:
        logger.error(f"❌ Image processing error: {str(e)}")
        return None

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return "OK", 200

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        try:
            if interpreter is None:
                flash('Model not loaded - contact administrator')
                return redirect(request.url)
                
            img_array = preprocess_image(file.stream)
            if img_array is None:
                flash('Invalid image format')
                return redirect(request.url)
            
            # TensorFlow Lite prediction
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            digit = int(np.argmax(prediction))
            
            # Save file only if needed
            if app.config['UPLOAD_FOLDER']:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return render_template('index.html', 
                               filename=file.filename,
                               prediction=digit,
                               success=True)
                               
        except Exception as e:
            logger.exception("Prediction failed")
            flash('Server error during prediction')
            return redirect(request.url)
    
    return render_template('index.html', success=False)

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"500 Error: {str(e)}")
    return render_template('index.html', 
                         error_message="Internal server error", 
                         success=False), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)