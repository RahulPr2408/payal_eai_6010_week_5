import os
from flask import Flask, render_template, request, flash
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for flash messages

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model with error handling
try:
    model = load_model('model/mnist_model.h5')
    print(" Model loaded successfully")
except Exception as e:
    print(f" Model loading failed: {str(e)}")
    model = None

def preprocess_image(image_path):
    """Convert uploaded image to MNIST format with error handling"""
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))                # Resize to 28x28
        img_array = np.array(img) / 255.0         # Normalize pixel values
        return img_array.reshape(1, 28, 28)       # Add batch dimension
    except Exception as e:
        print(f"Image processing error: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Verify model is loaded
            if model is None:
                flash('Model not loaded - contact administrator')
                return redirect(request.url)
                
            # Process image
            img_array = preprocess_image(filepath)
            if img_array is None:
                flash('Invalid image format - please upload a valid image')
                return redirect(request.url)
            
            # Make prediction
            prediction = model.predict(img_array)
            digit = int(np.argmax(prediction))
            
            return render_template('index.html', 
                                filename=filename, 
                                prediction=digit,
                                success=True)
                                
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    
    return render_template('index.html', success=False)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)  