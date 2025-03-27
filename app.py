import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Keras model - simplified
model = tf.keras.models.load_model('model/mnist_model.h5')
print("✅ Model loaded successfully")

@app.route('/health')
def health_check():
    return "OK", 200

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        try:
            # Process image
            img = Image.open(file.stream).convert('L').resize((28, 28))
            img_array = (np.array(img) / 255.0).astype(np.float32).reshape(1, 28, 28)
            
            # Predict
            prediction = model.predict(img_array, batch_size=1)
            digit = int(np.argmax(prediction))
            
            # Save file
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return render_template('index.html', 
                               prediction=digit,
                               filename=filename,
                               show_result=True)
                               
        except Exception as e:
            print(f"Error: {e}")
            return redirect(request.url)
    
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)