import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TFLite model - simplified
interpreter = tf.lite.Interpreter(model_path='model/mnist.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/health')
def health_check():
    return "OK", 200

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('home'))
            
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('home'))
        
        try:
            # Process and predict
            img = Image.open(file.stream).convert('L').resize((28, 28))
            img_array = (np.array(img) / 255.0).astype(np.float32).reshape(1, 28, 28)
            
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            digit = np.argmax(interpreter.get_tensor(output_details[0]['index']))
            
            return render_template('index.html', prediction=int(digit))
            
        except Exception as e:
            print(f"Error: {e}")
            return redirect(url_for('home'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))