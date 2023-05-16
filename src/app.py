from flask import Flask, render_template, request
import base64
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['imageData']

    # Extract base64 data from the request
    image_data = re.sub('^data:image/.+;base64,', '', image_data)

    # Convert base64 data to bytes
    image_bytes = base64.b64decode(image_data)

    # Save the image to a file
    with open('uploaded_image.png', 'wb') as f:
        f.write(image_bytes)

    return 'Image uploaded successfully!'

if __name__ == '__main__':
    app.run()
