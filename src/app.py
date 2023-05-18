from flask import Flask, render_template, request
import base64
import re
import pickle

app = Flask(__name__)

# with open('visualize_embeddings.pkl', 'rb') as file:
#     visualize_embeddings = pickle.load(file)

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

    # image_path = 'uploaded_image.png'
    # processed_image = visualize_embeddings.preprocess_image(image_path)
    

    
    # visualize_embeddings(processed_image)
    # plot_path = 'plot.png'
    # plt.savefig(plot_path)


if __name__ == '__main__':
    app.run()
