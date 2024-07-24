from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from skimage.io import imread


def extract_features(img_path):
    img = imread(img_path)
    gray = rgb2gray(img)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_personality(features):
    np.random.seed(42)
    coefficients = np.random.rand(features.shape[0], 5)
    intercept = np.random.rand(5)
    model = LinearRegression()
    model.coef_ = coefficients.T
    model.intercept_ = intercept
    personality_traits = model.predict([features])
    return personality_traits[0]

def adjust_trait_values(responses):
    traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    values = [50, 50, 50, 50, 50]

    if responses.get('q1') == 'no':
        values[0] -= 10
    if responses.get('q2') in ['family', 'money', 'friends']:
        values[0] -= 10
    if responses.get('q8') in ['very_low']:
        values[0] -= 20
    elif responses.get('q8') in ['low']:
        values[0] -= 10
    elif responses.get('q8') in ['high']:
        values[0] += 10
    elif responses.get('q8') in ['very_high']:
        values[0] += 20
    if responses.get('q9') in ['very_low']:
        values[0] -= 20
    elif responses.get('q9') in ['low']:
        values[0] -= 10
    elif responses.get('q9') in ['high']:
        values[0] += 10
    elif responses.get('q9') in ['very_high']:
        values[0] += 20

    if responses.get('q3') in ['very_low']:
        values[1] -= 20
    elif responses.get('q3') in ['low']:
        values[1] -= 10
    elif responses.get('q3') in ['high']:
        values[1] += 10
    elif responses.get('q3') in ['very_high']:
        values[1] += 20
    if responses.get('q7') in ['very_low']:
        values[1] -= 20
    elif responses.get('q7') in ['low']:
        values[1] -= 10
    elif responses.get('q7') in ['high']:
        values[1] += 10
    elif responses.get('q7') in ['very_high']:
        values[1] += 20

    if responses.get('q4') in ['very_low']:
        values[2] -= 20
    elif responses.get('q4') in ['low']:
        values[2] -= 10
    elif responses.get('q4') in ['high']:
        values[2] += 10
    elif responses.get('q4') in ['very_high']:
        values[2] += 20
    if responses.get('q5') in ['very_low']:
        values[2] -= 20
    elif responses.get('q5') in ['low']:
        values[2] -= 10
    elif responses.get('q5') in ['high']:
        values[2] += 10
    elif responses.get('q5') in ['very_high']:
        values[2] += 20
    if responses.get('q6') in ['very_low']:
        values[2] -= 20
    elif responses.get('q6') in ['low']:
        values[2] -= 10
    elif responses.get('q6') in ['high']:
        values[2] += 10
    elif responses.get('q6') in ['very_high']:
        values[2] += 20

    if responses.get('q10') in ['very_low']:
        values[3] -= 20
    elif responses.get('q10') in ['low']:
        values[3] -= 10
    elif responses.get('q10') in ['high']:
        values[3] += 10
    elif responses.get('q10') in ['very_high']:
        values[3] += 20
    if responses.get('q11') in ['very_low']:
        values[3] -= 20
    elif responses.get('q11') in ['low']:
        values[3] -= 10
    elif responses.get('q11') in ['high']:
        values[3] += 10
    elif responses.get('q11') in ['very_high']:
        values[3] += 20

    if responses.get('q12') in ['high']:
        values[4] += 10
    elif responses.get('q12') in ['very_high']:
        values[4] += 20
    if responses.get('q13') in ['high']:
        values[4] += 10
    elif responses.get('q13') in ['very_high']:
        values[4] += 20

    values = [max(0, min(100, v)) for v in values]

    return traits, values

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/facial-image')
def facial_image():
    return render_template('facial_image.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        features = extract_features(filepath)
        values = predict_personality(features)

        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        plt.figure(figsize=(10, 5))
        plt.bar(traits, values, color='skyblue')
        plt.xlabel('Personality Traits')
        plt.ylabel('Predicted Values')
        plt.title('Predicted Personality Traits from Facial Image')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        buf.close()

        return render_template('results_image.html', plot_url=plot_url)

    return redirect(request.url)

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    responses = request.form
    traits, values = adjust_trait_values(responses)

    plt.figure(figsize=(10, 5))
    plt.bar(traits, values, color='skyblue')
    plt.xlabel('Personality Traits')
    plt.ylabel('Scores')
    plt.title('Big Five Personality Traits')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()

    return render_template('results_questionnaire.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
