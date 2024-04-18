from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

model = pickle.load(open(r'C:/Users/nikhi/OneDrive/Desktop/Chandana Major Project/Training/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/findingcluster')
def findingcluster():
    return render_template('findingcluster.html')


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    # Convert form values to floats instead of integers
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output = prediction[0]
    return render_template('findingcluster.html', prediction_text='The cluster based on given user info is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
