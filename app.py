from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('churn_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = [
        float(request.form['tenure']),
        float(request.form['monthlycharges']),
        float(request.form['totalcharges']),
        int(request.form['seniorcitizen']),
        int(request.form['contract']),
        int(request.form['paymentmethod']),
        int(request.form['internetservice'])
    ]

    # Reshape and predict
    prediction = model.predict([np.array(input_data).reshape(1, -1)])[0]

    return render_template('result.html', prediction='Yes' if prediction == 1 else 'No')

if __name__ == '__main__':
    app.run(debug=True)
