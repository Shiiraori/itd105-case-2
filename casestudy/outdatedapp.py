from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")


@app.route("/todo")
def todo():
	return render_template("todo.html")

@app.route("/classification")
def classification():
	return render_template("classification.html")


model = joblib.load('model.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make a prediction
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]

    # Map the numeric prediction to the class name
    class_name = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    result = class_name[prediction]

    return render_template('classification.html', result=result)

@app.route("/regression")
def regression():
	return render_template("regression.html")

@app.route("/test")
def test():
	return render_template("test.html")

@app.route('/testpredict', methods=['POST'])
def testpredict():
    # Get input data from the form
    features = {
        'age': float(request.form['age']),
        'anaemia': int(request.form['anaemia']),
        'creatinine_phosphokinase': int(request.form['creatinine_phosphokinase']),
        'diabetes': int(request.form['diabetes']),
        'ejection_fraction': int(request.form['ejection_fraction']),
        'high_blood_pressure': int(request.form['high_blood_pressure']),
        'platelets': float(request.form['platelets']),
        'serum_creatinine': float(request.form['serum_creatinine']),
        'serum_sodium': int(request.form['serum_sodium']),
        'sex': int(request.form['sex']),
        'smoking': int(request.form['smoking'])
    }

    # Convert features to a DataFrame
    input_data = pd.DataFrame([features])

    # Make a prediction
    prediction = model.predict(input_data)[0]
    #input_data = input_data[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking']]
    
    return render_template('test.html', prediction=prediction)


if __name__ == '__main__':
	app.run(debug=True)