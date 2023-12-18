from flask import Flask, render_template, request, jsonify, url_for, redirect
import joblib
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database file path
db = SQLAlchemy(app)
migrate = Migrate(app, db)



@app.route("/todo")
def todo():
	return render_template("todo.html")

class Classprediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float, nullable=False)
    anaemia = db.Column(db.Integer, nullable=False)
    creatinine_phosphokinase = db.Column(db.Integer, nullable=False)
    diabetes = db.Column(db.Integer, nullable=False)
    ejection_fraction = db.Column(db.Integer, nullable=False)
    high_blood_pressure = db.Column(db.Integer, nullable=False)
    platelets = db.Column(db.Float, nullable=False)
    serum_creatinine = db.Column(db.Float, nullable=False)
    serum_sodium = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)
    smoking = db.Column(db.Integer, nullable=False)
    prediction = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.route("/classification")
def classification():
	return render_template("classification.html")


model = joblib.load('model.pkl')
@app.route('/classpredict', methods=['POST'])
def classpredict():
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
    prediction = int(model.predict(input_data)[0])

    
    new_prediction = Classprediction(
        age=features['age'],
        anaemia=features['anaemia'],
        creatinine_phosphokinase=features['creatinine_phosphokinase'],
        diabetes=features['diabetes'],
        ejection_fraction=features['ejection_fraction'],
        high_blood_pressure=features['high_blood_pressure'],
        platelets=features['platelets'],
        serum_creatinine=features['serum_creatinine'],
        serum_sodium=features['serum_sodium'],
        sex=features['sex'],
        smoking=features['smoking'],
        prediction=prediction
        )
    db.session.add(new_prediction)
    db.session.commit()

    print("Raw Prediction:", prediction, flush=True)

    return render_template('classification.html', prediction=prediction)

@app.route('/classdata')
# @dt.data
def Classpredictions():
    # Retrieve all predictions from the database
    all_predictions = Classprediction.query.all()
    return render_template('classdata.html', predictions=all_predictions)

@app.route('/classdelete/<int:id>', methods=['POST'])
def classdelete_prediction(id):
    prediction = Classprediction.query.get_or_404(id)
    db.session.delete(prediction)
    db.session.commit()
    return jsonify({'message': 'Prediction deleted successfully'})

@app.route('/classedit/<int:id>', methods=['GET', 'POST'])
def classedit_prediction(id):
    prediction = Classprediction.query.get_or_404(id)

    if request.method == 'POST':
        # Update prediction with new data
        prediction.age = float(request.form['age'])
        prediction.anaemia = int(request.form['anaemia'])
        prediction.creatinine_phosphokinase = int(request.form['creatinine_phosphokinase'])
        prediction.diabetes = int(request.form['diabetes'])
        prediction.ejection_fraction = int(request.form['ejection_fraction'])
        prediction.high_blood_pressure = int(request.form['high_blood_pressure'])
        prediction.platelets = float(request.form['platelets'])
        prediction.serum_creatinine = float(request.form['serum_creatinine'])
        prediction.serum_sodium = int(request.form['serum_sodium'])
        prediction.sex = int(request.form['sex'])
        prediction.smoking = int(request.form['smoking'])

        # Convert features to a DataFrame
        input_data = pd.DataFrame([{
            'age': prediction.age,
            'anaemia': prediction.anaemia,
            'creatinine_phosphokinase': prediction.creatinine_phosphokinase,
            'diabetes': prediction.diabetes,
            'ejection_fraction': prediction.ejection_fraction,
            'high_blood_pressure': prediction.high_blood_pressure,
            'platelets': prediction.platelets,
            'serum_creatinine': prediction.serum_creatinine,
            'serum_sodium': prediction.serum_sodium,
            'sex': prediction.sex,
            'smoking': prediction.smoking
        }])

        # Make a prediction
        prediction.prediction = int(model.predict(input_data)[0])

        db.session.commit()

        if request.endpoint == '/classdata':
            return redirect('/classdata')
        else:
            return redirect('/index')

    return render_template('classedit.html', prediction=prediction)

class Regprediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    house_age = db.Column(db.Float, nullable=False)
    distance_to_MRT = db.Column(db.Float, nullable=False)
    num_convenience_stores = db.Column(db.Integer, nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@app.route("/regression")
def regression():
	return render_template("regression.html")

knn_model = joblib.load('regressionmodel.pkl')
@app.route('/regpredict', methods=['POST'])
def regpredict():
    try:
        # Get data from the form
        house_age = float(request.form['house_age'])
        distance_to_MRT = float(request.form['distance_to_MRT'])
        num_convenience_stores = int(request.form['num_convenience_stores'])

        # Make prediction
        prediction = knn_model.predict([[house_age, distance_to_MRT, num_convenience_stores]])
	
        predicted_value = prediction[0] * 1000

        new_prediction = Regprediction(
            house_age=house_age,
            distance_to_MRT=distance_to_MRT,
            num_convenience_stores=num_convenience_stores,
            predicted_value=predicted_value
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Render the prediction on the HTML page
        return render_template('regression.html', prediction=f'Predicted House Value: ${predicted_value:.2f}')

    except Exception as e:
        return render_template('regression.html', error=str(e))
    
@app.route('/regdata')
def Regpredictions():
    # Retrieve all predictions from the database
    all_predictions = Regprediction.query.all()
    return render_template('regdata.html', predictions=all_predictions)

@app.route('/regedit/<int:id>', methods=['GET', 'POST'])
def regedit_prediction(id):
    prediction = Regprediction.query.get_or_404(id)

    if request.method == 'POST':
        # Update prediction with new data
        prediction.house_age = float(request.form['house_age'])
        prediction.distance_to_MRT = float(request.form['distance_to_MRT'])
        prediction.num_convenience_stores = int(request.form['num_convenience_stores'])
        prediction.predicted_value = knn_model.predict([[prediction.house_age, prediction.distance_to_MRT, prediction.num_convenience_stores]])[0] * 1000

        db.session.commit()

        if request.endpoint == '/regdata':
            return redirect('/regdata')
        else:
            return redirect('/index')
        

    return render_template('regdata.html', prediction=prediction)

@app.route('/regdelete/<int:id>', methods=['POST'])
def regdelete_prediction(id):
    prediction = Regprediction.query.get_or_404(id)
    db.session.delete(prediction)
    db.session.commit()
    return jsonify({'message': 'Prediction deleted successfully'})


@app.route('/test')
def test():
     return render_template("test.html")

@app.route("/")
@app.route("/index")
def index():
    class_predictions = Classprediction.query.all()
    reg_predictions = Regprediction.query.all()

    return render_template("index.html", class_predictions=class_predictions, reg_predictions=reg_predictions)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)