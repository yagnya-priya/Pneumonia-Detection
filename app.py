from flask import Flask, render_template, request, jsonify
import sqlite3
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
try:
    model = load_model('C:/Users/kared/Downloads/VGG16-Pneumonia (1)/VGG16-Pneumonia/model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)

def init_db():
    # Connect to the database
    conn = sqlite3.connect('patients.db')
    c = conn.cursor()

    # Create patients table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    result TEXT NOT NULL,
                    probability REAL NOT NULL,
                    advice TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        file = request.files['file']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        prediction = model.predict(img_resized)
        probability = prediction[0][0]

        result = "Pneumonia" if probability <= 0.5 else "Normal"
        advice = "Please consult a doctor." if probability <= 0.5 else "No need to worry, but if symptoms persist, consider consulting a doctor."

        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        phone = request.form['phone']

        # Connect to the database
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()

        # Insert patient details into the database
        c.execute('''INSERT INTO patients (name, age, gender, phone, result, probability, advice)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''', (name, age, gender, phone, result, float(1 - probability), advice))
        conn.commit()
        conn.close()  # Close the connection

        return jsonify({'result': result, 'probability': float(1 - probability), 'advice': advice})
    except Exception as e:
        print("Error during evaluation:", e)
        return jsonify({'error': 'Error during evaluation'}), 500

@app.route('/patients')
def patients():
    try:
        # Connect to the database
        conn = sqlite3.connect('patients.db')
        c = conn.cursor()

        # Fetch all patients from the database
        c.execute('''SELECT * FROM patients''')
        patients = c.fetchall()

        conn.close()  # Close the connection

        return render_template('patients.html', patients=patients)
    except Exception as e:
        print("Error fetching patients:", e)
        return jsonify({'error': 'Error fetching patients'}), 500

if __name__ == '__main__':
    init_db()  # Initialize the database
    app.run(debug=True)
