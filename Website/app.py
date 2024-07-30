from flask import Flask, request, render_template, redirect, url_for, jsonify, session
import flask_cors
from flask_cors import CORS

import numpy as np
import pickle

app = Flask(__name__)
CORS(app)
app.secret_key = '123'

from keras.models import load_model
import os
os.chdir('/Users/utsav/Documents/PCL-2/Code/Website')

# Ensure the model path is correct
model_path = 'crop_recommendation_model.keras'
model = load_model(model_path)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()
        state_input = data['state']
        nitrogen = float(data['nitrogen'])
        potassium = float(data['potassium'])
        phosphorus = float(data['phosphorus'])
        ph = float(data['ph'])
        temperature = float(data['temperature'])
        rainfall = float(data['rainfall'])
        humidity = float(data['humidity'])

        # Ensure the order of features matches the model's training data
        input_features = np.array([[nitrogen, potassium, phosphorus, temperature, humidity, ph, rainfall]])
        prediction = model.predict(input_features)

        labels = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
                  'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
                  'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
                  'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

        top_20_crop_indices = np.argsort(prediction, axis=1)[0][-20:][::-1]
        top_20_crops = [labels[index] for index in top_20_crop_indices]


        states_dict = {
            "Andhra Pradesh":[  {"crop": "Rice", "picture": "rice.jpeg", "text": "Rice is a staple food crop."},
        {"crop": "Maize", "picture": "maize.jpeg", "text": "Maize is used for various purposes including human consumption, animal feed, and industrial uses."},  {"crop": "Soyabean", "picture": "rice.jpeg", "text": "Rice is a staple food crop."},
        {"crop": "Coffee", "picture": "maize.jpeg", "text": "Maize is used for various purposes including human consumption, animal feed, and industrial uses."} ],
            "Arunachal Pradesh": ["Rice", "Maize", "Millet", "Ginger", "Chillies", "Oilseeds", "Orange"],
            "Assam": ["Tea", "Rice", "Maize", "Jute", "Pulses", "Oilseeds", "Sugarcane"],
            "Bihar": ["Rice", "Wheat", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Jute"],
            "Chhattisgarh": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Jute", "Tobacco"],
            "Goa": ["Rice", "Coconut", "Cashew nuts", "Arecanut"],
            "Gujarat": ["Cotton", "Groundnut", "Castor", "Bajra", "Tur", "Green gram", "Sesamum", "Paddy", "Maize", "Sugarcane"],
            "Haryana": ["Wheat", "Rice", "Sugarcane", "Barley", "Gram", "Sunflower", "Rapeseed", "Mustard", "Cotton"],
            "Himachal Pradesh": ["Maize", "Wheat", "Barley", "Rice", "Apple", "Citrus fruits", "Stone fruits", "Tobacco"],
            "Jharkhand": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane"],
            "Karnataka": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Coffee", "Rubber", "Tea", "Cashews", "Cardamom", "Chillies"],
            "Kerala": ["Coconut", "Rubber", "Coffee", "Pepper", "Cashewnuts", "Ginger", "Turmeric", "Tea", "Cardamom", "Cinnamon"],
            "Madhya Pradesh": ["Wheat", "Rice", "Gram", "Maize", "Soyabean", "Pulses", "Oilseeds", "Cotton"],
            "Maharashtra": ["Rice", "Jowar", "Bajra", "Maize", "Wheat", "Pulses", "Oilseeds", "Sugarcane", "Cotton", "Grapes"],
            "Manipur": ["Rice", "Maize", "Pulses", "Fruits", "Vegetables", "Spices", "Orange"],
            "Meghalaya": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices"],
            "Mizoram": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices"],
            "Nagaland": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices"],
            "Odisha": ["Rice", "Pulses", "Oilseeds", "Sugarcane", "Jute", "Cotton", "Tobacco", "Mango", "Papaya"],
            "Punjab": ["Wheat", "Rice", "Maize", "Barley", "Gram", "Mustard", "Sugarcane", "Cotton"],
            "Rajasthan": ["Wheat", "Barley", "Maize", "Millets", "Pulses", "Oilseeds", "Cotton", "Sugarcane", "Mango", "Pomegranate"],
            "Sikkim": ["Rice", "Maize", "Barley", "Buckwheat", "Potatoes", "Large cardamom", "Ginger", "Fruits", "Vegetables", "Orange"],
            "Tamil Nadu": ["Rice", "Jowar", "Bajra", "Maize", "Ragi", "Pulses", "Oilseeds", "Sugarcane", "Coconut", "Groundnut", "Cotton", "Coffee", "Papaya"],
            "Telangana": ["Rice", "Maize", "Pulses", "Oilseeds", "Sugarcane", "Chillies", "Turmeric", "Tobacco", "Cotton", "Mango", "Pomegranate"],
            "Tripura": ["Rice", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Rubber"],
            "Uttar Pradesh": ["Wheat", "Rice", "Sugarcane", "Barley", "Gram", "Pulses", "Oilseeds", "Cotton", "Mango", "Papaya"],
            "Uttarakhand": ["Rice", "Maize", "Wheat", "Barley", "Millets", "Pulses", "Oilseeds", "Fruits", "Vegetables"],
            "West Bengal": ["Rice", "Jute", "Sugarcane", "Wheat", "Maize", "Pulses", "Oilseeds", "Fruits", "Vegetables", "Spices", "Mango", "Orange", "Banana", "Papaya"] 
        }

        crop_list = []
        if state_input in states_dict:
            state_crops = [crop.lower() for crop in states_dict[state_input]]
            for crop in top_20_crops:
                if crop in state_crops:
                    crop_list.append(crop)

# Store results in session
        session['recommendedCrops'] = crop_list  # Assuming crop_list is defined as per your processing logic
        return jsonify({'redirect': url_for('results')})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/results')
def results():
    # Ensure you have a session key check if needed
    crops = session.get('recommendedCrops', [])
    return render_template('result.html', crops=crops)

if __name__ == '__main__':
    app.run(debug=True)

#%%
