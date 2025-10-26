from flask import Flask, render_template, request, Markup, jsonify
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pickle
import urllib.request
import urllib.parse
import json
import re
import random

classifier = load_model('Trained_model.h5')
classifier._make_predict_function()

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)

@ app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():

    crop_name = str(request.form['cropname'])
    N_filled = int(request.form['nitrogen'])
    P_filled = int(request.form['phosphorous'])
    K_filled = int(request.form['potassium'])

    df = pd.read_csv('Data/Crop_NPK.csv')

    N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
    P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
    K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = N_desired- N_filled
    p = P_desired - P_filled
    k = K_desired - K_filled

    if n < 0:
        key1 = "NHigh"
    elif n > 0:
        key1 = "Nlow"
    else:
        key1 = "NNo"

    if p < 0:
        key2 = "PHigh"
    elif p > 0:
        key2 = "Plow"
    else:
        key2 = "PNo"

    if k < 0:
        key3 = "KHigh"
    elif k > 0:
        key3 = "Klow"
    else:
        key3 = "KNo"

    abs_n = abs(n)
    abs_p = abs(p)
    abs_k = abs(k)

    response1 = Markup(str(fertilizer_dict[key1]))
    response2 = Markup(str(fertilizer_dict[key2]))
    response3 = Markup(str(fertilizer_dict[key3]))
    return render_template('Fertilizer-Result.html', recommendation1=response1,
                           recommendation2=response2, recommendation3=response3,
                           diff_n = abs_n, diff_p = abs_p, diff_k = abs_k)


def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = classifier.predict_classes(test_image)
        return result
    except:
        return 'x'

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # fetch input
        filename = file.filename

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
        if pred == 'x':
            return render_template('unaptfile.html')
        if pred[0] == 0:
            pest_identified = 'aphids'
        elif pred[0] == 1:
            pest_identified = 'armyworm'
        elif pred[0] == 2:
            pest_identified = 'beetle'
        elif pred[0] == 3:
            pest_identified = 'bollworm'
        elif pred[0] == 4:
            pest_identified = 'earthworm'
        elif pred[0] == 5:
            pest_identified = 'grasshopper'
        elif pred[0] == 6:
            pest_identified = 'mites'
        elif pred[0] == 7:
            pest_identified = 'mosquito'
        elif pred[0] == 8:
            pest_identified = 'sawfly'
        elif pred[0] == 9:
            pest_identified = 'stem borer'

        return render_template(pest_identified + ".html",pred=pest_identified)

@ app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/'+final_prediction+'.jpg')


# Weather route using OpenWeatherMap
@app.route('/weather', methods=['GET'])
def weather():
    city = request.args.get('city', '')
    weather_data = None
    error_message = None
    if city:
        try:
            api_key = os.environ.get('OPENWEATHER_API_KEY', '')
            if not api_key:
                error_message = 'Missing OPENWEATHER_API_KEY environment variable.'
            else:
                params = {
                    'q': city,
                    'appid': api_key,
                    'units': 'metric'
                }
                url = 'https://api.openweathermap.org/data/2.5/weather?' + urllib.parse.urlencode(params)
                with urllib.request.urlopen(url, timeout=10) as response:
                    import json as _json
                    weather_data = _json.loads(response.read().decode('utf-8'))
        except Exception as e:
            error_message = 'Unable to fetch weather data.'
    return render_template('weather.html', city=city, weather=weather_data, error=error_message)


# JSON Weather API for floating widget
@app.route('/weather_api', methods=['GET'])
def weather_api():
    city = request.args.get('city', '').strip()
    if not city:
        return jsonify({ 'error': 'Missing city' }), 400
    try:
        api_key = os.environ.get('OPENWEATHER_API_KEY', '')
        if not api_key:
            return jsonify({ 'error': 'Weather API not configured' }), 500
        params = { 'q': city, 'appid': api_key, 'units': 'metric' }
        url = 'https://api.openweathermap.org/data/2.5/weather?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        # reduce payload
        out = {
            'name': data.get('name'),
            'condition': (data.get('weather') or [{}])[0].get('main'),
            'description': (data.get('weather') or [{}])[0].get('description'),
            'temp': (data.get('main') or {}).get('temp'),
            'feels_like': (data.get('main') or {}).get('feels_like'),
            'humidity': (data.get('main') or {}).get('humidity'),
            'wind': (data.get('wind') or {}).get('speed')
        }
        return jsonify(out)
    except Exception:
        return jsonify({ 'error': 'Unable to fetch weather data' }), 500

# Simple rule-based chatbot endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.get_json(silent=True) or {}
        message = (data.get('message') or '').strip().lower()
        # Default opening
        default_openers = [
            "I'm here to help! Ask about crops, fertilizers, pesticides, or weather.",
            "Hi! I can guide you on crop, fertilizer, pesticide and weather info.",
            "Hello! Need crop advice, fertilizer balance, pest ID or weather?"
        ]
        if not message:
            return jsonify({ 'reply': random.choice(default_openers) })

        # Try OpenAI (if API key present) for general Q&A
        openai_key = os.environ.get('OPENAI_API_KEY', '').strip()
        if openai_key:
            try:
                payload = {
                    'model': 'gpt-4o-mini',
                    'temperature': 0.3,
                    'messages': [
                        { 'role': 'system', 'content': 'You are Growlytics Assistant. Be concise, helpful, and accurate. You can answer general questions and also guide users through crop, fertilizer, pesticide, and weather features of the app.' },
                        { 'role': 'user', 'content': message }
                    ]
                }
                req = urllib.request.Request(
                    url='https://api.openai.com/v1/chat/completions',
                    data=json.dumps(payload).encode('utf-8'),
                    headers={
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {openai_key}'
                    },
                    method='POST'
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    res = json.loads(resp.read().decode('utf-8'))
                    content = (res.get('choices') or [{}])[0].get('message', {}).get('content')
                    if content:
                        return jsonify({ 'reply': content.strip() })
            except Exception:
                # Fall back to local intents
                pass

        # Greetings
        if re.search(r"\b(hi|hello|hey|namaste|hola)\b", message):
            return jsonify({ 'reply': random.choice([
                'Hello! How can I assist you with Growlytics today?',
                'Hi there! What would you like help with?',
                'Hey! Need crop, fertilizer, pesticide or weather help?'
            ]) })

        # Thanks / closing
        if re.search(r"\b(thanks|thank you|ty)\b", message):
            return jsonify({ 'reply': random.choice(['You\'re welcome!', 'Glad to help!', 'Anytime!']) })
        if re.search(r"\b(bye|goodbye|see ya)\b", message):
            return jsonify({ 'reply': random.choice(['Goodbye! 👋', 'See you later!', 'Take care!']) })

        # Weather intent with optional city
        if 'weather' in message:
            m = re.search(r"weather (in|at|for)\s+([a-zA-Z\s]+)$", message)
            if m:
                city = m.group(2).strip()
                return jsonify({ 'reply': f"Here you go: /weather?city={urllib.parse.quote(city)}" })
            return jsonify({ 'reply': 'Open the Weather page and enter your city, or say “weather in Mumbai”.' })

        # Crop recommendation intent
        if 'crop' in message and ('recommend' in message or 'which' in message or 'best' in message):
            return jsonify({ 'reply': 'Use Crop Recommendation. Provide N, P, K, temperature, humidity, pH, and rainfall for a suggestion.' })

        # Fertilizer recommendation intent
        if 'fertilizer' in message or re.search(r"\bnpk\b", message):
            # If numbers present, hint about deficits
            if re.search(r"\d", message):
                return jsonify({ 'reply': 'Provide exact N, P, K values in Fertilizer Recommendation to get tailored advice.' })
            return jsonify({ 'reply': 'Open Fertilizer Recommendation and enter current N, P, K for guidance.' })

        # Pesticide / Pest
        if 'pesticide' in message or 'pest' in message:
            return jsonify({ 'reply': 'Use Pesticide Recommendation: upload a clear pest image to identify and get advice.' })

        # Help intent
        if 'help' in message or 'how' in message:
            return jsonify({ 'reply': 'You can say: “weather in Delhi”, “recommend crop”, “balance fertilizer N=50 P=30 K=20”, or “identify pest”.' })

        # Reflective fallback with partial echo
        snippet = (message[:60] + '…') if len(message) > 60 else message
        return jsonify({ 'reply': f"Got it about '{snippet}'. Ask me about weather, crop, fertilizer or pesticide for more specific help." })
    except Exception:
        return jsonify({ 'reply': 'Sorry, something went wrong processing your message.' }), 500


# Chat page
@app.route('/chat', methods=['GET'])
def chat_page():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)