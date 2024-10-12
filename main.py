from flask import Flask, request, jsonify
import joblib
import re
from flashtext import KeywordProcessor
import numpy as np

app = Flask(__name__)

# Health check route
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"message": "API is running"}), 200

# Disease prediction route
@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        # Get the symptoms query from the request body
        data = request.get_json()
        query = data.get("symptoms", "")

        # Call the prediction model
        predicted_disease = predict_disease_saved_model(query)

        # Return the prediction result as a JSON response
        return jsonify({"disease": predicted_disease}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def predict_disease_saved_model(query):
    # Load the saved model
    gbm = joblib.load('model.pkl')
    symptoms = joblib.load('symptoms.pkl')
    features = joblib.load('features.pkl')
    feature_dict = joblib.load('feature_dict.pkl')

    keyword_processor = KeywordProcessor()
    keyword_processor.add_keywords_from_list(symptoms)

    matched_keyword = keyword_processor.extract_keywords(query)
    if len(matched_keyword) == 0:
        return "GOOD NEWS! No disease detected. either you didn't mention any symptoms or you have mentioned all of them."
    else:
        regex = re.compile(' ')
        processed_keywords = [i if regex.search(i) is None else i.replace(' ', '_') for i in matched_keyword]
        print(processed_keywords)

        coded_features = []
        for keyword in processed_keywords:
            coded_features.append(feature_dict[keyword])

        sample_x = []
        for i in range(len(features)):
            try:
                sample_x.append(i / coded_features[coded_features.index(i)])
            except:
                sample_x.append(i * 0)

        sample_x = np.array(sample_x).reshape(1, len(sample_x))
        return gbm.predict(sample_x)[0]

if __name__ == '__main__':
    app.run(debug=True)
