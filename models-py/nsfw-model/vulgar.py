from flask import Flask, request, jsonify
from better_profanity import profanity

app = Flask(__name__)

# Load profanity words
profanity.load_censor_words()

@app.route('/censor', methods=['POST'])
def censor_text():
    data = request.get_json()
    if 'message' not in data:
        return jsonify({'error': 'Message field is required'}), 400
    
    censored_message = profanity.censor(data['message'])
    return jsonify({'original': data['message'], 'censored': censored_message})

if __name__ == '__main__':
    app.run(debug=True)
