from flask import Flask, request, jsonify
from inference import generate_response

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    input_text = data.get('input_text', '') 

    if not input_text:
        return jsonify({"error": "No input_text provided"}), 400

    generated_text = generate_response(input_text)

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)