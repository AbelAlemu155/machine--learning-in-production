from flask import Flask, request, jsonify, render_template
from analyze import get_llm_response

app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/api/v1/analyze", methods=['POST'])
def analyze():
    try:
        image_data = request.get_data(cache=False)
        llm_response = get_llm_response(image_data)
        response_data = {
            "text": llm_response
        }
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving response from LLM. Error: {e}'}), 500




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)