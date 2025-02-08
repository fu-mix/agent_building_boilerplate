from flask import Flask, request, jsonify
from dotenv import load_dotenv
import time
from flask_cors import CORS
import requests
load_dotenv()
import utils
import keys

app = Flask(__name__)
CORS(app)

MODEL = "gpt-4o"

message_history = utils.construct_messages("You are a helpful AI assistant who provides travel research and reccomendations")

@app.route('/')
def home():
    return "Server is running"

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    messages = data['messages']
    print("Incoming messages", messages)
    time.sleep(2)
    # 最新のメッセージのcontentを取得
    input_text = messages[-1]['content']
    response = utils.run_text_prompt_with_history(messages, input_text)
    print(response)
    return jsonify({
        "role": "assistant",
        "content": response
    })


# Replace with your actual SerpAPI key
SERP_API_KEY = keys.serp_api_key

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')  # Get query from request
    if not query:
        return jsonify({"error": "Missing search query"}), 400

    serp_url = "https://serpapi.com/search"
    params = {
        "q": query,
        "api_key": SERP_API_KEY,
        "hl": "en",  # Language
        "gl": "us",  # Country code
    }

    response = requests.get(serp_url, params=params)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": "Failed to fetch search results"}), response.status_code

if __name__ == '__main__':
    # Debug mode should be set to False in production
    app.run(debug=True, port=8000)
