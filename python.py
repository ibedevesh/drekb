from flask import Flask, request, jsonify
import os
from gradientai import Gradient
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

os.environ['GRADIENT_ACCESS_TOKEN'] = "qYBtXzJmTK2MzTIpqBqcuPsQSCIfbZHO"
os.environ['GRADIENT_WORKSPACE_ID'] = "e8ee528f-553e-45ca-98b1-480af025193c_workspace"

def extract_job_and_location(query):
    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        prompt = f"Extract the job and location from the following query: '{query}'. 'job, location:"
        completion = base_model.complete(query=prompt, max_generated_token_count=100).generated_output
        return completion.strip()

def search_jobs(job, location):
    url = "https://jsearch.p.rapidapi.com/search"
    query_string = f"{job} in {location}"
    querystring = {"query": query_string, "page": "1", "num_pages": "1"}
    headers = {
        "X-RapidAPI-Key": "674b369e0dmsh1f421d3fdf28ae2p17e46cjsnce591ae3e24f",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers, params=querystring)
    return response.json()

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json.get('query')
    extracted_info = extract_job_and_location(user_query)
    parts = extracted_info.split(',')
    if len(parts) == 2:
        job = parts[0].replace("job:", "").strip()
        location = parts[1].replace("location:", "").strip()
        job_results = search_jobs(job, location)
        with Gradient() as gradient:
            base_model = gradient.get_base_model(base_model_slug="mixtral-8x7b-instruct")
            prompt = f"User query: '{user_query}'. Based on the following job data, provide a detailed response: {job_results}"
            completion = base_model.complete(query=prompt, max_generated_token_count=511).generated_output
        response = completion.strip()
    else:
        response = "Could not extract job and location. Please try again."
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
