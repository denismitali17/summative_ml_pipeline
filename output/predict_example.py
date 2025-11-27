"""
Simple example script to POST an image to the running Flask API and print the JSON response.
Usage:
  python notebooks\predict_example.py C:\full\path\to\image.png

This is intended to be included in the notebook or README as a lightweight demonstration of the prediction API.
"""
import sys
import requests
import json

API_URL = 'http://127.0.0.1:5000/api/predict'


def predict(image_path):
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.split('\\')[-1], f, 'image/png')}
        resp = requests.post(API_URL, files=files)
    try:
        resp.raise_for_status()
        print(json.dumps(resp.json(), indent=2))
    except Exception as e:
        print('Request failed:', e)
        try:
            print('Response:', resp.text)
        except Exception:
            pass


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python notebook\\predict_example.py C:\\path\\to\\image.png')
        sys.exit(1)
    img = sys.argv[1]
    predict(img)
