"""
Send request to test serving model
"""
import requests
import imageio.v3 as iio
import pickle
import json
import time

video_path = '../videos/_q5Nwh4Z6ao_3.avi'

frames = iio.imread(video_path, extension='.avi')[:64]

arr = pickle.dumps(frames)

url = "http://127.0.0.1:5050/predict"

headers={'Content-Type': 'application/json'}

start = time.perf_counter()
response = requests.post(
    url=url,
    headers=headers,
    data=arr)

result = json.loads(response.content.decode('utf-8'))
print("Perf:", time.perf_counter() - start)
print(result)
