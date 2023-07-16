import requests
import imageio.v3 as iio
import pickle
import json

video_path = 'videos/_q5Nwh4Z6ao_3.avi'

frames = iio.imread(video_path, extension='.avi')[:64]

arr = pickle.dumps(frames)

print(type(arr))

url = "http://127.0.0.1:5050/predict"

headers={'Content-Type': 'application/json'}

response = requests.post(
    url=url,
    headers=headers,
    data=arr)

result = json.loads(response.content.decode('utf-8'))

print(result)