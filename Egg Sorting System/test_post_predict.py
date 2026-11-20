import requests

url = "http://127.0.0.1:8000/api/v1/predict"
img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'

try:
    with open(img_path, 'rb') as f:
        files = [('images', ('egg.jpg', f, 'image/jpeg'))]
        resp = requests.post(url, files=files, timeout=30)
        print("Status code:", resp.status_code)
        print("Response JSON:", resp.json())
except Exception as e:
    print("Error:", e)
