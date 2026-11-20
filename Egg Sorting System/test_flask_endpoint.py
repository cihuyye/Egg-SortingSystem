import sys
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import io
from app_flask import app

client = app.test_client()

img_path = r'C:/Users/THOR/.gemini/antigravity/brain/74a4bd7b-f41c-46ec-8fd3-481fe8d98149/.user_uploaded/media_1795099443896_525519d8.jpg'

with open(img_path, 'rb') as f:
    data = {
        'images': (io.BytesIO(f.read()), 'egg.jpg', 'image/jpeg')
    }
    response = client.post('/api/v1/predict', data=data, content_type='multipart/form-data')
    print("Response status:", response.status_code)
    json_data = response.get_json()
    print("Response is_egg:", json_data.get('is_egg'))
    print("Response final_grade:", json_data.get('final_grade'))
    print("Response details:", json_data)
