import requests
import json
import os

url = "http://127.0.0.1:8000/api/v1/predict"
sample_img = r"D:/Egg Dataset/Egg Dataset/test/images/cracked"

files_in_dir = os.listdir(sample_img)
if not files_in_dir:
    print("Tidak ada file uji di", sample_img)
    exit(1)

test_file_path = os.path.join(sample_img, files_in_dir[0])
print(f"[TEST CLIENT] Mengirim file: {test_file_path}")

with open(test_file_path, "rb") as f:
    files = {"file": (os.path.basename(test_file_path), f, "image/jpeg")}
    response = requests.post(url, files=files)

print("\n--- STATUS CODE HTTP ---")
print(response.status_code)

print("\n--- RESPONSE JSON DARI API ---")
try:
    data = response.json()
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Gagal decode JSON:", response.text)

