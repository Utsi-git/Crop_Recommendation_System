import requests
import json
import os

crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
         'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
         'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
         'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

os.makedirs('static/images/crops', exist_ok=True)

mapping = {
    'blackgram': 'Vigna_mungo',
    'kidneybeans': 'Kidney_bean',
    'pigeonpeas': 'Pigeon_pea',
    'mothbeans': 'Vigna_aconitifolia',
    'mungbean': 'Mung_bean',
    'chickpea': 'Chickpea',
    'muskmelon': 'Muskmelon',
    'lentil': 'Lentil',
    'jute': 'Jute',
    'maize': 'Maize',
    'coffee': 'Coffee',
    'cotton': 'Cotton',
    'rice': 'Rice',
    'watermelon': 'Watermelon'
}

def download_image(crop_name, file_name):
    query = mapping.get(crop_name, crop_name.capitalize())
    search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(search_url, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if 'originalimage' in data:
                img_url = data['originalimage']['source']
            elif 'thumbnail' in data:
                img_url = data['thumbnail']['source']
            else:
                return False
                
            img_data = requests.get(img_url, headers=headers).content
            # just save as .jpg for simplicity, some might be png but browser handles it normally if we don't declare explicitly, though better keep original extension.
            ext = img_url.split('.')[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png']:
                ext = 'jpg'
            
            with open(f"static/images/crops/{file_name}.jpg", "wb") as f:
                f.write(img_data)
            print(f"Downloaded {file_name}")
            return True
    except Exception as e:
        print(f"Failed {file_name}: {e}")
    return False

for crop in crops:
    if not os.path.exists(f"static/images/crops/{crop}.jpg"):
        res = download_image(crop, crop)
        if not res:
            print(f"Could not get image for {crop}")
