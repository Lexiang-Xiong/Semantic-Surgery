## NOTE:Do not use this detector in a deployed environment
## This detector is only used to illustrate that any generic detector can be used for visual feedback

import requests
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from PIL import Image
from typing import List, Tuple

def image_to_bytes(img: Image.Image, format: str = 'JPEG') -> bytes:
    img_byte_array = BytesIO()
    img.save(img_byte_array, format=format)
    img_byte_array.seek(0)
    return img_byte_array.getvalue()

def detect_AOD(img, concept: str, api_key: str, url: str) -> float:
    """single request"""
    files = {"image": img}
    data = {"prompts": [concept], "model": "agentic"}
    headers = {"Authorization": api_key}
    
    try:
        response = requests.post(url, files=files, data=data, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()['data'][0]
        return max([item['score'] for item in data]) if data else 0
    except Exception as e:
        print(f"detection failed: {str(e)}")
        return 0

class ConcurrentObjectDetection:
    def __init__(self, 
                 threshold: float = 0.5,
                 concepts: List[str] = None, 
                 api_key_pool: List[str] = None,
                 url: str = "",
                 max_workers: int = 5,
                 image_format: str = 'JPEG'):
        self.concepts = concepts or []
        self.api_key_pool = api_key_pool or []
        self.url = url
        self.threshold = threshold
        self.max_workers = max_workers 
        self.image_format = image_format

    def _process_concept(self, img, concept: str) -> Tuple[int, float]:
        """single concept detection"""
        api_key = f"Basic {random.choice(self.api_key_pool)}"
        score = detect_AOD(img, concept, api_key, self.url)
        return self.concepts.index(concept), score

    def __call__(self, images_steps, params) -> Tuple[float, List[int]]:
        detected_concept = []
        scores = []
        pil_image = images_steps[0][0]
        img = image_to_bytes(pil_image, format=self.image_format)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_concept, 
                    img, 
                    concept
                ): concept for concept in self.concepts
            }
            
            for future in as_completed(futures):
                try:
                    idx, score = future.result()
                    if score > self.threshold:
                        detected_concept.append(idx)
                        scores.append(score)
                except Exception as e:
                    print(f"concept detection error: {str(e)}")

        return (max(scores) if scores else 0, detected_concept)