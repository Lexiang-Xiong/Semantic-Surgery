import requests
import threading
import json
import os
import cv2
import pandas as pd
import random
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


api_url = "https://api.va.landing.ai/v1/tools/agentic-object-detection"
API_POOL = ["get your api from https://landing.ai/agentic-object-detection"]


def draw_boxes_and_save(image_path: str,
                     result: Dict,
                     save_path: str = None,
                     font_scale: float = 0.8,
                     thickness: int = 2,
                     color_map: Dict = None) -> np.ndarray:
    """
    Draw detection boxes using OpenCV (suitable for video streams/BGR processing).

    Args:
    - image_path: Path to the original image.
    - result: Dictionary containing detection results.
    - save_path: Path to save the image (None if not saving).
    - font_scale: Font scale.
    - thickness: Line/font thickness.
    - color_map: Custom color map {label: BGR color}.

    Returns:
    - NumPy array in BGR format with bounding boxes drawn.
    """
    # Default color map (extendable)
    default_colors = {
        'airplane': (0, 255, 0),   # Green
        'automobile': (0, 0, 255), # Red
        'default': (255, 0, 0)     # Blue
    }
    color_map = color_map or default_colors

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Parse detection results
    try:
        detections = result['data'][0]  # Get the first set of detection results
    except (KeyError, IndexError):
        return image

    # Draw each detection box
    for det in detections:
        label = det['label']
        score = det.get('score', 1.0)
        bbox = list(map(int, det['bounding_box']))  # Convert coordinates to integers

        # Get color
        color = color_map.get(label.lower(), color_map['default'])

        # Draw rectangle box
        cv2.rectangle(
            image,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color=color,
            thickness=thickness
        )

        # Construct label text
        text = f"{label} {score:.2f}" if 'score' in det else label

        # Calculate text position
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_org = (bbox[0], bbox[1] - 10 if bbox[1] > 20 else bbox[1] + 20)

        # Draw text background
        cv2.rectangle(
            image,
            (bbox[0], text_org[1] - text_h),
            (bbox[0] + text_w, text_org[1] + 10),
            color=color,
            thickness=cv2.FILLED
        )

        # Draw text
        cv2.putText(
            image,
            text,
            org=text_org,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),  # White font
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

    # Save the result
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, image)

    return image


# ==================== Cache Management Module ====================
class DetectionCache:
    def __init__(self, cache_path: str = "detection_cache.json"):
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()
        self.lock = threading.Lock()

    def _load_cache(self) -> Dict:
        """Load or initialize the cache"""
        try:
            if self.cache_path.exists():
                with open(self.cache_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load cache: {str(e)}")
        return {}

    def save_cache(self):
        """Save the cache to a file"""
        with self.lock:
            try:
                temp_path = self.cache_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                temp_path.replace(self.cache_path)
            except Exception as e:
                print(f"Failed to save cache: {str(e)}")

    def get_task_status(self, img_path: str) -> Optional[dict]:
        """Get task status"""
        return self.cache.get(img_path)

    def mark_task_start(self, img_path: str):
        """Mark task as started"""
        self.cache[img_path] = {
            'status': 'processing',
            'timestamp': time.time(),
            'retries': 0
        }
        self.save_cache()

    def mark_task_complete(self, img_path: str, result: dict):
        """Mark task as completed"""
        self.cache[img_path] = {
            'status': 'completed',
            'result': result,
            'timestamp': time.time()
        }
        self.save_cache()

    def mark_task_failed(self, img_path: str, error: str):
        """Mark task as failed"""
        record = self.cache.get(img_path, {})
        record.update({
            'status': 'failed',
            'error': error,
            'retries': record.get('retries', 0) + 1,
            'timestamp': time.time()
        })
        self.cache[img_path] = record
        self.save_cache()

# ==================== Exception Definitions ====================
class InvalidResponseError(Exception):
    """Custom invalid response exception"""
    pass

# ==================== Core Function Enhancement ====================
@retry(
    wait=wait_exponential(multiplier=1, min=15, max=100),  # Extend wait time
    stop=stop_after_attempt(5),  # Maximum number of retries
    retry=retry_if_exception_type((requests.RequestException, InvalidResponseError))
)
def safe_single_detection(
    image_path: str,
    prompts: List[str],
    api_key: str,
    cache: DetectionCache,
    model: str = "agentic",
    timeout: int = 30,
    save_detections: bool = False,
) -> Tuple[str, dict]:
    """
    Enhanced single detection request
    """
    # Check cache
    cached = cache.get_task_status(image_path)
    if cached and cached['status'] == 'completed':
        print(f"Skipping completed task: {image_path}")
        return (image_path, cached['result'])

    # Mark task as started
    cache.mark_task_start(image_path)

    url = api_url
    result = {}
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"prompts": prompts,
                    "model": model}
            headers = {"Authorization": f"Basic {api_key}"}
            # Define proxies
            # proxies = {
            #     'http': 'http://127.0.0.1:7890',
            #     'https': 'http://127.0.0.1:7890',
            # }
            response = requests.post(
                url,
                files=files,
                data=data,
                headers=headers,
                # proxies=proxies,
                # timeout=timeout
            )
            response.raise_for_status()
            # Validate response data
            result = response.json()
            if 'data' not in result or not isinstance(result['data'], list):
                raise InvalidResponseError("Invalid response structure")

            # Save detection results
            if save_detections:
                detect_save_path = os.path.join(
                    os.path.dirname(image_path),
                    'detect_AOD',
                    os.path.basename(image_path)
                )
                os.makedirs(os.path.dirname(detect_save_path), exist_ok=True)
                draw_boxes_and_save(image_path, result, detect_save_path)

            # Mark as completed
            cache.mark_task_complete(image_path, result)
            return (image_path, result)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        cache.mark_task_failed(image_path, error_msg)

        # Cool-down time: Increase wait time based on retry count
        current_retries = cache.get_task_status(image_path).get('retries', 0)
        print(current_retries)
        cool_down = min(2 ** current_retries, 300)  # Maximum cool-down of 5 minutes
        print(f"Task {image_path} failed, waiting {cool_down} seconds before retrying...")
        time.sleep(cool_down)

        raise

# ==================== Concurrency Control Enhancement ====================
def concurrent_detection(
    tasks: List[Tuple[str, List[str]]],
    api_key: str,
    cache_path: str = "detection_cache.json",
    max_workers: int = 3,  # Reduce default concurrency
    request_interval: float = 1.0,  # Increase default interval
    save_detections: bool = False
) -> Tuple[Dict[str, dict], list]:
    """
    Enhanced concurrent detection function
    """
    cache = DetectionCache(cache_path)
    responses = {}
    results = []

    # Filter completed tasks
    pending_tasks = []
    for img_path, prompts in tasks:
        status = cache.get_task_status(img_path)
        if not status or status['status'] != 'completed':
            pending_tasks.append((img_path, prompts))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit new tasks
        for img_path, prompts in pending_tasks:
            future = executor.submit(
                safe_single_detection,
                image_path=img_path,
                prompts=prompts,
                api_key=api_key,
                cache=cache,
                save_detections=save_detections
            )
            futures[future] = img_path
            time.sleep(request_interval)  # Request interval control

        # Process results
        for future in as_completed(futures):
            try:
                img_path, result = future.result()
                responses[img_path] = result
                # Validate data validity
                if result.get('data'):
                    results.append(len(result['data'][0]) > 0)
                else:
                    results.append(False)
            except Exception as e:
                print(f"Task ultimately failed: {str(e)}")

    return responses, results



def evaluate_cifar10_AOD(folder_path, prompts_path, eval_result_path, detection_result_path,
                     erased_concept=None, save_detections=False, max_works=10,
                     request_interval=0.5, cache_path="detection_cache.json", api_index=0):

    os.makedirs(os.path.dirname(eval_result_path), exist_ok=True)
    os.makedirs(os.path.dirname(detection_result_path), exist_ok=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # ==================== Data Preparation Section ====================
    # Collect image paths
    names = [
        name for name in os.listdir(folder_path)
        if name.endswith('.png') or name.endswith('.jpg')
    ]
    image_paths = [os.path.join(folder_path, name) for name in names]

    # Read prompt file to establish case mapping
    df = pd.read_csv(prompts_path)
    df['case_number'] = df['case_number'].astype('int')
    casenum2class = {row['case_number']: row['class'] for _, row in df.iterrows()}

    # Parse filenames to get case and index
    image_cases, image_indices = zip(*[
        map(int, name.split('.')[0].split('_')) for name in names
    ])
    class_labels = [casenum2class[case] for case in image_cases]

    # Build task list (maintaining original order)
    task_list = [
        (img_path, [cls])
        for img_path, cls in zip(image_paths, class_labels)
    ]

    # ==================== Core Execution Section ====================
    cache = DetectionCache(cache_path)

    # Modified evaluation result collection
    def get_results_from_cache(task_list: list, cache: DetectionCache) -> list:
        """Build an ordered list of results from the cache"""
        results = []
        for img_path, _ in task_list:
            cache_entry = cache.get_task_status(img_path)
            if cache_entry and cache_entry['status'] == 'completed':
                # Parse detection result validity
                result_data = cache_entry.get('result', {}).get('data', [])
                detected = len(result_data) > 0 and len(result_data[0]) > 0
                results.append(detected)
            else:
                # Handle unfinished/failed tasks
                results.append(None)  # Or mark as None based on requirements
        return results

    API_KEY = API_POOL[api_index]

    # Run detection (results from return are no longer used directly)
    _, _ = concurrent_detection(
        tasks=task_list,
        api_key=API_KEY,
        max_workers=max_works,
        request_interval=request_interval,
        save_detections=save_detections,
        cache_path=cache_path
    )

    # Build results from cache (ensure correct order)
    final_results = get_results_from_cache(task_list, cache)

    # Generate DataFrame (using ordered results)
    df_results = pd.DataFrame({
        'case_number': image_cases,
        'img_index': image_indices,
        'class_detected': final_results  # Using ordered results built from cache
    })

    merged_df = pd.merge(df, df_results)
    # move 'img_index' to the second column
    columns = merged_df.columns.tolist()
    columns.remove('img_index')
    columns.insert(1, 'img_index')
    merged_df = merged_df[columns]
    # save csv
    merged_df.to_csv(detection_result_path)

    # calculate erasing accuracy
    df_results = pd.DataFrame({
        'concept': class_labels,
        'class_detected': final_results
    })

    # Calculate accuracy for each concept
    accuracy_df = df_results.groupby('concept')['class_detected'].agg(['sum', 'count'])
    accuracy_df['accuracy'] = (accuracy_df['sum'] / accuracy_df['count'] * 100).round(3)
    accuracy_df = accuracy_df.reset_index()

    # Save per-concept accuracy
    accuracy_df[['concept', 'accuracy']].to_csv(eval_result_path, index=False)

    # Calculate in-class and other classes accuracy if erased_concept is provided
    if erased_concept is not None:
        in_class = accuracy_df[accuracy_df['concept'] == erased_concept]
        other_classes = accuracy_df[accuracy_df['concept'] != erased_concept]

        in_cls_acc = in_class['accuracy'].values[0] if not in_class.empty else float('inf')
        other_cls_acc = other_classes['accuracy'].mean() if not other_classes.empty else float('inf')

        with open(eval_result_path, 'a') as f:
            f.write(f'\nIn class accuracy ({erased_concept}): {in_cls_acc:.3f}\n')
            f.write(f'Other classes averaged accuracy: {other_cls_acc:.3f}\n')

    print("===============================")
    print(f"G.AOD's result saved in {detection_result_path}")
    print(f"Evaluation result saved in {eval_result_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 AOD.")

    parser.add_argument('--folder_path', required=True, type=str,
                        help='Path to the folder containing images.')
    parser.add_argument('--prompts_path', required=True, type=str,
                        help='Path to the CSV file containing prompts.')
    parser.add_argument('--eval_result_path', required=True, type=str,
                        help='Path to save evaluation results.')
    parser.add_argument('--detection_result_path', required=True, type=str,
                        help='Path to save detection results.')
    parser.add_argument('--erased_concept', default="automobile", type=str,
                        help='The concept that was erased during unlearning (default: automobile).')
    parser.add_argument('--save_detections', action='store_true',
                        help='Whether to save detection results (default: False).')
    parser.add_argument('--max_workers', default=1, type=int,
                        help='Maximum number of workers for parallel processing (default: 1).')
    parser.add_argument('--request_interval', default=10.0, type=float,
                        help='Interval between requests in seconds (default: 10.0).')
    parser.add_argument('--cache_path', default="./cache/default_cache.json", type=str,
                        help='Path to the cache file (default: ./cache/default_cache.json).')
    parser.add_argument('--api', default=0, type=int,
                        help='Api index to post requests (default: 0)..')

    args = parser.parse_args()

    evaluate_cifar10_AOD(
        folder_path=args.folder_path,
        prompts_path=args.prompts_path,
        eval_result_path=args.eval_result_path,
        detection_result_path=args.detection_result_path,
        erased_concept=args.erased_concept,
        save_detections=args.save_detections,
        max_works=args.max_workers,
        request_interval=args.request_interval,
        cache_path=args.cache_path,
        api_index=args.api
    )