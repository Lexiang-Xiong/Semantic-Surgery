import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import argparse

# ==================== OwlViT Detection Core Module ====================

# Global variables for storing model and processor to avoid repeated loading
OWL_MODEL = None
OWL_PROCESSOR = None

def load_owl_vit_model(cache_dir, device):
    """Load the OwlViT model and processor once into global variables."""
    global OWL_MODEL, OWL_PROCESSOR
    if OWL_MODEL is None or OWL_PROCESSOR is None:
        print("Loading OwlViT model and processor...")
        try:
            OWL_PROCESSOR = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32", cache_dir=cache_dir)
            OWL_MODEL = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32", cache_dir=cache_dir)
            OWL_MODEL.to(device)
            OWL_MODEL.eval()
            print(f"OwlViT model loaded to {device}.")
        except Exception as e:
            print(f"Failed to load OwlViT model. Error: {e}")
            raise

def get_concept_detected_with_owlvit(image_path, concept, device, score_threshold=0.1):
    """
    Use OwlViT to detect whether a specific concept exists in a single image.
    Returns a boolean value, consistent with the output logic of the AOD script.
    """
    if not os.path.exists(image_path):
        return False
    try:
        image = Image.open(image_path).convert("RGB")
        text_queries = [[f"a photo of a {concept}"]]
        inputs = OWL_PROCESSOR(text=text_queries, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = OWL_MODEL(**inputs)
            
        target_sizes = torch.tensor([image.size[::-1]])
        results = OWL_PROCESSOR.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=score_threshold)
        
        # If any bounding box is detected (non-empty score list), we consider the concept present
        return len(results[0]['scores']) > 0

    except Exception as e:
        print(f"Error processing {image_path} with OwlViT: {e}")
        return False

# ==================== AOD Evaluation Framework Main Function (adapted for OwlViT) ====================

def evaluate_cifar10_OwlViT(folder_path, prompts_path, eval_result_path, detection_result_path, erased_concept=None, score_threshold=0.1, cache_dir=None):
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(eval_result_path), exist_ok=True)
    os.makedirs(os.path.dirname(detection_result_path), exist_ok=True)
    
    # Load device and model
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_owl_vit_model(cache_dir, DEVICE)

    # ==================== Data Preparation (identical to the AOD script) ====================
    print("Preparing image paths and prompts...")
    names = [
        name for name in os.listdir(folder_path)
        if name.endswith('.png') or name.endswith('.jpg')
    ]
    image_paths = [os.path.join(folder_path, name) for name in names]

    df_prompts = pd.read_csv(prompts_path)
    df_prompts['case_number'] = df_prompts['case_number'].astype('int')
    casenum2class = {row['case_number']: row['class'] for _, row in df_prompts.iterrows()}

    image_cases, image_indices = zip(*[
        map(int, name.split('.')[0].split('_')) for name in names
    ])
    class_labels = [casenum2class[case] for case in image_cases]

    # ==================== Core Execution (replaced with OwlViT detection) ====================
    print("Starting detection using OwlViT...")
    detection_results = []
    
    # Iterate over all images and perform detection
    for img_path, concept_to_detect in tqdm(zip(image_paths, class_labels), total=len(image_paths), desc="OwlViT Detecting"):
        detected = get_concept_detected_with_owlvit(
            image_path=img_path,
            concept=concept_to_detect,
            device=DEVICE,
            score_threshold=score_threshold
        )
        detection_results.append(detected)

    # ==================== Result Processing & Saving (identical to the AOD script) ====================
    print("Processing and saving results...")
    # 1. Save detailed detection results (detection_result_path)
    df_detection = pd.DataFrame({
        'case_number': image_cases,
        'img_index': image_indices,
        'class_detected': detection_results
    })

    merged_df = pd.merge(df_prompts, df_detection, on='case_number')
    # Reorder columns
    columns = merged_df.columns.tolist()
    columns.remove('img_index')
    columns.insert(1, 'img_index')
    merged_df = merged_df[columns]
    merged_df.to_csv(detection_result_path, index=False)

    # 2. Compute and save accuracy (eval_result_path)
    df_eval = pd.DataFrame({
        'concept': class_labels,
        'class_detected': detection_results
    })

    # Compute accuracy by concept
    accuracy_df = df_eval.groupby('concept')['class_detected'].agg(['sum', 'count'])
    accuracy_df['accuracy'] = (accuracy_df['sum'] / accuracy_df['count'] * 100).round(3)
    accuracy_df = accuracy_df.reset_index()

    # Save per-concept accuracy
    accuracy_df[['concept', 'accuracy']].to_csv(eval_result_path, index=False)

    # If an erased_concept is specified, compute and append within-class and cross-class accuracy
    if erased_concept is not None:
        in_class_df = accuracy_df[accuracy_df['concept'] == erased_concept]
        other_classes_df = accuracy_df[accuracy_df['concept'] != erased_concept]

        # Compute within-class accuracy
        in_cls_acc = in_class_df['accuracy'].values[0] if not in_class_df.empty else 0.0
        # Compute mean cross-class accuracy
        other_cls_acc = other_classes_df['accuracy'].mean() if not other_classes_df.empty else 0.0

        with open(eval_result_path, 'a') as f:
            f.write(f'\nIn class accuracy ({erased_concept}): {in_cls_acc:.3f}\n')
            f.write(f'Other classes averaged accuracy: {other_cls_acc:.3f}\n')

    print("===============================")
    print(f"OwlViT detection details saved in: {detection_result_path}")
    print(f"Evaluation summary saved in: {eval_result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 generation using OwlViT detector.")
    
    # Keep command-line arguments identical to the original AOD script
    parser.add_argument('--folder_path', required=True, type=str,
                        help='Path to the folder containing images.')
    parser.add_argument('--prompts_path', required=True, type=str,
                        help='Path to the CSV file containing prompts.')
    parser.add_argument('--eval_result_path', required=True, type=str,
                        help='Path to save evaluation summary (accuracy).')
    parser.add_argument('--detection_result_path', required=True, type=str,
                        help='Path to save detailed detection results.')
    parser.add_argument('--erased_concept', type=str, default=None,
                        help='The concept that was erased in this experiment run.')
    parser.add_argument('--score_threshold', type=float, default=0.1,
                        help='Confidence threshold for OwlViT detection.')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache Hugging Face models.')

    args = parser.parse_args()

    # Map parameters from the AOD script to the new function
    evaluate_cifar10_OwlViT(
        folder_path=args.folder_path,
        prompts_path=args.prompts_path,
        eval_result_path=args.eval_result_path,
        detection_result_path=args.detection_result_path,
        erased_concept=args.erased_concept,
        score_threshold=args.score_threshold,
        cache_dir=args.cache_dir
    )