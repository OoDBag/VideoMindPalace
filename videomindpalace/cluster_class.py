import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import glob

# Directory paths
input_dir = '/AMEGO/HOI_EGO_NEW'
frames_base_dir = '/AMEGO/processed_ego_videos_new'
output_dir = '/AMEGO/HOI_EGO_NEW_with_categories'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = clip.load("ViT-B/32", device=device)

# Common object categories for egocentric videos
object_categories = [
    "knife", "fork", "spoon", "plate", "bowl", "cup", "glass", "bottle", "pan", "pot",
    "spatula", "cutting board", "hand", "finger", "food", "vegetable", "fruit", "meat",
    "bread", "pasta", "rice", "utensil", "container", "appliance", "phone", "watch",
    "bag", "box", "tool", "device", "remote", "key", "wallet", "cloth", "towel"
]

# Category to ID mapping (assign an ID to each category)
category_to_id = {category: i + 1 for i, category in enumerate(object_categories)}
category_to_id["unknown"] = 0  # Unknown category gets ID 0


def get_frame_path(frame_dir, frame_num):
    """Get path to frame based on frame number"""
    frame_filename = f"frame_{frame_num:010d}.jpg"
    return os.path.join(frame_dir, frame_filename)


def classify_object(image, bbox):
    """Classify object in the bounding box using CLIP"""
    x, y, w, h = [int(coord) for coord in bbox]

    # Ensure bbox coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # If bbox is too small, return None
    if w < 10 or h < 10:
        return "unknown", 0.0

    # Crop the bbox
    crop = image[y:y + h, x:x + w]

    # Convert to PIL image and preprocess for CLIP
    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    # Prepare text prompts
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in object_categories]).to(device)

    # Calculate features and similarities
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculate similarity scores
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get the most likely category
        values, indices = similarity[0].topk(1)
        category = object_categories[indices[0].item()]
        confidence = values[0].item()

    return category, confidence


def find_keyframe(track, frames_dir):
    """
    Find the key frame in a track by extracting features and finding the one
    closest to the cluster centroid
    """
    bbox_list = track.get('obj_bbox', [])
    frame_nums = track.get('num_frame', [])

    # If there's only one frame, it's the key frame
    if len(frame_nums) == 1:
        return 0, frame_nums[0]

    # Extract features from all frames in the track
    features = []
    valid_indices = []
    valid_frames = []

    for i, (frame_num, bbox) in enumerate(zip(frame_nums, bbox_list)):
        frame_path = get_frame_path(frames_dir, frame_num)
        if not os.path.exists(frame_path):
            continue

        image = cv2.imread(frame_path)
        if image is None:
            continue

        # Extract features
        x, y, w, h = [int(coord) for coord in bbox]
        if w < 10 or h < 10:
            continue

        # Crop the bbox
        crop = image[y:y + h, x:x + w]

        # Convert to PIL image and preprocess for CLIP
        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        # Calculate and return image features
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            features.append(image_feature.cpu().numpy().flatten())
            valid_indices.append(i)
            valid_frames.append(frame_num)

    # If no valid features were extracted, return the first frame
    if not features:
        return 0, frame_nums[0]

    # Convert features to numpy array
    features = np.array(features)

    # Calculate mean feature (centroid)
    centroid = np.mean(features, axis=0)

    # Find the closest feature to the centroid
    distances = np.linalg.norm(features - centroid, axis=1)
    closest_idx = np.argmin(distances)

    # Return the original index and frame number
    return valid_indices[closest_idx], valid_frames[closest_idx]


def add_categories_to_tracks(data, frames_dir):
    """Add categories to tracks based on CLIP classification of key frames"""
    # Make a deep copy of the data to avoid modifying the original
    enhanced_data = data.copy()

    print("Finding key frames and classifying objects...")
    for track in tqdm(enhanced_data):
        track_id = track.get('track_id')
        bbox_list = track.get('obj_bbox', [])
        frame_nums = track.get('num_frame', [])

        if not bbox_list or not frame_nums:
            track["category"] = ["unknown"]
            track["category_id"] = 0
            continue

        # Find the key frame for this track
        key_idx, key_frame = find_keyframe(track, frames_dir)

        # Get the bbox for the key frame
        key_bbox = bbox_list[key_idx]

        # Read the key frame
        frame_path = get_frame_path(frames_dir, key_frame)
        if not os.path.exists(frame_path):
            track["category"] = ["unknown"]
            track["category_id"] = 0
            continue

        image = cv2.imread(frame_path)
        if image is None:
            track["category"] = ["unknown"]
            track["category_id"] = 0
            continue

        # Classify the object in the key frame
        category, confidence = classify_object(image, key_bbox)

        # Add category and category_id to the track
        track["category"] = [category]
        track["category_id"] = category_to_id.get(category, 0)
        track["key_frame"] = key_frame

    return enhanced_data


def process_json_file(json_path):
    """Process a single JSON file"""
    # Extract video ID from filename
    video_id = os.path.basename(json_path).split('.')[0]

    # Define frames directory based on video ID
    frames_dir = os.path.join(frames_base_dir, video_id, 'rgb_frames')

    # Define output JSON path
    output_json_path = os.path.join(output_dir, os.path.basename(json_path))

    print(f"\nProcessing {video_id}...")
    print(f"JSON: {json_path}")
    print(f"Frames: {frames_dir}")
    print(f"Output: {output_json_path}")

    # Check if frames directory exists
    if not os.path.exists(frames_dir):
        print(f"WARNING: Frames directory not found: {frames_dir}")
        return False

    try:
        # Load JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Add categories to tracks
        enhanced_data = add_categories_to_tracks(data, frames_dir)

        # Save the enhanced data
        with open(output_json_path, 'w') as f:
            json.dump(enhanced_data, f, indent=2)

        # Print summary
        category_counts = {}
        for track in enhanced_data:
            category = track.get("category", ["unknown"])[0]
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1

        print(f"Classification summary for {video_id}:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")

        return True

    except Exception as e:
        print(f"ERROR processing {json_path}: {str(e)}")
        return False


def main():
    # Get list of all JSON files in input directory
    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    print(f"Found {len(json_files)} JSON files in {input_dir}")

    # Process each JSON file
    successful = 0
    failed = 0

    for json_file in json_files:
        result = process_json_file(json_file)
        if result:
            successful += 1
        else:
            failed += 1

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed to process: {failed} files")


if __name__ == "__main__":
    main()