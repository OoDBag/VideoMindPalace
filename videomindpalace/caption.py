import csv
import json
import requests
import random
import os
import base64
import glob
from tqdm import tqdm

# Define your OpenAI deployments with endpoint and corresponding API keys
OPENAI_DEPLOYMENTS = [
    (
        'endpoints',
        'key')
]


def local_image_to_data_url(image_path):

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def process_frame_group(frame_paths, questions):
    endpoints = random.choice(OPENAI_DEPLOYMENTS)
    url = f'https://{endpoints[0]}.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview'
    api_key = endpoints[1]

    # Encode all images in the group
    image_urls = []
    for frame_path in frame_paths:
        try:
            image_urls.append(local_image_to_data_url(frame_path))
        except Exception as e:
            print(f"Error encoding image {frame_path}: {e}")
            return None

    # Prepare the messages for the API call
    response = requests.post(
        url,
        headers={'Content-Type': 'application/json', 'api-key': api_key},
        json={
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a highly capable assistant tasked with analyzing video content. Your job is to generate detailed and contextually accurate captions for five consecutive frames, "
                        "each representing a one-second interval from a five-second segment of a video. Ensure the captions capture the key visual elements, actions, and context depicted in each frame. "
                        "The captions should provide a clear understanding of the scene, highlighting any transitions or interactions occurring over time."
                        "The captions should be provided in a format without frame numbering or asterisks. Each caption should start directly with the description."
                    )
                },
                {"role": "user", "content":
                    [
                        {"type": "image_url", "image_url": {"url": image_urls[0]}},
                        {"type": "image_url", "image_url": {"url": image_urls[1]}},
                        {"type": "image_url", "image_url": {"url": image_urls[2]}},
                        {"type": "image_url", "image_url": {"url": image_urls[3]}},
                        {"type": "image_url", "image_url": {"url": image_urls[4]}}
                    ]
                 }
            ],
            "max_tokens": 800,
            "temperature": 0,
            "seed": 1024
        }
    )

    if 'error' in response.json():
        print("Error:", response.json())
        return None
    else:
        # Process the response to remove numbering and clean up format
        content = response.json()['choices'][0]['message']['content']
        # Split by newline, filter out empty lines and clean up formatting
        descriptions = [line.strip() for line in content.split('\n') if line.strip()]
        # Remove numbering and formatting
        cleaned_descriptions = []
        for desc in descriptions:
            # Remove various forms of numbering and formatting
            desc = desc.replace('**', '')  # Remove bold formatting
            desc = ' '.join(desc.split(':')[1:]).strip() if ':' in desc else desc  # Remove "Frame X:" format
            # Remove numbered list format (e.g., "1. ", "2. ")
            if desc[0].isdigit() and '. ' in desc[:4]:
                desc = desc[desc.find('. ') + 2:]
            cleaned_descriptions.append(desc)

        return cleaned_descriptions


def load_existing_results():
    processed_files = glob.glob('./data/egoschema/camera_ready_processed_videos_*.json')
    if not processed_files:
        return {}, 0

    latest_file = max(processed_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Found existing results: {latest_file}")

    with open(latest_file, 'r') as f:
        results = json.load(f)

    return results, len(results)


def main():
    # Base path for frames
    frames_base_path = './data/egoschema_frames'

    # Load questions data
    with open('./data/egoschema/subset_anno.json', 'r') as f:
        question_data = json.load(f)

    # Load alternative captions for fallback
    with open('./data/egoschema/uu_lavila_subset.json', 'r') as f:
        fallback_captions = json.load(f)
    print(f"Loaded fallback captions for {len(fallback_captions)} videos")

    # Load existing results if any
    results, processed_count = load_existing_results()
    print(f"Loaded {len(results)} processed videos")

    # Get the list of all video directories
    all_video_dirs = [d for d in os.listdir(frames_base_path)
                      if os.path.isdir(os.path.join(frames_base_path, d))]

    # Filter out already processed videos
    videos_to_process = [vid for vid in all_video_dirs if vid not in results]
    print(f"Found {len(videos_to_process)} videos to process")

    # Process remaining videos
    for video_id in tqdm(videos_to_process):
        video_path = os.path.join(frames_base_path, video_id)
        frame_files = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))]
        frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

        # Get questions for this video
        if video_id not in question_data:
            continue

        content = question_data[video_id]
        question = f"Q: {content['question']} | Option 0: {content['option 0']} Option 1: {content['option 1']} Option 2: {content['option 2']} Option 3: {content['option 3']} Option 4: {content['option 4']}"
        video_questions = [question]

        # Process frames in groups of 5
        descriptions = []
        consecutive_errors = 0
        i = 0

        while i < len(frame_files):
            end_idx = min(i + 5, len(frame_files))
            frame_group = frame_files[i:end_idx]
            frame_paths = [os.path.join(video_path, f) for f in frame_group]
            print(f"Processing frames {i} to {end_idx - 1} of video {video_id}")

            # Generate descriptions for this group of frames
            result = process_frame_group(frame_paths, video_questions)

            if result:
                descriptions.extend(result)
                consecutive_errors = 0
                i += 5
            else:
                consecutive_errors += 1
                print(f"Error processing group {i // 5} of video {video_id}. Consecutive errors: {consecutive_errors}")


                if video_id in fallback_captions:
                    start_idx = i
                    end_idx = min(i + 5, len(frame_files))


                    if start_idx < len(fallback_captions[video_id]):
                        fallback_group = fallback_captions[video_id][start_idx:end_idx]
                        print(f"Using fallback captions for frames {start_idx} to {end_idx - 1}")
                        descriptions.extend(fallback_group)
                    else:
                        print(
                            f"Warning: Fallback captions index out of range for video {video_id} at index {start_idx}")

                        for _ in range(end_idx - start_idx):
                            descriptions.append("No caption available for this frame")

                    i += 5
                    consecutive_errors = 0
                else:
                    print(f"No fallback captions available for video {video_id}")


                    if consecutive_errors >= 5:
                        print(f"Skipping remaining frames for video {video_id} due to 5 consecutive errors")


                        remaining_frames = len(frame_files) - i
                        if remaining_frames > 0 and video_id in fallback_captions:
                            frames_to_process = remaining_frames
                            print(f"Filling remaining {frames_to_process} frames with fallback captions")

                            while i < len(frame_files):
                                end_idx = min(i + 5, len(frame_files))
                                if i < len(fallback_captions[video_id]):
                                    fallback_group = fallback_captions[video_id][
                                                     i:min(end_idx, len(fallback_captions[video_id]))]
                                    descriptions.extend(fallback_group)


                                    for _ in range((end_idx - i) - len(fallback_group)):
                                        descriptions.append("No caption available for this frame")
                                else:

                                    for _ in range(end_idx - i):
                                        descriptions.append("No caption available for this frame")

                                i += 5
                        break

                    i += 5

        # Store results for this video
        results[video_id] = descriptions
        processed_count += 1

        # Save intermediate results every 5 videos
        if processed_count % 5 == 0:
            with open(f'./data/egoschema/processed_videos_{processed_count}.json', 'w') as f:
                json.dump(results, f, indent=4)

    # Save final results
    with open('./data/egoschema/final_processed_videos.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()