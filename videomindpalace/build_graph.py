import os
import json
from collections import defaultdict, Counter


def convert_framerate(frame_number, source_fps, target_fps):
    """
    Convert a frame number from source fps to target fps.

    Args:
        frame_number: The original frame number
        source_fps: Source frames per second
        target_fps: Target frames per second

    Returns:
        Converted frame number
    """
    return frame_number * target_fps // source_fps


def get_video_captions(captions_data, video_id):
    """
    Extract captions for a specific video from the captions data.

    Args:
        captions_data: The complete captions data dictionary
        video_id: The ID of the video to extract captions for

    Returns:
        Dictionary mapping frame numbers to captions
    """
    # Check if the video_id exists in the captions data
    if video_id not in captions_data:
        print(f"Warning: No captions found for video {video_id}")
        return {}

    # The captions in the provided structure are a list of captions per frame
    # The index in the array is the frame number (at 1 fps)
    captions_list = captions_data[video_id]
    return {i: caption for i, caption in enumerate(captions_list)}


def get_uncovered_ranges(covered_frames, total_frames=180):
    """
    Find ranges of frames that are not covered by any interaction.

    Args:
        covered_frames: Dictionary with frame numbers as keys
        total_frames: Total number of frames to consider

    Returns:
        List of [start_frame, end_frame] ranges that are not covered
    """
    uncovered_ranges = []
    start = None

    # Check each frame from 1 to total_frames
    for i in range(1, total_frames + 1):
        if i not in covered_frames:
            if start is None:
                start = i
        else:
            if start is not None:
                uncovered_ranges.append([start, i - 1])
                start = None

    # Handle the case where the last frames are uncovered
    if start is not None:
        uncovered_ranges.append([start, total_frames])

    return uncovered_ranges


def get_caption_for_frame(video_captions, frame, nearby_frames=5):
    """
    Get a caption for a frame, trying nearby frames if the exact frame doesn't have a caption.

    Args:
        video_captions: Dictionary of all captions indexed by frame number
        frame: The frame to get caption for
        nearby_frames: How many frames to check in each direction

    Returns:
        A caption string
    """
    # First try the exact frame
    if frame in video_captions and video_captions[frame]:
        return video_captions[frame]

    # Try nearby frames
    for offset in range(1, nearby_frames + 1):
        # Try frame + offset
        if frame + offset in video_captions and video_captions[frame + offset]:
            return video_captions[frame + offset]

        # Try frame - offset
        if frame - offset in video_captions and video_captions[frame - offset]:
            return video_captions[frame - offset]


    return "#C C working with dough"


def construct_activity_graph(tracking_json_path, clustering_dir_path, captions_json_path, captions_json_path2,
                             video_id):
    """
    Construct a graph representing activity zones with interactions between hands and objects.

    Args:
        tracking_json_path: Path to the JSON file containing tracking information (30 fps)
        clustering_dir_path: Path to the directory containing cluster folders (15 fps)
        captions_json_path: Path to the JSON file containing frame captions (1 fps)
        captions_json_path2: Path to the alternative JSON file containing frame captions (1 fps)
        video_id: ID of the video being processed

    Returns:
        A dictionary representing the graph structure
    """
    try:
        # Load tracking data (30 fps)
        with open(tracking_json_path, 'r') as f:
            tracking_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Tracking data file not found at {tracking_json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in tracking data file at {tracking_json_path}")
        return None

    # Load primary captions data
    try:
        with open(captions_json_path, 'r') as f:
            all_captions_data = json.load(f)

        # Extract captions for this specific video
        video_captions = get_video_captions(all_captions_data, video_id)
        length = len(video_captions)
        print(f"Caption length from primary source: {length}")

        # Check if length is not 180, use alternative captions source
        if length != 180:
            print(f"Caption length is not 180, trying alternative captions source...")
            try:
                with open(captions_json_path2, 'r') as f:
                    alt_captions_data = json.load(f)

                video_captions = get_video_captions(alt_captions_data, video_id)
                length = len(video_captions)
                print(f"Caption length from alternative source: {length}")
            except FileNotFoundError:
                print(f"Error: Alternative captions data file not found at {captions_json_path2}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in alternative captions data file at {captions_json_path2}")

    except FileNotFoundError:
        print(f"Error: Primary captions data file not found at {captions_json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in primary captions data file at {captions_json_path}")
        return None

    # Get list of cluster directories (15 fps)
    try:
        cluster_dirs = [d for d in os.listdir(clustering_dir_path)
                        if os.path.isdir(os.path.join(clustering_dir_path, d))]

        # We can use cluster_dirs to verify clusters or process additional data if needed
    except FileNotFoundError:
        print(f"Warning: Clustering directory not found at {clustering_dir_path}")
        cluster_dirs = []

    # Convert tracking data frame numbers from 30 fps to 1 fps
    for track in tracking_data:
        track["num_frame_original"] = track["num_frame"].copy()  # Keep original frame numbers
        track["num_frame"] = [convert_framerate(frame, 30, 1) for frame in track["num_frame"]]

    # Initialize graph structure with video_id as the key
    graph = {
        "Layer2: activity_zones": []
    }

    # Keep track of covered frames
    covered_frames = {}

    # Get all cluster IDs
    cluster_ids = set(track.get("cluster") for track in tracking_data if "cluster" in track)

    # Process each activity zone (cluster)
    for zone_index, cluster_id in enumerate(cluster_ids):
        # Filter tracks for this cluster
        cluster_tracks = [track for track in tracking_data if track.get("cluster") == cluster_id]

        # Initialize activity zone with index-based naming
        activity_zone = {
            "zone_id": f"activity_zones {zone_index}",
            "original_cluster_id": cluster_id,
            "Layer1: Human and object": {}
        }

        # Process all objects in the tracking data
        # Note: User mentioned all objects are already known to interact with hands
        for obj_track in cluster_tracks:
            obj_frames = obj_track["num_frame"]
            obj_id = obj_track["track_id"]

            # Find overlapping frames with other objects in the same cluster
            # This assumes hands are implicitly involved in these interactions
            for start_frame, end_frame in [(obj_frames[0], obj_frames[-1])]:  # Simplified to use full range
                # Get key frame for the interaction
                obj_key_frame = obj_track.get("key_frame", -1)

                if obj_key_frame != -1:
                    key_frame = convert_framerate(obj_key_frame, 30, 1)
                else:
                    key_frame = (start_frame + end_frame) // 2

                # Get caption for key frame
                # Access dictionary with integer key, not string key
                caption = video_captions.get(key_frame, "No caption available")

                # Get object identifier string
                obj_category = obj_track.get("category", ["unknown"])
                obj_category_str = obj_category[0] if obj_category else "unknown"

                # Use category_id for object identification if available, otherwise fallback to track_id
                obj_id = obj_track.get("category_id", obj_track["track_id"])

                # Determine spatial relation (side) based on available data
                most_common_side = "unknown"
                if "side" in obj_track:
                    sides = obj_track["side"]
                    if sides:
                        # Flatten list if needed and get most common side
                        flat_sides = []
                        for side in sides:
                            if isinstance(side, list):
                                flat_sides.extend(side)
                            else:
                                flat_sides.append(side)
                        most_common_side = Counter(flat_sides).most_common(1)[0][0] if flat_sides else "unknown"

                # Format the key as requested
                hand_to_obj_key = f"hand->({obj_category_str}, ID: {obj_id})"

                # Create interaction with the requested structure
                interaction = {
                    hand_to_obj_key: {
                        "temporal_window": [start_frame, end_frame],
                        "spatial_relation": most_common_side,
                        "interaction": caption
                    }
                }

                # Mark these frames as covered
                for frame in range(start_frame, end_frame + 1):
                    covered_frames[frame] = True

                # Add interaction to activity zone
                activity_zone["Layer1: Human and object"].update(interaction)

        # Add activity zone to graph if it has interactions
        if activity_zone["Layer1: Human and object"]:
            graph["Layer2: activity_zones"].append(activity_zone)

    # Find uncovered frame ranges
    uncovered_ranges = get_uncovered_ranges(covered_frames)

    if uncovered_ranges:
        print(f"Found {len(uncovered_ranges)} uncovered frame ranges")


        uncovered_frames_data = []


        for start_frame, end_frame in uncovered_ranges:

            frame_captions = {}
            for frame in range(start_frame, end_frame + 1):

                frame_caption = get_caption_for_frame(video_captions, frame)
                frame_captions[str(frame)] = frame_caption


            uncovered_item = {
                "temporal_window": [start_frame, end_frame],
                "interaction": frame_captions
            }


            uncovered_frames_data.append(uncovered_item)


        if uncovered_frames_data:
            graph["uncovered_frames"] = uncovered_frames_data

    return graph


def process_all_videos(tracking_dir, clustering_base_dir, captions_json_path, captions_json_path2, output_path):
    """
    Process all videos and save the combined results to a single JSON file.

    Args:
        tracking_dir: Directory containing all tracking JSON files (30 fps)
        clustering_base_dir: Base directory containing clustering subfolders (15 fps)
        captions_json_path: Path to the JSON file containing frame captions (1 fps)
        captions_json_path2: Path to the alternative JSON file containing frame captions (1 fps)
        output_path: Path where the combined JSON file should be saved
    """

    all_videos_results = {}


    try:
        tracking_files = [f for f in os.listdir(tracking_dir) if f.endswith('.json')]
    except FileNotFoundError:
        print(f"Error: Tracking directory not found at {tracking_dir}")
        return

    print(f"Found {len(tracking_files)} tracking files to process")


    for i, tracking_file in enumerate(tracking_files):

        video_id = tracking_file.replace('.json', '')

        print(f"\nProcessing video {i + 1}/{len(tracking_files)}: {video_id}")


        tracking_json_path = os.path.join(tracking_dir, tracking_file)
        clustering_dir_path = os.path.join(clustering_base_dir, video_id)

        video_graph = construct_activity_graph(
            tracking_json_path,
            clustering_dir_path,
            captions_json_path,
            captions_json_path2,
            video_id
        )


        if video_graph:
            all_videos_results[video_id] = video_graph
            print(f"Successfully processed video {video_id}")
        else:
            print(f"Failed to process video {video_id}")


    if all_videos_results:
        try:
            with open(output_path, 'w') as f:
                json.dump(all_videos_results, f, indent=2)
            print(f"\nAll video results successfully saved to {output_path}")
            print(f"Processed {len(all_videos_results)} videos in total")
        except Exception as e:
            print(f"Error saving combined results to {output_path}: {e}")
    else:
        print("No video results to save")


if __name__ == "__main__":

    tracking_dir = "./HOI_EGO_NEW_with_categories/"  # 30 fps
    clustering_base_dir = "./clustering_results/"  # 15 fps
    captions_json_path = "./data/egoschema/final_processed_videos.json"  # 1 fps
    captions_json_path2 = "./data/egoschema/uu_lavila_subset.json"  # 1 fps

    output_path = "./data/egoschema/all_activity_graphs.json"

    print("Starting to process all videos...")
    print(f"- Tracking data directory: {tracking_dir}")
    print(f"- Clustering base directory: {clustering_base_dir}")
    print(f"- Primary captions file: {captions_json_path}")
    print(f"- Alternative captions file: {captions_json_path2}")

    process_all_videos(
        tracking_dir,
        clustering_base_dir,
        captions_json_path,
        captions_json_path2,
        output_path
    )