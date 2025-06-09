import os
import numpy as np
import torch
import networkx as nx
from PIL import Image
from tqdm import tqdm
import clip
from collections import defaultdict
from functools import lru_cache
import json
import shutil


class EnvGraph:
    def __init__(self, frames_dir, output_dir, sample_rate=1):
        """
        Initialize the environment graph for scene clustering

        Args:
            frames_dir: Directory containing video frames
            output_dir: Directory to save results
            sample_rate: Sample 1 frame every N frames
        """
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        os.makedirs(output_dir, exist_ok=True)

        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        # Get all frame paths
        print("Getting frame paths...")
        self.frame_paths = self._get_frame_paths()
        self.sampled_frame_paths = self.frame_paths[::sample_rate]
        print(f"Number of frames after sampling: {len(self.sampled_frame_paths)}")

        # Create frame index mapping for quick lookup
        self.frame_to_idx = {path: idx for idx, path in enumerate(self.sampled_frame_paths)}

        # Setup graph parameters
        self.thresh_upper = 0.85  # Threshold for localizing to existing node
        self.thresh_lower = 0.85  # Threshold for creating new node
        self.window_size = 14  # Window size for temporal consistency

        # Setup graph
        self.G = nx.Graph()
        self.G.pos = None
        self.state = {
            'frame': None,
            'node': None,
            'inactive': True,
            'viz_node': 0,
            'last_node': 0,
            'phase': 'localize'
        }

        # Initialize with a temporary node
        self.create_new_node(self.G, 0, [{'start': 0, 'stop': 0}])

        # Buffer for temporal consistency
        self.reset_buffer()
        self.history = []

        # Feature cache to avoid recomputing features
        self.feature_cache = {}

    def _get_frame_paths(self):
        """Get all frame paths and sort them in numerical order"""
        frame_paths = [os.path.join(self.frames_dir, f) for f in os.listdir(self.frames_dir)
                       if f.endswith(('.jpg', '.png', '.jpeg'))]
        # Sort based on frame numbers
        frame_paths.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        return frame_paths

    def reset_buffer(self):
        """Reset the buffer used for temporal consistency"""
        self.buffer = {'start': None, 'stop': None, 'node': None}

    @lru_cache(maxsize=100000)
    def get_frame_features(self, frame_path):
        """Extract CLIP features from an image with caching"""
        if frame_path in self.feature_cache:
            return self.feature_cache[frame_path]

        image = Image.open(frame_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features = image_features.cpu().numpy().flatten()

        # Cache the features
        self.feature_cache[frame_path] = features
        return features

    def compute_similarity(self, path_a, path_b):
        """Compute similarity between two frames using CLIP features"""
        feat_a = self.get_frame_features(path_a)
        feat_b = self.get_frame_features(path_b)

        # Compute cosine similarity
        similarity = np.dot(feat_a, feat_b)
        return similarity

    def score_pair_sets(self, set1, set2):
        """Score the similarity between two sets of frames"""
        if len(set1) == 0 or len(set2) == 0:
            return -1

        scores = []
        for i_path in [self.sampled_frame_paths[i] for i in set1]:
            set_scores = [self.compute_similarity(i_path, self.sampled_frame_paths[j]) for j in set2]
            scores.append(np.mean(set_scores))

        return np.mean(scores)

    def score_nodes(self, frame_idx):
        """
        Score the current frame against all existing nodes

        Args:
            frame_idx: Index of the current frame in sampled_frame_paths

        Returns:
            List of dictionaries with node scores
        """
        scores = []
        for node in self.G.nodes():
            visits = self.G.nodes[node]['members']

            # 20 uniformly sampled visits for efficient scoring
            if len(visits) > 20:
                visits = [visits[idx] for idx in np.round(np.linspace(0, len(visits) - 1, 20)).astype(int)]

            key_frames = []
            for visit in visits:
                frames = list(range(visit['start'], visit['stop'] + 1))
                if len(frames) < self.window_size:
                    key_frames += frames
                else:
                    mid = len(frames) // 2
                    key_frames += frames[mid - self.window_size // 2:mid + self.window_size // 2]

            # Get frames within window for current frame
            window = list(range(frame_idx - self.window_size // 2,
                                frame_idx + self.window_size // 2 + 1))
            window = [i for i in window if 0 <= i < len(self.sampled_frame_paths)]

            # Score current window against key frames of this node
            score = self.score_pair_sets(window, key_frames)
            scores.append({'node': node, 'score': score})

        return scores

    def create_new_node(self, G, frame_idx, members=[]):
        """Create a new node in the graph"""
        G.add_node(frame_idx, members=members)
        print(f'Created new NODE: {frame_idx}')

    def create_new_edge(self, G, frame_idx, src, dst):
        """Create a new edge between nodes"""
        if src is None or dst is None or src == dst:
            return

        # Calculate temporal distance for the edge attribute
        if self.state['last_node'] is not None:
            last_visit = G.nodes[self.state['last_node']]['members'][-1]
            last_frame = last_visit['stop']
        else:
            last_frame = 0

        dT = frame_idx - last_frame

        if G.has_edge(src, dst):
            G[src][dst]['dT'].append(dT)
        else:
            G.add_edge(src, dst, dT=[dT])

        print(f'Created EDGE: {src} --> {dst} | T: {dT}')

    def score_state(self, frame_idx):
        """
        Determine action to take for current frame

        Returns:
            top1_node: Highest scoring node
            top2_node: Second highest scoring node
            trigger: Action to take ('localize node', 'create node', or 'skip')
            node_scores: All node scores
        """
        node_scores = self.score_nodes(frame_idx)
        node_scores = sorted(node_scores, key=lambda node: -node['score'])

        top1_node = node_scores[0]
        top2_node = node_scores[1] if len(node_scores) > 1 else {'score': 0}

        trigger = 'skip'
        if top1_node['score'] > self.thresh_upper:
            trigger = 'localize node'
        elif top1_node['score'] < self.thresh_lower:
            trigger = 'create node'

        return top1_node, top2_node, trigger, node_scores

    def create_step(self, frame_idx):
        """Handle node creation"""
        best_node, _, trigger, _ = self.score_state(frame_idx)

        if 'create_count' not in self.buffer:
            self.reset_buffer()
            self.buffer['create_count'] = 0
            self.buffer['start'] = frame_idx
            self.buffer['stop'] = frame_idx

        # Verify the need to create a new node with temporal consistency
        if trigger == 'create node':
            if self.buffer['create_count'] < 5:  # Need consistent evidence to create node
                self.buffer['create_count'] += 1
                self.buffer['stop'] = frame_idx
                return None, None
            else:
                node_i = self.buffer['start']
                visit = {'start': self.buffer['start'], 'stop': self.buffer['stop'], 'node': node_i}
                self.create_new_node(self.G, node_i, members=[visit])
                self.create_new_edge(self.G, frame_idx, self.state['last_node'], node_i)

                self.state['last_node'] = node_i
                self.state['phase'] = 'localize'
                self.buffer = visit
                return node_i, 1.0

        elif trigger == 'localize node':
            self.reset_buffer()
            return self.localize_step(frame_idx)

        elif trigger == 'skip':
            self.reset_buffer()
            self.state['phase'] = 'localize'

        return None, None

    def localize_step(self, frame_idx):
        """Handle localizing to existing nodes"""
        best_node, _, trigger, _ = self.score_state(frame_idx)

        if trigger == 'localize node':
            node_i, score_i = best_node['node'], best_node['score']

            if self.buffer['node'] == node_i:  # Continuing visit to same node
                self.buffer['stop'] = frame_idx
            else:  # New visit to existing node
                visit = {'start': frame_idx, 'stop': frame_idx, 'node': node_i}
                self.G.nodes[node_i]['members'].append(visit)
                self.buffer = visit
                self.create_new_edge(self.G, frame_idx, self.state['last_node'], node_i)

            self.state['phase'] = 'localize'
            self.state['last_node'] = node_i
            return node_i, score_i

        elif trigger == 'create node':
            self.state['phase'] = 'create'
            return None, None

        elif trigger == 'skip':
            self.state['phase'] = 'localize'
            return None, None

    def log_history(self, frame_idx):
        """Log current state for visualization and analysis"""
        self.history.append({
            'frame': frame_idx,
            'G': self.G.copy(),
            'state': dict(self.state)
        })

    def build(self):
        """Build the scene graph by processing all frames"""
        # Process frames in window_size/2 to len-window_size/2 range
        start_idx = self.window_size // 2
        end_idx = len(self.sampled_frame_paths) - self.window_size // 2

        for frame_idx in tqdm(range(start_idx, end_idx)):
            # Process frame based on current state
            if self.state['phase'] == 'localize':
                node_i, score_i = self.localize_step(frame_idx)
            elif self.state['phase'] == 'create':
                node_i, score_i = self.create_step(frame_idx)

            # Update state
            self.state.update({
                'frame': frame_idx,
                'inactive': node_i is None,
                'node': node_i,
                'score': score_i,
                'viz_node': node_i if node_i is not None else self.state['viz_node']
            })

            # Log history for visualization
            self.log_history(frame_idx)

        # Save the final graph and analysis
        self.save_results()

    def save_results(self):
        """Save clustering results according to requested format"""
        print("Saving results...")

        # Create mapping from graph nodes to original frames
        node_to_frames = {}
        for node in self.G.nodes():
            node_frames = []
            for visit in self.G.nodes[node]['members']:
                for idx in range(visit['start'], visit['stop'] + 1):
                    if idx < len(self.sampled_frame_paths):
                        node_frames.append(self.sampled_frame_paths[idx])
            node_to_frames[node] = node_frames

        # Create directories for each node and save all frames
        for node in self.G.nodes():
            node_dir = os.path.join(self.output_dir, str(node))
            os.makedirs(node_dir, exist_ok=True)

            # Save all frames for this node
            frames = node_to_frames[node]
            for i, frame_path in enumerate(frames):
                # Extract original frame number from path
                frame_number = os.path.basename(frame_path).split('.')[0]
                dst = os.path.join(node_dir, f"{frame_number}.jpg")

                # Use hard links to save space
                if not os.path.exists(dst):
                    try:
                        os.link(frame_path, dst)
                    except OSError:
                        # If hard linking fails, copy the file
                        import shutil
                        shutil.copy2(frame_path, dst)

        # Extract timeline and analyze transitions
        timeline = self._extract_timeline()
        transitions = []
        scenes = []

        # Find transitions
        current_scene = {'start': 0, 'label': timeline[0] if timeline[0] != -1 else None}
        for i, label in enumerate(timeline):
            if i > 0 and label != timeline[i - 1] and label != -1 and timeline[i - 1] != -1:
                # End previous scene
                current_scene['end'] = i - 1
                current_scene['duration'] = current_scene['end'] - current_scene['start'] + 1
                scenes.append(current_scene)

                # Record transition
                transitions.append((i, timeline[i - 1], label))

                # Start new scene
                current_scene = {'start': i, 'label': label}

        # Add the last scene
        if timeline[-1] != -1:
            current_scene['end'] = len(timeline) - 1
            current_scene['duration'] = current_scene['end'] - current_scene['start'] + 1
            scenes.append(current_scene)

        # Scene statistics
        scene_stats = defaultdict(lambda: {'count': 0, 'total_frames': 0, 'durations': []})
        for scene in scenes:
            if scene['label'] is not None:
                scene_stats[scene['label']]['count'] += 1
                scene_stats[scene['label']]['total_frames'] += scene['duration']
                scene_stats[scene['label']]['durations'].append(scene['duration'])

        # Calculate average durations
        for node in scene_stats:
            durations = scene_stats[node]['durations']
            scene_stats[node]['avg_duration'] = np.mean(durations) if durations else 0

        # Save clustering metadata as JSON
        metadata = {
            'nodes': list(self.G.nodes()),
            'edges': list(self.G.edges()),
            'transitions': transitions,
            'scenes': scenes,
            'scene_stats': {str(k): v for k, v in scene_stats.items()},
        }

        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Print cluster statistics
        print("\nCluster Statistics:")
        print("-" * 60)
        print("Node ID | Scene Count | Total Frames | Avg Scene Duration")
        print("-" * 60)
        for node in sorted(scene_stats.keys()):
            stats = scene_stats[node]
            print(f"{node:7} | {stats['count']:11d} | {stats['total_frames']:12d} | {stats['avg_duration']:18.2f}")

        print("\nSignificant Scene Transitions:")
        print("-" * 60)
        significant_transitions = [
            (frame_idx, prev_label, new_label) for frame_idx, prev_label, new_label in transitions
            if self._get_scene_duration(scenes, prev_label) >= 5 and
               self._get_scene_duration(scenes, new_label) >= 5
        ]

        for i, (frame_idx, prev_label, new_label) in enumerate(significant_transitions[:10]):
            print(f"Transition {i + 1}: Frame {frame_idx * self.sample_rate} - "
                  f"Node {prev_label} to Node {new_label}")

        print(f"\nComplete! Results saved in {self.output_dir}")

    def _get_scene_duration(self, scenes, label):
        """Helper to get average duration of scenes with a given label"""
        durations = [s['duration'] for s in scenes if s['label'] == label]
        return np.mean(durations) if durations else 0

    def _extract_timeline(self):
        """Extract scene timeline from history without visualization"""
        timeline = []
        for t in range(len(self.history)):
            state = self.history[t]['state']
            if 'node' in state and state['node'] is not None:
                timeline.append(state['node'])
            elif t > 0 and len(timeline) > 0:
                timeline.append(timeline[-1])
            else:
                timeline.append(-1)  # No node
        return timeline




def main():
    # Get the base directory for frames and results
    base_frames_dir = './data/egoschema_frames_all'
    results_base_dir = './clustering_results'

    # Get video directories
    video_dirs = [d for d in os.listdir(base_frames_dir)
                  if os.path.isdir(os.path.join(base_frames_dir, d))]

    for video_name in video_dirs:
        frames_dir = os.path.join(base_frames_dir, video_name)
        if not os.path.exists(frames_dir):
            print(f"Skipping {video_name} - no rgb_frames directory")
            continue

        output_dir = os.path.join(results_base_dir, video_name)

        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            print(f"Skipping {video_name} - already processed")
            continue

        print(f"Processing video: {video_name}")
        print(f"Frames directory: {frames_dir}")
        print(f"Output directory: {output_dir}")

        # Initialize and build the environment graph
        env_graph = EnvGraph(frames_dir, output_dir, sample_rate=1)
        env_graph.build()


if __name__ == "__main__":
    main()