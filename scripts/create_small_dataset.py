"""
Script to create a small subset of MSR-VTT dataset for testing
Creates annotation file with 100 videos split 70-15-15 (train-val-test)
"""

import json
import os
import random
from collections import defaultdict

# Paths
ANNOTATION_FILE = "data/annotations/train_val_videodatainfo.json"
VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/annotations"

def load_annotations(filepath):
    """Load MSR-VTT annotations"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def analyze_dataset(data):
    """Analyze the dataset structure"""
    sentences = data['sentences']
    
    # Group by video_id
    video_captions = defaultdict(list)
    for item in sentences:
        video_captions[item['video_id']].append(item)
    
    # Count splits
    split_count = defaultdict(int)
    for item in sentences:
        split_count[item['split']] += 1
    
    print("=" * 50)
    print("MSR-VTT Dataset Analysis")
    print("=" * 50)
    print(f"Total sentences/captions: {len(sentences)}")
    print(f"Total unique videos: {len(video_captions)}")
    print(f"\nSplit distribution (by caption):")
    for split, count in split_count.items():
        print(f"  {split}: {count}")
    
    # Check video files exist
    video_dir = VIDEO_DIR
    existing_videos = 0
    missing_videos = []
    for video_id in list(video_captions.keys())[:100]:  # Check first 100
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            existing_videos += 1
        else:
            missing_videos.append(video_id)
    
    print(f"\nVideo files check (first 100):")
    print(f"  Existing: {existing_videos}")
    print(f"  Missing: {len(missing_videos)}")
    if missing_videos[:5]:
        print(f"  Sample missing: {missing_videos[:5]}")
    
    return video_captions

def create_small_subset(video_captions, num_videos=100, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create a small subset with 100 videos split 70-15-15
    """
    # Get videos that exist
    available_videos = []
    for video_id in video_captions.keys():
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            available_videos.append(video_id)
    
    print(f"\nTotal available videos: {len(available_videos)}")
    
    # Random sample
    random.seed(42)
    selected_videos = random.sample(available_videos, min(num_videos, len(available_videos)))
    
    # Split
    n_train = int(num_videos * train_ratio)
    n_val = int(num_videos * val_ratio)
    n_test = num_videos - n_train - n_val
    
    train_videos = selected_videos[:n_train]
    val_videos = selected_videos[n_train:n_train + n_val]
    test_videos = selected_videos[n_train + n_val:]
    
    print(f"\nSubset distribution:")
    print(f"  Train: {len(train_videos)} videos")
    print(f"  Val: {len(val_videos)} videos")
    print(f"  Test: {len(test_videos)} videos")
    
    # Create new annotation data
    new_sentences = []
    
    for video_id in train_videos:
        for cap in video_captions[video_id]:
            new_cap = cap.copy()
            new_cap['split'] = 'train'
            new_sentences.append(new_cap)
    
    for video_id in val_videos:
        for cap in video_captions[video_id]:
            new_cap = cap.copy()
            new_cap['split'] = 'validate'  # MSR-VTT uses 'validate'
            new_sentences.append(new_cap)
    
    for video_id in test_videos:
        for cap in video_captions[video_id]:
            new_cap = cap.copy()
            new_cap['split'] = 'test'
            new_sentences.append(new_cap)
    
    new_data = {'sentences': new_sentences}
    
    print(f"  Total captions: {len(new_sentences)}")
    
    return new_data, train_videos, val_videos, test_videos

def save_subset(data, train_videos, val_videos, test_videos, output_dir):
    """Save the subset annotation and split files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main annotation
    output_file = os.path.join(output_dir, "train_val_videodatainfo_small.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\nSaved annotation to: {output_file}")
    
    # Save split files
    train_list = os.path.join(VIDEO_DIR, "train_list_small.txt")
    val_list = os.path.join(VIDEO_DIR, "val_list_small.txt")
    test_list = os.path.join(VIDEO_DIR, "test_list_small.txt")
    
    with open(train_list, 'w') as f:
        for video_id in train_videos:
            f.write(f"{video_id}.mp4\n")
    
    with open(val_list, 'w') as f:
        for video_id in val_videos:
            f.write(f"{video_id}.mp4\n")
    
    with open(test_list, 'w') as f:
        for video_id in test_videos:
            f.write(f"{video_id}.mp4\n")
    
    print(f"Saved train list to: {train_list}")
    print(f"Saved val list to: {val_list}")
    print(f"Saved test list to: {test_list}")

def main():
    print("Loading MSR-VTT annotations...")
    data = load_annotations(ANNOTATION_FILE)
    
    print("\nAnalyzing dataset...")
    video_captions = analyze_dataset(data)
    
    print("\nCreating small subset (100 videos, 70-15-15)...")
    new_data, train_videos, val_videos, test_videos = create_small_subset(
        video_captions, 
        num_videos=100,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("\nSaving subset...")
    save_subset(new_data, train_videos, val_videos, test_videos, OUTPUT_DIR)
    
    print("\n" + "=" * 50)
    print("DONE! Small dataset created successfully.")
    print("=" * 50)

if __name__ == "__main__":
    main()
