import json
import os
import argparse

def preprocess_msrvtt_annotations(json_path, output_dir, video_dir):
    """
    Chuyen doi MSR-VTT JSON annotation sang format train_list.txt va test_list.txt
    Format output: video_path\tcaption1;caption2;...
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_dict = {}
    
    for annotation in data.get('sentences', []):
        video_id = annotation['video_id']
        caption = annotation['caption'].strip()
        
        if video_id not in video_dict:
            video_dict[video_id] = []
        video_dict[video_id].append(caption)
    
    train_videos = set()
    val_videos = set()
    test_videos = set()
    
    for video_info in data.get('videos', []):
        video_id = video_info['video_id']
        split = video_info.get('split', 'train')
        
        if split == 'train':
            train_videos.add(video_id)
        elif split == 'validate':
            val_videos.add(video_id)
        elif split == 'test':
            test_videos.add(video_id)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train_list_new.txt'), 'w') as f:
        for video_id in sorted(train_videos):
            if video_id in video_dict:
                video_path = f"{video_id}.mp4"
                captions = ';'.join(video_dict[video_id])
                f.write(f"{video_path}\t{captions}\n")
    
    with open(os.path.join(output_dir, 'test_list_new.txt'), 'w') as f:
        for video_id in sorted(test_videos):
            if video_id in video_dict:
                video_path = f"{video_id}.mp4"
                captions = ';'.join(video_dict[video_id])
                f.write(f"{video_path}\t{captions}\n")
    
    with open(os.path.join(output_dir, 'val_list_new.txt'), 'w') as f:
        for video_id in sorted(val_videos):
            if video_id in video_dict:
                video_path = f"{video_id}.mp4"
                captions = ';'.join(video_dict[video_id])
                f.write(f"{video_path}\t{captions}\n")
    
    print(f"Train videos: {len(train_videos)}")
    print(f"Val videos: {len(val_videos)}")
    print(f"Test videos: {len(test_videos)}")
    print(f"Output saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, 
                        default='data/other/annotation/MSR_VTT.json',
                        help='Path to MSR-VTT JSON annotation')
    parser.add_argument('--output_dir', type=str,
                        default='data/videos',
                        help='Output directory for processed lists')
    parser.add_argument('--video_dir', type=str,
                        default='data/videos/all',
                        help='Directory containing video files')
    args = parser.parse_args()
    
    preprocess_msrvtt_annotations(args.json_path, args.output_dir, args.video_dir)
