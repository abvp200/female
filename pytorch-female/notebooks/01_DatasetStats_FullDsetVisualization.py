import json
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# Function to read annotations from JSON file
def read_json(json_file_path: str):
    with open(json_file_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict

def process_annotations(coco_json):
    video_frames = defaultdict(list)
    frame_annotations = defaultdict(lambda: defaultdict(int))
    category_colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 255, 0),
        4: (128, 0, 128),
        5: (255, 165, 0),
        6: (255, 192, 203),
        7: (165, 42, 42),
        8: (128, 128, 128),
        9: (0, 255, 255)
    }

    for image_info in coco_json['images']:
        for image in image_info:
            image_id = image['id']
            video_id = image_id.split('__')[0]
            video_frames[video_id].append(image_id)

    for annotation in coco_json['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        frame_annotations[image_id][category_id] += 1

    return video_frames, frame_annotations, category_colors

def create_visualization(video_frames, frame_annotations, category_colors, max_frames_per_line=800):
    frame_height = 10  # Height of each frame in pixels

    for video_id, frames in tqdm(video_frames.items()):
        num_frames = int(frames[-1].split('__')[1])
        num_lines = (num_frames // max_frames_per_line) + 1
        img_width = min(max_frames_per_line, num_frames)
        img_height = num_lines * frame_height
        image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        x_pos = 0
        y_pos = 0
        for frame_num in range(num_frames):
        # for frame_id in frames:
            frame_id = f'{video_id}__{frame_num}'
            if frame_id in frames:
                if x_pos >= max_frames_per_line:
                    x_pos = 0
                    y_pos += 1
                y_offset = y_pos * frame_height
                frame_annotation = frame_annotations[frame_id]
                for category_id, count in frame_annotation.items():
                    color = category_colors[category_id]
                    for i in range(count):
                        draw.point((x_pos, y_offset + i), fill=color)
            x_pos += 1

        image.show()  # Display the image
        image.save(f'{video_id}.png')  # Save the image with the video_id as the filename

# Specify paths to dataset directories
coco_json_path = '/mnt/data_vilen/02_intermediate/glenda_full_v5_bbox_and_segm.json'

# Load the data
coco_json = read_json(coco_json_path)

# Process the annotations
video_frames, frame_annotations, category_colors = process_annotations(coco_json)

# Create the visualization
create_visualization(video_frames, frame_annotations, category_colors)
