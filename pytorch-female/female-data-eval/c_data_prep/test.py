from itertools import groupby
from operator import itemgetter
from pydantic import BaseModel
from typing import List, Optional


class ImageAnnotation(BaseModel):
    category_id: int
    bbox: List[int]
    segmentation: List[int]

class ImageData(BaseModel):
    new_image_id: int
    source_video_name: str
    image_frame_no: int
    annotations: Optional[List[ImageAnnotation]]

def group_subsequent_frames(records: List[ImageData]) -> List[List[ImageData]]:
    # Group by video file name
    grouped_by_video = groupby(records, key=lambda x: x.source_video_name)

    result = []

    # For each group (each video file)
    for video_file, group in grouped_by_video:
        group = list(group)
        
        # Sort the group by frame number
        group.sort(key=lambda x: x.image_frame_no)
        
        current_group = [group[0]]  # Start a new group with the first frame

        # Create groups of subsequent frames
        for i in range(1, len(group)):
            current_frame = group[i - 1].image_frame_no
            next_frame = group[i].image_frame_no
            if next_frame == current_frame + 1:
                current_group.append(group[i])
            else:
                result.append(current_group)
                current_group = [group[i]]  # Start a new group
        
        # Append the last group
        result.append(current_group)

    return result

# Example usage
new_image_ids_videos_frames = [
    ImageData(new_image_id=0, source_video_name='2022-01-27_034100_VID001_Trim_2', image_frame_no=6, annotations=None),
    ImageData(new_image_id=1, source_video_name='2022-01-27_034100_VID001_Trim_2', image_frame_no=7, annotations=None),
    ImageData(new_image_id=2, source_video_name='2022-01-27_034100_VID001_Trim_2', image_frame_no=8, annotations=None),
    ImageData(new_image_id=3, source_video_name='2022-01-27_034100_VID001_Trim_2', image_frame_no=12, annotations=None),
    ImageData(new_image_id=4, source_video_name='2022-01-27_034100_VID001_Trim_2', image_frame_no=13, annotations=None),
    ImageData(new_image_id=5, source_video_name='2022-01-27_034100_VID001_Trim_2', image_frame_no=14, annotations=None),
    ImageData(new_image_id=6, source_video_name='2022-01-27_034100_VID001_Trim_3', image_frame_no=1, annotations=None),
    ImageData(new_image_id=7, source_video_name='2022-01-27_034100_VID001_Trim_3', image_frame_no=2, annotations=None),
    ImageData(new_image_id=8, source_video_name='2022-01-27_034100_VID001_Trim_3', image_frame_no=3, annotations=None),
    ImageData(new_image_id=9, source_video_name='2022-01-27_034100_VID001_Trim_3', image_frame_no=4, annotations=None)
]

grouped_frames = group_subsequent_frames(new_image_ids_videos_frames)

for group in grouped_frames:
    for image_data in group:
        print(image_data)
    print("--- Group End ---")
