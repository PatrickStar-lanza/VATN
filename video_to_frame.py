import cv2
import os
import tqdm
import pandas as pd

"""
For MPII-Cooking dataset
This file segments the whole video sequence representing a high-level activity into a set of lower level activity(video-to-clip).
Then split the video clips into video frames.
"""


def video_segment(dataset, imshow, annotations):
    """
    This function segment the videos into video clips according to the annotations provided, each video will be saved
    according to the example path "{dataset}/pre_processed/video_clip/{video(file)_name}/{activity}.avi"
    :param dataset: the name of the dataset
    :param imshow: whether the images are presented during the process
    :param annotations: a data frame includes at least four columns, which are ['fileName','startFrame','endFrame','activity']
    """
    # The folder for the videos to be processed
    input_path = "/home/zheng/VATN/videos/"
    videos = os.listdir(input_path)

    # Iterate all the videos in the folder
    for video in tqdm.tqdm(videos, desc='Clipping videos'):
        clip_dir = os.path.join(input_path, video)
        video_name = os.path.splitext(video)[0]

        output_path = os.path.join(os.path.dirname(__file__), '..', f'data/{dataset}/pre_processed/video_clip', video_name)
        os.makedirs(output_path, exist_ok=True)

        # Create a list for the start frame, end frame and the corresponding activity labels for one video
        time_and_label = []
        start_index = []
        for index, row in annotations.iterrows():
            if row['fileName'] == video_name:
                time_and_label.append([row['startFrame'], row['endFrame'], row['activity']])
                # index list is used to anchor the start frame of a clip. the index of index list is the same to the time and label list.
                start_index.append(row['startFrame'])

        # capture img according to the start and end frame in a video
        cap = cv2.VideoCapture(clip_dir)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        down_sampling = 1
        count = 0
        start_point = 0
        end_point = 0
        with tqdm.tqdm(desc=f'Processing {video_name}', leave=False) as inner_progress:
            while True:
                success, image = cap.read()
                inner_progress.update()
                if not success:
                    tqdm.tqdm.write(f'All frames ({count}) have been presented for {video_name}')
                    break
                count = count + 1
                if count in start_index:
                    item_index = start_index.index(count)
                    start_point = count
                    end_point = time_and_label[item_index][1]
                    activity = time_and_label[item_index][2]
                    # create the path for a video clip, namely, a lower level activity
                    output_video_path = f"{output_path}/{activity}.avi"
                    # if the same activity already exist, create a new file name with the index
                    if os.path.exists(output_video_path):
                        output_video_path = f"{output_path}/{activity}_{item_index}.avi"
                    # create the clip video
                    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
                    tqdm.tqdm.write(f'Generated {output_video_path}')
                if count % down_sampling == 0:
                    if start_point <= count <= end_point:
                        if imshow:
                            cv2.imshow('img', image)
                        out_video.write(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        if imshow:
            cv2.destroyAllWindows()


def video_clip_to_frame(dataset, imshow, img_size, sample_step):
    """
    This function is used to segment all the video clips for all the higher level activities into a set of frames based on the sample step
    :param dataset: the name of the dataset
    :param imshow: whether the images are presented during the process
    :param img_size: the size of the output images
    :param sample_step: the sample steps, which defines the fps, for example, in mpii-cooking
    dataset, the sample step is 6, and the original fps is 30, after this downsampling,
    the fps become 5.
    :return: None
    """
    if dataset == 'mpii_cooking':
        input_path = "/home/zheng/VATN/cookings/video_clip/"
        output_path = "/home/zheng/VATN/cookings/clip_frame/"
        video_folders = os.listdir(input_path)

    for video_folder in tqdm.tqdm(video_folders, desc='Clip to frames'):
        video_folder_path = os.path.join(input_path, video_folder)
        video_clips = os.listdir(video_folder_path)
        output_folder_path = os.path.join(output_path, video_folder)
        for video_clip in tqdm.tqdm(video_clips, desc=f'Processing {video_folder}', leave=False):
            video_name = os.path.splitext(video_clip)[0]
            output_frame_folder = os.path.join(output_folder_path, video_name)
            os.makedirs(output_frame_folder, exist_ok=True)
            video_path = os.path.join(video_folder_path, video_clip)
            cap = cv2.VideoCapture(video_path)
            count = 0
            with tqdm.tqdm(desc=f'Processing {video_name}', leave=False) as inner_progress:
                while True:
                    success, image = cap.read()
                    inner_progress.update()
                    if not success:
                        tqdm.tqdm.write(f'All frames ({count}) have been presented for {video_name}')
                        break
                    if count % sample_step == 0:
                        image = cv2.resize(image, img_size)
                        if imshow:
                            cv2.imshow('frame', image)
                        cv2.imwrite(f"{output_frame_folder}/{int(count/sample_step)}.jpg", image)
                    count += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            if imshow:
                cv2.destroyAllWindows()



# Read the annotation file


def main():
    # Define paths
    dataset_path = "/home/zheng/VATN/videos/"
    output_path = "/home/zheng/VATN/cookings/"
    annotation_file = "/home/zheng/VATN/annotation.csv"

    # Load annotations
    annotations = pd.read_csv(annotation_file)

    # Define the dataset name, here we simply use 'cooking' as its name
    dataset_name = "cooking"

    # Update paths in video_segment and video_clip_to_frame functions
    video_segment(dataset_name, False, annotations)  # Set imshow to False for no display
    video_clip_to_frame(dataset_name, False, (960, 720), 6)  # Image size set to 960x720 and sample_step to 6

    # Moving the processed files to the desired location
    # os.rename() might not work across different filesystems or partitions
    # So, I'm using a safer way to move the directories
    os.system(f"mv {os.path.dirname(__file__)}/../data/{dataset_name}/pre_processed/video_clip {output_path}")
    os.system(f"mv {os.path.dirname(__file__)}/../data/{dataset_name}/pre_processed/clip_frame {output_path}")

if __name__ == "__main__":
    main()


