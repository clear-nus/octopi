import os
import numpy as np
import cv2
import random
from constants import TRAIN_OBJECTS, VAL_OBJECTS, TEST_OBJECTS
import json
import argparse


def get_frames(dataset_path, frames_output_path):
    def extract_span(dataset_file_path, frames_output_path, obj_sample_count, threshold):
        object_name = "_".join(dataset_file_path.split("/")[-1].split("_")[:-1])
        if object_name not in obj_sample_count.keys():
            obj_sample_count[object_name] = 0
        else:
            obj_sample_count[object_name] += 1

        cap = cv2.VideoCapture(dataset_file_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, prev_frame = cap.read()
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        movement_start = -1
        movement_end = -1

        for i in range(1, frame_number):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            total_diff = np.sum(thresh)

            if total_diff > threshold and i > 7:
                if movement_start == -1:
                    movement_start = i
                movement_end = i
            prev_frame = gray

        if movement_start != -1 and movement_end != -1 and movement_end - movement_start>=10:
            os.makedirs(os.path.join(frames_output_path, f'physiclear_{object_name}_{obj_sample_count[object_name]}'), exist_ok=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, movement_start-1)
            for i in range(movement_start-1, movement_end + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frames_output_path, f'physiclear_{object_name}_{obj_sample_count[object_name]}', str(i).rjust(10, '0') + '.jpg')
                cv2.imwrite(frame_path, frame)
        cap.release()

    
    def extract_diff(dataset_file_path, frames_output_path, obj_sample_count, threshold, min_len, num_frame_percent):
        object_name = "_".join(dataset_file_path.split("/")[-1].split("_")[:-1])
        if object_name not in obj_sample_count.keys():
            obj_sample_count[object_name] = 0
        else:
            obj_sample_count[object_name] += 1

        cap = cv2.VideoCapture(dataset_file_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, prev_frame = cap.read()
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        all_diffs = []

        for i in range(1, frame_number):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            total_diff = np.sum(thresh)
            all_diffs.append((i, total_diff))
            prev_frame = gray

        cap = cv2.VideoCapture(dataset_file_path)
        frame_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = int(num_frame_percent * frame_number)
        all_diffs = sorted(all_diffs, key=lambda t: t[1], reverse=True)[:num_frames]
        all_diffs = sorted(all_diffs, key=lambda t: t[0])
        all_diffs = [i[0] for i in all_diffs]
        if len(all_diffs) >= min_len:
            os.makedirs(os.path.join(frames_output_path, f'physiclear_{object_name}_{obj_sample_count[object_name]}'), exist_ok=True)
            for i in range(1, frame_number):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in all_diffs:
                    frame_path = os.path.join(frames_output_path, f'physiclear_{object_name}_{obj_sample_count[object_name]}', str(i).rjust(10, '0') + '.jpg')
                    cv2.imwrite(frame_path, frame)
        cap.release()

    
    def extract_all(dataset_file_path, frames_output_path, obj_sample_count):
        object_name = "_".join(dataset_file_path.split("/")[-1].split("_")[:-1])
        if object_name not in obj_sample_count.keys():
            obj_sample_count[object_name] = 0
        else:
            obj_sample_count[object_name] += 1

        cap = cv2.VideoCapture(dataset_file_path)

        os.makedirs(os.path.join(frames_output_path, f'physiclear_{object_name}_{obj_sample_count[object_name]}'), exist_ok=True)
        # for i in range(0, frame_number):
        i = 0
        while True:
            ret, frame = cap.read()
            frame_path = os.path.join(frames_output_path, f'physiclear_{object_name}_{obj_sample_count[object_name]}', str(i).rjust(10, '0') + '.jpg')
            try:
                cv2.imwrite(frame_path, frame)
            except:
                break
            i += 1
            cv2.waitKey(1)
        cap.release()


    # extract videos
    dataset_files = os.listdir(dataset_path)
    if '.DS_Store' in dataset_files:
        dataset_files.remove('.DS_Store')

    obj_sample_count = {}
    for i in range(len(dataset_files)):
        if "csv" not in dataset_files[i]:
            dataset_file_path = os.path.join(dataset_path, dataset_files[i])
            # extract_span(dataset_file_path, frames_output_path, obj_sample_count, threshold=12)
            # extract_all(dataset_file_path, frames_output_path, obj_sample_count)
            extract_diff(dataset_file_path, frames_output_path, obj_sample_count, threshold=0, min_len=5, num_frame_percent=0.3)
            if i % 10 == 0:
                print(f"{i} / {len(dataset_files)} done.")


def get_samples(data_output_path, train_json_path, val_json_path, test_json_path):
    # shuffle seen objects before train/val split
    random.shuffle(TRAIN_OBJECTS)
    samples = [i for i in os.listdir(data_output_path) if os.path.isdir(os.path.join(data_output_path, i))]
    train_sample_paths = {}
    val_sample_paths = {}
    test_sample_paths = {}
    for sample in samples:
        sample_obj = "_".join(sample.split("_")[:-1])
        if len(VAL_OBJECTS) == 0:
            if sample_obj in TRAIN_OBJECTS:
                rand = random.random()
                if rand < 0.8:
                    if sample_obj not in train_sample_paths.keys():
                        train_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                    else:
                        train_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
                elif rand >= 0.8:
                    if sample_obj not in val_sample_paths.keys():
                        val_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                    else:
                        val_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
        else:
            if sample_obj in TRAIN_OBJECTS:
                if sample_obj not in train_sample_paths.keys():
                    train_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                else:
                    train_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
            if sample_obj in VAL_OBJECTS:
                if sample_obj not in val_sample_paths.keys():
                    val_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
                else:
                    val_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
        if sample_obj in TEST_OBJECTS:
            if sample_obj not in test_sample_paths.keys():
                test_sample_paths[sample_obj] = [os.path.join(data_output_path, sample)]
            else:
                test_sample_paths[sample_obj].append(os.path.join(data_output_path, sample))
    with open(train_json_path, 'w') as f:
        json.dump(train_sample_paths, f)
        f.close()
    with open(val_json_path, 'w') as f:
        json.dump(val_sample_paths, f)
        f.close()
    with open(test_json_path, 'w') as f:
        json.dump(test_sample_paths, f)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='directory with tactile videos')
    parser.add_argument('--output_path', help='directory to save processed frames and sample files')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # 1) get frames
    print(f"Getting frames...")
    get_frames(args.dataset_path, args.output_path)
    print("Done!")

    # 2) create samples for each set
    print(f"\nGetting sample files...")
    train_json_path = os.path.join(args.output_path, "train_samples.json")
    val_json_path = os.path.join(args.output_path, "val_samples.json")
    test_json_path = os.path.join(args.output_path, "test_samples.json")
    get_samples(args.output_path, train_json_path, val_json_path, test_json_path)
    print("Done!")