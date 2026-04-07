# -*- coding: utf-8 -*-
import cv2
import os
import json
import glob
import numpy as np
import shutil
from sklearn.model_selection import train_test_split


def json_to_yolo(json_path, output_folder_img, output_folder_label):
    """
    Convert JSON annotation files to YOLO format label files
    :param json_path: Path to folder containing JSON files
    :param output_folder_img: Output folder for images
    :param output_folder_label: Output folder for label files
    """
    json_files = glob.glob(json_path + "/*.json")
    print("Found JSON files:", json_files)

    if not os.path.exists(output_folder_img):
        os.makedirs(output_folder_img)
    if not os.path.exists(output_folder_label):
        os.makedirs(output_folder_label)

    # Define lithology class mapping
    rock_class_mapping = {
        'Granite': 0,
        'Sandstone': 1,
        'Gneiss': 2,
        'Tuff': 3,
        'Rhyolite': 4,
        'Dolomite': 5
    }

    print("Using class mapping:", rock_class_mapping)

    for json_file in json_files:
        print("Processing file:", json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            json_info = json.load(f)

        # Fix image path reading issue
        image_path = os.path.join(json_path, json_info["imagePath"])
        if not os.path.exists(image_path):
            # Try looking in parent directory
            image_path = os.path.join(os.path.dirname(json_path), json_info["imagePath"])
            if not os.path.exists(image_path):
                print(f"Warning: Cannot find image file: {json_info['imagePath']}")
                continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read image: {image_path}")
            continue

        height, width, _ = img.shape
        np_w_h = np.array([[width, height]], np.int32)

        # Save image (using original filename)
        img_filename = os.path.basename(json_info["imagePath"])
        img_output_path = os.path.join(output_folder_img, img_filename)
        cv2.imwrite(img_output_path, img)

        # Generate label file
        txt_file = os.path.join(output_folder_label, os.path.basename(json_file).replace(".json", ".txt"))
        with open(txt_file, "w") as f:
            for point_json in json_info["shapes"]:
                # Get class label
                label_name = point_json.get("label", "")
                if label_name not in rock_class_mapping:
                    print(f"Warning: Unknown label '{label_name}', skipping this annotation")
                    continue

                class_id = rock_class_mapping[label_name]

                # Generate YOLO format annotation
                np_points = np.array(point_json["points"], np.int32)
                norm_points = np_points / np_w_h
                norm_points_list = norm_points.tolist()

                # Convert polygon points to YOLO format (first is class id, then normalized coordinates)
                points_str = " ".join([f"{point[0]:.6f} {point[1]:.6f}" for point in norm_points_list])
                txt_content = f"{class_id} {points_str}\n"
                f.write(txt_content)

        print(f"Successfully processed: {json_file} -> {txt_file}")


def split_dataset(val_size, test_size, imgpath, txtpath,
                  output_train_img_folder, output_val_img_folder, output_test_img_folder,
                  output_train_txt_folder, output_val_txt_folder, output_test_txt_folder,
                  postfix='jpg'):
    """
    Split dataset into training, validation and test sets, then copy corresponding files
    :param val_size: Validation set ratio (e.g., 0.2)
    :param test_size: Test set ratio (e.g., 0.2)
    :param imgpath: Original image folder path
    :param txtpath: Original label folder path (YOLO format .txt files)
    :param output_train_img_folder: Output folder for training images
    :param output_val_img_folder: Output folder for validation images
    :param output_test_img_folder: Output folder for test images
    :param output_train_txt_folder: Output folder for training labels
    :param output_val_txt_folder: Output folder for validation labels
    :param output_test_txt_folder: Output folder for test labels
    :param postfix: Image file extension (default 'jpg')
    """
    # Ensure all output directories exist
    os.makedirs(output_train_img_folder, exist_ok=True)
    os.makedirs(output_val_img_folder, exist_ok=True)
    os.makedirs(output_test_img_folder, exist_ok=True)
    os.makedirs(output_train_txt_folder, exist_ok=True)
    os.makedirs(output_val_txt_folder, exist_ok=True)
    os.makedirs(output_test_txt_folder, exist_ok=True)

    # Get list of all txt files
    listdir = [i for i in os.listdir(txtpath) if i.endswith('.txt')]
    print(f"Found {len(listdir)} label files")

    if len(listdir) == 0:
        print("Error: No label files found")
        return

    # First split: separate training set and temporary set (validation + test)
    train, temp = train_test_split(listdir, test_size=(val_size + test_size),
                                   shuffle=True, random_state=0)

    # Second split: split temporary set into validation and test sets
    # Calculate the proportion of test set within the temporary set
    test_ratio_in_temp = test_size / (val_size + test_size)
    val, test = train_test_split(temp, test_size=test_ratio_in_temp,
                                 shuffle=True, random_state=0)

    print(f"Dataset split: training {len(train)}, validation {len(val)}, test {len(test)}")

    # Copy training set files
    for i in train:
        img_source_path = os.path.join(imgpath, f'{i[:-4]}.{postfix}')
        txt_source_path = os.path.join(txtpath, i)

        # Check if file exists
        if not os.path.exists(img_source_path):
            # Try different image formats
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                alt_path = os.path.join(imgpath, f'{i[:-4]}.{ext}')
                if os.path.exists(alt_path):
                    img_source_path = alt_path
                    break
            else:
                print(f"Warning: Cannot find image file {i[:-4]}.{postfix}, skipping")
                continue

        img_dest_path = os.path.join(output_train_img_folder, f'{i[:-4]}.{postfix}')
        txt_dest_path = os.path.join(output_train_txt_folder, i)

        shutil.copy(img_source_path, img_dest_path)
        shutil.copy(txt_source_path, txt_dest_path)

    # Copy validation set files
    for i in val:
        img_source_path = os.path.join(imgpath, f'{i[:-4]}.{postfix}')
        txt_source_path = os.path.join(txtpath, i)

        # Check if file exists
        if not os.path.exists(img_source_path):
            # Try different image formats
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                alt_path = os.path.join(imgpath, f'{i[:-4]}.{ext}')
                if os.path.exists(alt_path):
                    img_source_path = alt_path
                    break
            else:
                print(f"Warning: Cannot find image file {i[:-4]}.{postfix}, skipping")
                continue

        img_dest_path = os.path.join(output_val_img_folder, f'{i[:-4]}.{postfix}')
        txt_dest_path = os.path.join(output_val_txt_folder, i)

        shutil.copy(img_source_path, img_dest_path)
        shutil.copy(txt_source_path, txt_dest_path)

    # Copy test set files
    for i in test:
        img_source_path = os.path.join(imgpath, f'{i[:-4]}.{postfix}')
        txt_source_path = os.path.join(txtpath, i)

        # Check if file exists
        if not os.path.exists(img_source_path):
            # Try different image formats
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                alt_path = os.path.join(imgpath, f'{i[:-4]}.{ext}')
                if os.path.exists(alt_path):
                    img_source_path = alt_path
                    break
            else:
                print(f"Warning: Cannot find image file {i[:-4]}.{postfix}, skipping")
                continue

        img_dest_path = os.path.join(output_test_img_folder, f'{i[:-4]}.{postfix}')
        txt_dest_path = os.path.join(output_test_txt_folder, i)

        shutil.copy(img_source_path, img_dest_path)
        shutil.copy(txt_source_path, txt_dest_path)

    print("Dataset splitting completed")


def main():
    json_path = r"E:\******\Dataset_Split\******\jsons"  # Path to JSON annotation files
    output_folder_img = r'E:\******\Dataset_Split\******\images'  # Output folder for images
    output_folder_label = r'E:\******\Dataset_Split\******\txt'  # Output folder for labels

    # Convert JSON to YOLO format
    json_to_yolo(json_path, output_folder_img, output_folder_label)

    val_size = 0.2  # Validation set ratio
    test_size = 0.2  # Test set ratio
    imgpath = output_folder_img  # Use converted image path
    txtpath = output_folder_label  # Use converted label path

    # Output split folder paths
    output_train_img_folder = r'E:\******\Dataset_Split\******\dataset\images\train'
    output_val_img_folder = r'E:\******\Dataset_Split\******\dataset\images\val'
    output_test_img_folder = r'E:\******\Dataset_Split\******\dataset\images\test'
    output_train_txt_folder = r'E:\******\Dataset_Split\******\dataset\labels\train'
    output_val_txt_folder = r'E:\******\Dataset_Split\******\dataset\labels\val'
    output_test_txt_folder = r'E:\******\Dataset_Split\******\dataset\labels\test'

    split_dataset(
        val_size=val_size,
        test_size=test_size,
        imgpath=imgpath,
        txtpath=txtpath,
        output_train_img_folder=output_train_img_folder,
        output_val_img_folder=output_val_img_folder,
        output_test_img_folder=output_test_img_folder,
        output_train_txt_folder=output_train_txt_folder,
        output_val_txt_folder=output_val_txt_folder,
        output_test_txt_folder=output_test_txt_folder
    )

    # Create data.yaml file
    create_data_yaml(
        output_folder=r'E:\******\Dataset_Split\******\dataset',
        class_names=['Granite', 'Sandstone', 'Gneiss', 'Tuff', 'Rhyolite', 'Dolomite']
    )


def create_data_yaml(output_folder, class_names):
    """
    Create data.yaml file required for YOLO training
    :param output_folder: Dataset root directory (contains images and labels folders)
    :param class_names: List of class names
    """
    yaml_content = f"""# Rock core detection dataset
path: {output_folder}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}
"""

    yaml_path = os.path.join(output_folder, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created data.yaml file: {yaml_path}")


if __name__ == "__main__":
    main()
