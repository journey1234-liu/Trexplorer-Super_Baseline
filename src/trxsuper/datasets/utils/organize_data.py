import os
import re
import json
import shutil
from tqdm import tqdm

if __name__ == '__main__':
    dataset = "Aorta24"  # "atm22", "parse2022", "syntrx"
    data_dir = "/home/lyyu/data/Aorta24"
    cl_dir = "data/Aorta24"  # Path to the dataset
    dst_dir = "data/Aorta24"  # Destination directory for the organized dataset
    splits = "/home/lyyu/data/Aorta24/splits_final.json"
    paths_dict = {
        'annots_extra': os.path.join(dst_dir, "annots_extra"),
        'annotst_sep_test': os.path.join(dst_dir, "annots_sep_test"),
        'annotst_test': os.path.join(dst_dir, "annots_test"),
        'annotst_train': os.path.join(dst_dir, "annots_train"),
        'annotst_val': os.path.join(dst_dir, "annots_val"),
        'annotst_val_sub_vol': os.path.join(dst_dir, "annots_val_sub_vol"),
        'images_extra': os.path.join(dst_dir, "images_extra"),
        'images_sep_test': os.path.join(dst_dir, "images_sep_test"),
        'images_test': os.path.join(dst_dir, "images_test"),
        'images_train': os.path.join(dst_dir, "images_train"),
        'images_val': os.path.join(dst_dir, "images_val"),
        'images_val_sub_vol': os.path.join(dst_dir, "images_val_sub_vol"),
        'masks_extra': os.path.join(dst_dir, "masks_extra"),
        'masks_sep_test': os.path.join(dst_dir, "masks_sep_test"),
        'masks_test': os.path.join(dst_dir, "masks_test"),
        'masks_train': os.path.join(dst_dir, "masks_train"),
        'masks_val': os.path.join(dst_dir, "masks_val"),
        'masks_val_sub_vol': os.path.join(dst_dir, "masks_val_sub_vol")}

    os.makedirs(dst_dir, exist_ok=True)
    for key in paths_dict:
        os.makedirs(paths_dict[key], exist_ok=True)

    annot_dir = os.path.join(cl_dir, "centerlines")

    with open(splits, "r") as f:
        splits = json.load(f)

    if dataset == "atm22":
        num_train_samples = 220
        num_val_samples = 16
        num_test_samples = 60
    elif dataset == "parse2022":
        num_train_samples = 72
        num_val_samples = 8
        num_test_samples = 20
    elif dataset == "syntrx":
        num_train_samples = 368
        num_val_samples = 32
        num_test_samples = 100
    elif dataset == "ASOCA":
        train_cases = splits["train"]
        train_cases.sort()
        train_img_files = [os.path.join(
            data_dir, f"{case_id.split('_')[0]}", "CTCA", f"{case_id}.nrrd") for case_id in train_cases]
        train_mask_files = [os.path.join(
            data_dir, f"{case_id.split('_')[0]}", "Annotations", f"{case_id}.nrrd") for case_id in train_cases]
        train_annot_files = [os.path.join(
            annot_dir, f"{case_id}.pickle") for case_id in train_cases]

        val_cases = splits["val"]
        val_cases.sort()
        val_img_files = [os.path.join(
            data_dir, f"{case_id.split('_')[0]}", "CTCA", f"{case_id}.nrrd") for case_id in val_cases]
        val_mask_files = [os.path.join(
            data_dir, f"{case_id.split('_')[0]}", "Annotations", f"{case_id}.nrrd") for case_id in val_cases]
        val_annot_files = [os.path.join(
            annot_dir, f"{case_id}.pickle") for case_id in val_cases]

        test_cases = splits["test"]
        test_cases.sort()
        test_img_files = [os.path.join(
            data_dir, f"{case_id.split('_')[0]}", "CTCA", f"{case_id}.nrrd") for case_id in test_cases]
        test_mask_files = [os.path.join(
            data_dir, f"{case_id.split('_')[0]}", "Annotations", f"{case_id}.nrrd") for case_id in test_cases]
        test_annot_files = [os.path.join(
            annot_dir, f"{case_id}.pickle") for case_id in test_cases]

    elif dataset == "Aorta24":
        train_cases = splits["train"]
        train_cases.sort()
        train_img_files = [os.path.join(
            data_dir, "imagesTr", f"{case_id}_0000.nii.gz") for case_id in train_cases]
        train_mask_files = [os.path.join(
            data_dir, "labelsTr_bin", f"{case_id}.nii.gz") for case_id in train_cases]
        train_annot_files = [os.path.join(
            annot_dir, f"{case_id}.pickle") for case_id in train_cases]

        val_cases = splits["val"]
        val_cases.sort()
        val_img_files = [os.path.join(
            data_dir, "imagesTr", f"{case_id}_0000.nii.gz") for case_id in val_cases]
        val_mask_files = [os.path.join(
            data_dir, "labelsTr_bin", f"{case_id}.nii.gz") for case_id in val_cases]
        val_annot_files = [os.path.join(
            annot_dir, f"{case_id}.pickle") for case_id in val_cases]

        test_cases = splits["test"]
        test_cases.sort()
        test_img_files = [os.path.join(
            data_dir, "imagesTs", f"{case_id}_0000.nii.gz") for case_id in test_cases]
        test_mask_files = [os.path.join(
            data_dir, "labelsTs_bin", f"{case_id}.nii.gz") for case_id in test_cases]
        test_annot_files = [os.path.join(
            annot_dir, f"{case_id}.pickle") for case_id in test_cases]

    elif dataset == "ATM22":
        train_cases = splits["train"]
        train_cases.sort()
        train_img_files = [os.path.join(
            data_dir, "imagesTr", case_id) for case_id in train_cases]
        train_mask_files = [os.path.join(
            data_dir, "labelsTr", case_id) for case_id in train_cases]
        train_annot_files = [os.path.join(
            annot_dir, f"{case_id.split('.')[0].removesuffix('_0000')}.pickle") for case_id in train_cases]

        val_cases = splits["val"]
        val_cases.sort()
        val_img_files = [os.path.join(
            data_dir, "imagesTr", case_id) for case_id in val_cases]
        val_mask_files = [os.path.join(
            data_dir, "labelsTr", case_id) for case_id in val_cases]
        val_annot_files = [os.path.join(
            annot_dir, f"{case_id.split('.')[0].removesuffix('_0000')}.pickle") for case_id in val_cases]

        test_cases = splits["test"]
        test_cases.sort()
        test_img_files = [os.path.join(
            data_dir, "imagesTr", case_id) for case_id in test_cases]
        test_mask_files = [os.path.join(
            data_dir, "labelsTr", case_id) for case_id in test_cases]
        test_annot_files = [os.path.join(
            annot_dir, f"{case_id.split('.')[0].removesuffix('_0000')}.pickle") for case_id in test_cases]

    # move the first num_train_samples samples to the training set
    for img_file, mask_file, annot_file in tqdm(zip(train_img_files, train_mask_files, train_annot_files)):
        idx = img_file.split("/")[-1].removesuffix("_0000.nii.gz")
        shutil.copy(img_file, os.path.join(
            paths_dict['images_train'], idx + '.nii.gz'))
        shutil.copy(mask_file, os.path.join(
            paths_dict['masks_train'], idx + '.nii.gz'))
        shutil.copy(annot_file, os.path.join(
            paths_dict['annotst_train'], idx + '.pickle'))

    # move the next num_val_samples samples to the validation set
    for img_file, mask_file, annot_file in tqdm(zip(val_img_files, val_mask_files, val_annot_files)):
        idx = img_file.split("/")[-1].removesuffix("_0000.nii.gz")
        shutil.copy(img_file, os.path.join(
            paths_dict['images_val'], idx + '.nii.gz'))
        shutil.copy(mask_file, os.path.join(
            paths_dict['masks_val'], idx + '.nii.gz'))
        shutil.copy(annot_file, os.path.join(
            paths_dict['annotst_val'], idx + '.pickle'))
        shutil.copy(img_file, os.path.join(
            paths_dict['images_val_sub_vol'], idx + '.nii.gz'))
        shutil.copy(mask_file, os.path.join(
            paths_dict['masks_val_sub_vol'], idx + '.nii.gz'))
        shutil.copy(annot_file, os.path.join(
            paths_dict['annotst_val_sub_vol'], idx + '.pickle'))

    # move the next num_test_samples samples to the test set
    for img_file, mask_file, annot_file in tqdm(zip(test_img_files, test_mask_files, test_annot_files)):
        idx = img_file.split("/")[-1].removesuffix(".nii.gz")
        shutil.copy(img_file, os.path.join(
            paths_dict['images_test'], idx + '_0000.nii.gz'))
        shutil.copy(mask_file, os.path.join(
            paths_dict['masks_test'], idx + '.nii.gz'))
        shutil.copy(annot_file, os.path.join(
            paths_dict['annotst_test'], idx + '.pickle'))
        shutil.copy(img_file, os.path.join(
            paths_dict['images_sep_test'], idx + '-0.nii.gz'))
        shutil.copy(mask_file, os.path.join(
            paths_dict['masks_sep_test'], idx + '-0.nii.gz'))
        shutil.copy(annot_file, os.path.join(
            paths_dict['annotst_sep_test'], idx + '-0.pickle'))
