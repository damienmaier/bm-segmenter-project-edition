import pathlib

import numpy as np
import pydicom


def get_image_from_dicom(dicom_file_path: str) -> np.ndarray:
    dicom_file_data = pydicom.dcmread(dicom_file_path)
    image = np.array(dicom_file_data.pixel_array, dtype=np.int16)

    # this is the code used by the bm-segmenter software to get an image from a dicom
    image[image == -2000] = 0

    intercept = dicom_file_data.RescaleIntercept
    slope = dicom_file_data.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return image


def get_dicom_path_from_case_path(case_directory_path: pathlib.Path) -> pathlib.Path:
    first_directory_path = next(filter(pathlib.Path.is_dir, case_directory_path.iterdir()))
    second_directory_path = next(filter(pathlib.Path.is_dir, first_directory_path.iterdir()))
    dicom_file_path = next(element_path
                           for element_path in second_directory_path.iterdir() if element_path.name.startswith("I"))
    return dicom_file_path