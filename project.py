"""
Provides code for reading and editing a BM-segmenter project.

A BM-segmenter project has the following structure :
<project root>
    <project name>.ml_prj
    dataset.toml
    models
        <segmentation1>.seg
        <segmentation2>.seg
        ...
    data
        dicoms
            <case1>
                0.npz
            <case2>
                0.npz
            <case3>
                0.npz
            ...
    masks
        <segmentation1>
            <case1>.npz
            <case2>.npz
            <case3>.npz
            ...
        <segmentation2>
            <case1>.npz
            <case2>.npz
            <case3>.npz
            ...
        <segmentation3>
            <case1>.npz
            <case2>.npz
            <case3>.npz
            ...

<project name>.ml_prj contains the name and description of the project and the user names.

`dataset.toml` contains a list of all the case names. It also contains a mapping between case groups and the cases
they contain. The group feature does not seem to be often used by the actual users.

<segmentation>.seg stores the name and color of the segmentation

0.npz contains key-value pairs :
    - matrix : a 2D numpy array containing the HU values
    - spacing : a pair of floats indicating the physical size corresponding to a pixel
    - windowing : the min and max HU values set by the user for the grayscale conversion
    - crop_x : a pair of value, each between 0 and 100 that indicates the start and end horizontal percentage of the image that must be displayed
    - crop_y : same with vertical percentages
    - slice_info : unknown


<segmentation>/<case>.npz contains key-value pairs :
    - predicted : a 2D array for the predicted matrix
    - current : a 2D array for the current segmentation matrix
    - validated : a 2D array for the validated. The image is validated if and only if this key has a value.
    - users : a list of the users that have validated the segmentation

"""

import pathlib
import sys

import numpy as np
import toml

# This is probably not the best way to do things, but I did not find any other simple way to reuse the code of
# mlsegmentation project
sys.path.append(str(pathlib.Path(__file__) / r".."))
sys.path.append(str(pathlib.Path(__file__) / r"../mlsegmentation"))
sys.path.append(str(pathlib.Path(__file__) / r"../mlsegmentation/src"))
import mlsegmentation.src.final_model


class ProjectElement:
    """
    Represents a BM-Segmenter project element, i.e. a case that has an image and some segmentations
    """
    def __init__(self, project: "Project", element_name: str) -> None:
        self.project = project
        self.name = element_name

        self._image_file_data_cache = None

    def image_directory_path(self) -> pathlib.Path:
        return self.project.images_directory() / self.name

    def image_file_data(self):
        # This cache is only to avoid performance issues
        if self._image_file_data_cache is None:
            self._image_file_data_cache = dict(np.load(self.image_directory_path() / "0.npz", allow_pickle=True))

        return self._image_file_data_cache

    def image(self) -> np.ndarray:
        return self.image_file_data()["matrix"]

    def mask_file_path(self, mask_name: str) -> pathlib.Path:
        return self.project.masks_directory() / mask_name / (self.name + ".npz")

    def mask_file_data(self, mask_name: str) -> dict:
        mask_file_path = self.mask_file_path(mask_name)
        if not mask_file_path.exists():
            raise FileNotFoundError(mask_file_path, "does not exist")
        return dict(np.load(mask_file_path, allow_pickle=True))

    def save_mask_file_data(self, mask_name: str, mask_file_data: dict):
        np.savez(self.mask_file_path(mask_name), **mask_file_data)

    def set_prediction_mask(self, mask_name: str, prediction_mask: np.ndarray):
        try:
            mask_file_data = self.mask_file_data(mask_name)
        except FileNotFoundError:
            mask_file_data = {}
        mask_file_data["predicted"] = prediction_mask.astype(np.uint8)
        self.save_mask_file_data(mask_name=mask_name, mask_file_data=mask_file_data)

    def rename(self, new_name: str) -> None:
        image_directory_path = self.image_directory_path()
        image_directory_path.rename(image_directory_path.with_name(new_name))

        for mask_name in self.project.mask_names():
            mask_file_path = self.mask_file_path(mask_name)
            if mask_file_path.exists():
                mask_file_path.rename(mask_file_path.with_stem(new_name))

        project_element_names = self.project.element_names()
        project_element_names.remove(self.name)
        project_element_names.append(new_name)
        self.project.set_dataset_file_element_names(project_element_names)

        self.name = new_name


class Project:
    """
    Represents a BM-segmenter project
    """
    def __init__(self, project_path: pathlib.Path) -> None:
        self.path = project_path

    def images_directory(self) -> pathlib.Path:
        return self.path / "data/dicoms"

    def masks_directory(self) -> pathlib.Path:
        return self.path / "data/masks"

    def dataset_file_path(self) -> pathlib.Path:
        return self.path / "dataset.toml"

    def mask_names(self) -> list[str]:
        return [mask_file.stem for mask_file in self.masks_directory().iterdir()]

    def element_names(self) -> list[str]:
        return self.dataset_file_data()["files"]

    def elements(self) -> list[ProjectElement]:
        return [ProjectElement(self, element_name) for element_name in self.element_names()]

    def set_dataset_file_element_names(self, element_names: list[str]):
        dataset_file_data = self.dataset_file_data()

        dataset_file_data["files"] = element_names
        groups_dict = dataset_file_data["groups"]
        groups_dict[next(iter(groups_dict))] = element_names

        self.set_dataset_file_data(dataset_file_data)

    def dataset_file_data(self):
        return toml.load(self.dataset_file_path())

    def set_dataset_file_data(self, dataset_file_data: dict):
        with self.dataset_file_path().open("w") as dataset_file_descriptor:
            toml.dump(dataset_file_data, dataset_file_descriptor)

    def add_ml_predictions(self, mask_name: str) -> None:
        images = [element.image() for element in self.elements()]
        predicted_masks = mlsegmentation.src.final_model.predict_from_images_iterable(images)
        for element, predicted_mask in zip(self.elements(), predicted_masks):
            element.set_prediction_mask(prediction_mask=predicted_mask, mask_name=mask_name)
