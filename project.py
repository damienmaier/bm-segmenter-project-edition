import pathlib

import numpy as np
import toml


class ProjectElement:
    def __init__(self, project: "Project", element_name: str) -> None:
        self.project = project
        self.name = element_name

        self._image_file_data_cache = None

    def image_directory_path(self) -> pathlib.Path:
        return self.project.images_directory() / self.name

    def image_file_data(self):
        if self._image_file_data_cache is None:
            self._image_file_data_cache = dict(np.load(self.image_directory_path() / "0.npz"))

        return self._image_file_data_cache

    def image(self):
        return self.image_file_data()["matrix"]

    def mask_file_path(self, mask_name) -> pathlib.Path:
        return self.project.masks_directory() / mask_name / (self.name + ".npz")

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
