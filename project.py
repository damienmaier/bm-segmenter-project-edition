import pathlib

import toml

IMAGES_DIRECTORY = "data/dicoms"


class ProjectElement:
    def __init__(self, project: "Project", element_name: str) -> None:
        self.project = project
        self.element_name = element_name

    def image_directory_path(self) -> pathlib.Path:
        return self.project.images_directory() / self.element_name

    def mask_file_path(self, mask_name) -> pathlib.Path:
        return self.project.masks_directory() / mask_name / (self.element_name + ".npz")

    def rename(self, new_name: str) -> None:
        image_directory_path = self.image_directory_path()
        image_directory_path.rename(image_directory_path.with_name(new_name))

        for mask_name in self.project.mask_names():
            mask_file_path = self.mask_file_path(mask_name)
            if mask_file_path.exists():
                mask_file_path.rename(mask_file_path.with_stem(new_name))

        dataset_file_data = self.project.dataset_file_data()

        dataset_file_files_list = dataset_file_data["files"]
        dataset_file_files_list.remove(self.element_name)
        dataset_file_files_list.append(new_name)

        dataset_file_group_list = next(iter(dataset_file_data["groups"].values()))
        dataset_file_group_list.remove(self.element_name)
        dataset_file_files_list.append(new_name)

        self.project.set_dataset_file_data(dataset_file_data)

        self.element_name = new_name


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

    def dataset_file_data(self):
        return toml.load(self.dataset_file_path())

    def set_dataset_file_data(self, dataset_file_data: dict):
        with self.dataset_file_path().open("w") as dataset_file_descriptor:
            toml.dump(dataset_file_data, dataset_file_descriptor)

