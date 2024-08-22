import project
import csv
import pathlib


def write_measurement_csv_for_bmsegmenter_project(
        project_path: str, mask_name: str,
        restricted_hu_range_min: int, restricted_hu_range_max: int
) -> None:
    bms_project = project.Project(pathlib.Path(project_path))

    with open(bms_project.path / f"{mask_name}_measurements.csv", "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Name",
            "Area [full] (cm^2)",
            "Area [restricted] (cm^2)",
            "Mean [full] (HU)",
            "Mean [restricted] (HU)",
            "IMAT (cm^2)"
        ])

        for element in bms_project.elements():

            if not element.is_validated(mask_name):
                csv_writer.writerow([element.name_prefix(), "N/A", "N/A", "N/A", "N/A", "N/A"])
            else:
                image = element.image()
                project_mask = element.validated_mask(mask_name).astype(bool)

                restricted_project_mask = (
                        (image >= restricted_hu_range_min) &
                        (image <= restricted_hu_range_max) &
                        project_mask
                )

                imat_project_mask = (
                        (image >= -190) &
                        (image <= -31) &
                        project_mask
                )

                pixel_area_cm2 = element.pixel_dimensions_mm()[0] * element.pixel_dimensions_mm()[1] / 100

                project_mask_area = project_mask.sum() * pixel_area_cm2
                restricted_project_mask_area = restricted_project_mask.sum() * pixel_area_cm2
                imat_project_mask_area = imat_project_mask.sum() * pixel_area_cm2

                project_mask_mean = image[project_mask].mean()
                restricted_project_mask_mean = image[restricted_project_mask].mean()

                csv_writer.writerow([
                    element.name_prefix(),
                    round(project_mask_area, 3),
                    round(restricted_project_mask_area, 3),
                    round(project_mask_mean, 3),
                    round(restricted_project_mask_mean, 3),
                    round(imat_project_mask_area, 3)
                ])
