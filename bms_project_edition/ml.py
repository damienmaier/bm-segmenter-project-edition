import sys
import pathlib
from bms_project_edition import project

sys.path.append(str(pathlib.Path(__file__).parent.parent / 'mlsegmentation'))
sys.path.append(str(pathlib.Path(__file__).parent.parent / 'mlsegmentation' / 'src'))
import final_model


def compute_mask_predictions_from_ml_model(project_path: str, mask_name: str) -> None:
    """
    For a BM-segmenter project, computes the muscle mask predictions using the ml model, and saves them in the project.
    """
    bms_project = project.Project(pathlib.Path(project_path))

    images = [element.image() for element in bms_project.elements()]
    predicted_masks = final_model.predict_from_images_iterable(images)
    for element, predicted_mask in zip(bms_project.elements(), predicted_masks):
        element.set_prediction_mask(prediction_mask=predicted_mask, mask_name=mask_name)
