from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import json
import cv2
import numpy as np
from common_code.tasks.service import get_extension

api_description = """
This service blurs the image in the given areas.
The areas are given as a list of [x1, y1, x2, y2] coordinates.
"""
api_summary = "Blurs the image in the given areas."
api_title = "Image Blur API."
version = "1.0.0"

settings = get_settings()


def clamp(val, smallest, largest):
    return max(smallest, min(val, largest))


class MyService(Service):
    """
    Image blur model
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Image Blur",
            slug="image-blur",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="image", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
                FieldDescription(name="areas", type=[FieldDescriptionType.APPLICATION_JSON]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.IMAGE_PNG, FieldDescriptionType.IMAGE_JPEG]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.IMAGE_PROCESSING,
                    acronym=ExecutionUnitTagAcronym.IMAGE_PROCESSING
                ),
            ],
            has_ai=False,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/image-blur/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        raw = data["image"].data
        input_type = data["image"].type
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)

        areas = json.loads(data["areas"].data)["areas"]

        rows = img.shape[0]
        cols = img.shape[1]

        for a in areas:
            a[0] = clamp(int(a[0]), 0, cols)
            a[1] = clamp(int(a[1]), 0, rows)
            a[2] = clamp(int(a[2]), 0, cols)
            a[3] = clamp(int(a[3]), 0, rows)

            x1, x2, y1, y2 = a[0], a[2], a[1], a[3]
            # We need to compute the blur kernel size according to the area size
            areaSize = max(x2 - x1, y2 - y1)
            kernelSize = int(areaSize * 0.08)
            img[y1:y2 + 1, x1:x2 + 1] = cv2.blur(img[y1:y2 + 1, x1:x2 + 1], (kernelSize, kernelSize))
        guessed_extension = get_extension(input_type)
        is_ok, out_buff = cv2.imencode(guessed_extension, img)

        task_data = TaskData(
            data=out_buff.tobytes(),
            type=input_type
        )

        return {
            "result": task_data
        }
