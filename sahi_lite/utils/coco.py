# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.
# Modified by Sinan O Altinuc, 2020.


import logging
import os
from typing import Dict, Optional
from sahi_lite.utils.shapely import ShapelyAnnotation


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


class CocoCategory:
    """
    COCO formatted category.
    """

    def __init__(self, id=None, name=None, supercategory=None):
        self.id = int(id)
        self.name = name
        self.supercategory = supercategory if supercategory else name

    @classmethod
    def from_coco_category(cls, category):
        """
        Creates CocoCategory object using coco category.

        Args:
            category: Dict
                {"supercategory": "person", "id": 1, "name": "person"},
        """
        return cls(
            id=category["id"],
            name=category["name"],
            supercategory=category["supercategory"] if "supercategory" in category else category["name"],
        )

    @property
    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }

    def __repr__(self):
        return f"""CocoCategory<
    id: {self.id},
    name: {self.name},
    supercategory: {self.supercategory}>"""


class CocoAnnotation:
    """
    COCO formatted annotation.
    """

    @classmethod
    def from_coco_segmentation(cls, segmentation, category_id, category_name, iscrowd=0):
        """
        Creates CocoAnnotation object using coco segmentation.

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            segmentation=segmentation,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_bbox(cls, bbox, category_id, category_name, iscrowd=0):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_annotation_dict(cls, annotation_dict: Dict, category_name: Optional[str] = None):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
        """
        if annotation_dict.__contains__("segmentation") and not isinstance(annotation_dict["segmentation"], list):
            has_rle_segmentation = True
            logger.warning(
                f"Segmentation annotation for id {annotation_dict['id']} is skipped since RLE segmentation format is not supported."
            )
        else:
            has_rle_segmentation = False

        if (
                annotation_dict.__contains__("segmentation")
                and annotation_dict["segmentation"]
                and not has_rle_segmentation
        ):
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )

    @classmethod
    def from_shapely_annotation(
            cls,
            shapely_annotation: ShapelyAnnotation,
            category_id: int,
            category_name: str,
            iscrowd: int,
    ):
        """
        Creates CocoAnnotation object from ShapelyAnnotation object.

        Args:
            shapely_annotation (ShapelyAnnotation)
            category_id (int): Category id of the annotation
            category_name (str): Category name of the annotation
            iscrowd (int): 0 or 1
        """
        coco_annotation = cls(
            bbox=[0, 0, 0, 0],
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )
        coco_annotation._segmentation = shapely_annotation.to_coco_segmentation()
        coco_annotation._shapely_annotation = shapely_annotation
        return coco_annotation

    def __init__(
            self,
            segmentation=None,
            bbox=None,
            category_id=None,
            category_name=None,
            image_id=None,
            iscrowd=0,
    ):
        """
        Creates coco annotation object using bbox or segmentation

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            iscrowd: int
                0 or 1
        """
        assert bbox or segmentation, "you must provide a bbox or polygon"

        self._segmentation = segmentation
        bbox = [round(point) for point in bbox] if bbox else bbox
        self._category_id = category_id
        self._category_name = category_name
        self._image_id = image_id
        self._iscrowd = iscrowd

        shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=bbox)
        self._shapely_annotation = shapely_annotation

    @property
    def area(self):
        """
        Returns area of annotation polygon (or bbox if no polygon available)
        """
        return self._shapely_annotation.area

    @property
    def bbox(self):
        """
        Returns coco formatted bbox of the annotation as [xmin, ymin, width, height]
        """
        return self._shapely_annotation.to_coco_bbox()

    @property
    def category_id(self):
        """
        Returns category id of the annotation as int
        """
        return self._category_id

    @category_id.setter
    def category_id(self, i):
        if not isinstance(i, int):
            raise Exception("category_id must be an integer")
        self._category_id = i

    @property
    def image_id(self):
        """
        Returns image id of the annotation as int
        """
        return self._image_id

    @image_id.setter
    def image_id(self, i):
        if not isinstance(i, int):
            raise Exception("image_id must be an integer")
        self._image_id = i

    @property
    def category_name(self):
        """
        Returns category name of the annotation as str
        """
        return self._category_name

    @category_name.setter
    def category_name(self, n):
        if not isinstance(n, str):
            raise Exception("category_name must be a string")
        self._category_name = n

    @property
    def iscrowd(self):
        """
        Returns iscrowd info of the annotation
        """
        return self._iscrowd


    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoAnnotation<
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoPrediction(CocoAnnotation):
    """
    Class for handling predictions in coco format.
    """

    @classmethod
    def from_coco_bbox(cls, bbox, category_id, category_name, score, iscrowd=0, image_id=None):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            score=score,
            iscrowd=iscrowd,
            image_id=image_id,
        )

    @classmethod
    def from_coco_annotation_dict(cls, category_name, annotation_dict, score, image_id=None):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
            score: float
                Prediction score between 0 and 1
        """
        if annotation_dict["segmentation"]:
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                score=score,
                image_id=image_id,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                image_id=image_id,
            )

    def __init__(
            self,
            segmentation=None,
            bbox=None,
            category_id=None,
            category_name=None,
            image_id=None,
            score=None,
            iscrowd=0,
    ):
        """

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        self.score = score
        super().__init__(
            segmentation=segmentation,
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            iscrowd=iscrowd,
        )

    @property
    def json(self):
        return {
            "image_id": self.image_id,
            "bbox": self.bbox,
            "score": self.score,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoPrediction<
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    score: {self.score},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoImage:
    @classmethod
    def from_coco_image_dict(cls, image_dict):
        """
        Creates CocoImage object from COCO formatted image dict (with fields "id", "file_name", "height" and "weight").

        Args:
            image_dict: dict
                COCO formatted image dict (with fields "id", "file_name", "height" and "weight")
        """
        return cls(
            id=image_dict["id"],
            file_name=image_dict["file_name"],
            height=image_dict["height"],
            width=image_dict["width"],
        )

    def __init__(self, file_name: str, height: int, width: int, id: int = None):
        """
        Creates CocoImage object

        Args:
            id : int
                Image id
            file_name : str
                Image path
            height : int
                Image height in pixels
            width : int
                Image width in pixels
        """
        self.id = int(id) if id else id
        self.file_name = file_name
        self.height = int(height)
        self.width = int(width)
        self.annotations = []  # list of CocoAnnotation that belong to this image

    def add_annotation(self, annotation):
        """
        Adds annotation to this CocoImage instance

        annotation : CocoAnnotation
        """

        assert isinstance(annotation, CocoAnnotation), "annotation must be a CocoAnnotation instance"
        self.annotations.append(annotation)

    @property
    def json(self):
        return {
            "id": self.id,
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
        }

    def __repr__(self):
        return f"""CocoImage<
    id: {self.id},
    file_name: {self.file_name},
    height: {self.height},
    width: {self.width},
    annotations: List[CocoAnnotation]>"""
