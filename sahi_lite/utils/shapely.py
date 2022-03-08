# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

from typing import List


def get_bbox_from_shapely(shapely_object):
    """
    Accepts shapely box/poly object and returns its bounding box in coco and voc formats
    """
    minx, miny, maxx, maxy = shapely_object.bounds
    width = maxx - minx
    height = maxy - miny
    coco_bbox = [minx, miny, width, height]
    coco_bbox = [round(point) for point in coco_bbox] if coco_bbox else coco_bbox
    voc_bbox = [minx, miny, maxx, maxy]
    voc_bbox = [round(point) for point in voc_bbox] if voc_bbox else voc_bbox

    return coco_bbox, voc_bbox


class ShapelyAnnotation:
    """
    Creates ShapelyAnnotation (as shapely MultiPolygon).
    Can convert this instance annotation to various formats.
    """

    def __init__(self, slice_bbox=None):
        self.slice_bbox = slice_bbox

    @property
    def area(self):
        return int(self.__area)

    def to_list(self):
        """
        [
            [(x1, y1), (x2, y2), (x3, y3), ...],
            [(x1, y1), (x2, y2), (x3, y3), ...],
            ...
        ]
        """
        list_of_list_of_points: List = []
        for shapely_polygon in self.multipolygon.geoms:
            # create list_of_points for selected shapely_polygon
            if shapely_polygon.area != 0:
                x_coords = shapely_polygon.exterior.coords.xy[0]
                y_coords = shapely_polygon.exterior.coords.xy[1]
                # fix coord by slice_bbox
                if self.slice_bbox:
                    minx = self.slice_bbox[0]
                    miny = self.slice_bbox[1]
                    x_coords = [x_coord - minx for x_coord in x_coords]
                    y_coords = [y_coord - miny for y_coord in y_coords]
                list_of_points = list(zip(x_coords, y_coords))
            else:
                list_of_points = []
            # append list_of_points to list_of_list_of_points
            list_of_list_of_points.append(list_of_points)
        # return result
        return list_of_list_of_points

    def to_opencv_contours(self):
        """
        [
            [[[1, 1]], [[325, 125]], [[250, 200]], [[5, 200]]],
            [[[1, 1]], [[325, 125]], [[250, 200]], [[5, 200]]]
        ]
        """
        opencv_contours: List = []
        for shapely_polygon in self.multipolygon.geoms:
            # create opencv_contour for selected shapely_polygon
            if shapely_polygon.area != 0:
                x_coords = shapely_polygon.exterior.coords.xy[0]
                y_coords = shapely_polygon.exterior.coords.xy[1]
                # fix coord by slice_bbox
                if self.slice_bbox:
                    minx = self.slice_bbox[0]
                    miny = self.slice_bbox[1]
                    x_coords = [x_coord - minx for x_coord in x_coords]
                    y_coords = [y_coord - miny for y_coord in y_coords]
                opencv_contour = [[[int(x_coords[ind]), int(y_coords[ind])]] for ind in range(len(x_coords))]
            else:
                opencv_contour: List = []
            # append opencv_contour to opencv_contours
            opencv_contours.append(opencv_contour)
        # return result
        return opencv_contours

    def to_coco_bbox(self):
        """
        [xmin, ymin, width, height]
        """
        if self.multipolygon.area != 0:
            coco_bbox, _ = get_bbox_from_shapely(self.multipolygon)
            # fix coord by slice box
            if self.slice_bbox:
                minx = round(self.slice_bbox[0])
                miny = round(self.slice_bbox[1])
                coco_bbox[0] = round(coco_bbox[0] - minx)
                coco_bbox[1] = round(coco_bbox[1] - miny)
        else:
            coco_bbox: List = []
        return coco_bbox

    def to_voc_bbox(self):
        """
        [xmin, ymin, xmax, ymax]
        """
        if self.multipolygon.area != 0:
            _, voc_bbox = get_bbox_from_shapely(self.multipolygon)
            # fix coord by slice box
            if self.slice_bbox:
                minx = self.slice_bbox[0]
                miny = self.slice_bbox[1]
                voc_bbox[0] = round(voc_bbox[0] - minx)
                voc_bbox[2] = round(voc_bbox[2] - minx)
                voc_bbox[1] = round(voc_bbox[1] - miny)
                voc_bbox[3] = round(voc_bbox[3] - miny)
        else:
            voc_bbox = []
        return voc_bbox
