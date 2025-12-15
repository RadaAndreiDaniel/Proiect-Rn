from datetime import datetime
import json
from typing import Optional
import sys

import cv2
import numpy as np
from treadscan.extractor import TireModel
from treadscan.utilities import *


class Annotator:

    def __init__(self, image: np.ndarray, max_width: int, max_height: int):

        self.image = image.copy()
        self.scale = min(max_width / image.shape[1], max_height / image.shape[0])
        self.scale = min(self.scale, 1)
        self.image = scale_image(self.image, self.scale)
        self._original_image = self.image.copy()

        # convert to BGR if grayscale
        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        self.tire_model = TireModel((self.image.shape[0], self.image.shape[1]))

        self.mouse_pos = None
        self.points = {
            ord('t'): None,
            ord('b'): None,
            ord('r'): None,
            ord('s'): None,
            ord('w'): None
        }

        self.__prev_top = (0, 0)
        self.__prev_bottom = (0, 0)

        # ★★★ HERE WE STORE THE PREVIEW FROM GUI
        self._last_preview = None


    def draw_only_annotation_points(self, image: np.ndarray):
        point_size = 5
        y_pos = 8
        for key, value in self.points.items():
            if value is not None:
                cv2.circle(image, value, point_size, (0, 128, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                point = (value[0] - 4, value[1] + 4)
                cv2.putText(image, chr(key), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(image, (4, y_pos), point_size, (0, 128, 255), cv2.FILLED, lineType=cv2.LINE_AA)
                cv2.putText(image, chr(key), (0, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            1, cv2.LINE_AA)
                y_pos += 16


    def draw(self, image: np.ndarray) -> bool:
        top = self.points[ord('t')]
        bottom = self.points[ord('b')]
        right = self.points[ord('r')]
        sidewall = self.points[ord('s')]
        width = self.points[ord('w')]

        if top and bottom and top[1] > bottom[1]:
            top = (top[0], bottom[1])
            bottom = (bottom[0], top[1])
            self.points[ord('t')] = top
            self.points[ord('b')] = bottom

        if top and bottom and right:
            cx, cy = (bottom[0] + top[0]) / 2, (bottom[1] + top[1]) / 2
            h = euclidean_dist(top, bottom)
            w = 2 * euclidean_dist(right, (cx, cy))
            w += (sys.float_info.epsilon if w == 0 else 0)

            if int(w) > int(h):
                t = h / w
                right = (int((1 - t) * cx + t * right[0]), int((1 - t) * cy + t * right[1]))
                self.points[ord('r')] = right

            epsilon = 1
            if -epsilon < (euclidean_dist(top, right) +
                           euclidean_dist(right, bottom) -
                           euclidean_dist(top, bottom)) < epsilon:
                self.points[ord('r')] = right

            ellipse = ellipse_from_points(top, bottom, right)
            if ellipse.height == 0 or ellipse.width == 0:
                self.draw_only_annotation_points(image)
                return False

            (center, axes, angle, startAngle, endAngle) = ellipse.cv2_ellipse()
            cv2.ellipse(image, center, axes, angle, startAngle, endAngle, (0, 0, 255), 2)

            if sidewall and width:
                self.tire_model.from_keypoints(top, bottom, right, sidewall, width)
                top_left, bottom_right = self.tire_model.bounding_box()
                cv2.rectangle(image, top_left, bottom_right, (255, 255, 0), 2, cv2.LINE_AA)
                self.tire_model.draw(image, (0, 255, 255), 2, cv2.LINE_AA)
                self.draw_only_annotation_points(image)
                return True

        self.draw_only_annotation_points(image)
        return False


    def annotate_keypoints(self) -> str:

        self.points = {ord('t'): None, ord('b'): None, ord('r'): None, ord('s'): None, ord('w'): None}

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                self.mouse_pos = (x, y)

        submitted = False
        bounding_boxes = []
        keypoints = []

        def save_keypoints():
            top = self.points[ord('t')]
            bottom = self.points[ord('b')]
            right = self.points[ord('r')]
            sidewall = self.points[ord('s')]
            width = self.points[ord('w')]

            self.points = {ord('t'): None, ord('b'): None, ord('r'): None, ord('s'): None, ord('w'): None}

            top_left, bottom_right = self.tire_model.bounding_box()
            rect = self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            colored = np.zeros(rect.shape, np.uint8)
            cv2.rectangle(colored, (0, 0), (colored.shape[1], colored.shape[0]), (255, 255, 0), -1)
            rect = cv2.addWeighted(rect, 0.5, colored, 0.5, 1.0)
            self.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] = rect

            # convert coords
            top = (int(top[0]/self.scale), int(top[1]/self.scale), 1)
            bottom = (int(bottom[0]/self.scale), int(bottom[1]/self.scale), 1)
            right = (int(right[0]/self.scale), int(right[1]/self.scale), 1)
            sidewall = (int(sidewall[0]/self.scale), int(sidewall[1]/self.scale), 1)
            width = (int(width[0]/self.scale), int(width[1]/self.scale), 1)

            tl = (int(top_left[0]/self.scale), int(top_left[1]/self.scale))
            br = (int(bottom_right[0]/self.scale), int(bottom_right[1]/self.scale))

            keypoints.append([top, bottom, right, sidewall, width])
            bounding_boxes.append([*tl, *br])


        show_preview = True

        while True:

            image = self.image.copy()
            success = self.draw(image)

            tread = None
            if success and show_preview:
                try:
                    tread = self.tire_model.unwrap(self._original_image)

                    if len(tread.shape) == 3:
                        tread = cv2.cvtColor(tread, cv2.COLOR_BGR2GRAY)
                    tread = remove_gradient(tread)
                    tread = clahe(tread)
                    tread = cv2.cvtColor(tread, cv2.COLOR_GRAY2BGR)

                    # ★★★ SAVE PREVIEW IMAGE HERE ★★★
                    self._last_preview = tread.copy()

                except RuntimeError:
                    tread = None

            if tread is not None:
                h1, w1 = image.shape[:2]
                h2, w2 = tread.shape[:2]
                if h2 > h1 or w2 > w1:
                    scale = min(w1/w2, h1/h2)
                    scale = min(scale, 1)
                    tread = scale_image(tread, scale)
                    h2, w2 = tread.shape[:2]

                image[0:h2, w1 - w2:w1, :] = tread

            cv2.imshow("Image", image)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("Image", mouse_callback)
            key = cv2.waitKey(100)

            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1 or key == 27:
                break
            elif key == 13:
                if success:
                    save_keypoints()
                submitted = True
                break
            elif key == ord(' '):
                show_preview = True
            elif key == 8:
                show_preview = False
            elif key != -1 and key in self.points:
                self.points[key] = self.mouse_pos
            elif key == ord('n') and success:
                save_keypoints()
                show_preview = True

        cv2.destroyAllWindows()

        if submitted and keypoints and bounding_boxes:
            return json.dumps({"bboxes": bounding_boxes, "keypoints": keypoints})
        else:
            return ""
