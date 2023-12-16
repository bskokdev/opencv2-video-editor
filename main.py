from collections import deque

import numpy as np
import numpy.typing as npt
import cv2 as cv
from typing import Tuple
import math

Frame_t = npt.NDArray[np.uint8]
Pixel_t = npt.NDArray[np.uint8]


class VideoEditor:
    """
    Class for editing videos, supports grayscale, chromakey, cut, shaky cam and image effects.
    We store video captures to support videos merging. Each effect call puts an event tuple into the effect queue.
    The effects and rendering are chained using the builder pattern.
    """

    def __init__(self) -> None:
        self.video_captures = []
        self.effects_queue = deque()

    def add_video(self, path: str) -> 'VideoEditor':
        """
        Create a video capture and store in the instance list of captures.
        :param path: path to the video.
        :return: self instance.
        """
        self.video_captures.append(cv.VideoCapture(path))
        return self

    def grayscale(self, start: float, end: float) -> 'VideoEditor':
        """
        Triggers grayscale effect event and puts it into the queue.
        :param start: start time in seconds of the effect in the final render.
        :param end: end time in seconds of the effect in the final render.
        :return: self instance.
        """
        self.effects_queue.append((start, end, "grayscale"))
        return self

    def chromakey(
            self,
            start: float,
            end: float,
            img: str,
            color: Tuple[int, int, int],
            similarity: int
    ) -> 'VideoEditor':
        """
        Triggers chromakey (greenscreen) effect event and puts it into the queue.
        :param start: start time in seconds of the effect in the final render.
        :param end: end time in seconds of the effect in the final render.
        #TODO(bskokdev): verify the parameters below
        :param img: path of the image
        :param color: color which will be hidden
        :param similarity: int value how similar is the color to the background
        :return: self instance.
        """
        self.effects_queue.append((start, end, img, color, similarity, "chromakey"))
        return self

    def cut(self, start: float, end: float) -> 'VideoEditor':
        """
        Triggers the cut event and puts it into the queue.
        This event is always last in the queue to apply this on top of all the other effects.
        :param start: start time in seconds of the effect in the final render.
        :param end: end time in seconds of the effect in the final render.
        :return: self instance.
        """
        self.effects_queue.append((start, end, "cut"))
        return self

    def shaky_cam(self, start: float, end: float) -> 'VideoEditor':
        """
        Triggers the shaky cam event and puts it into the queue.
        :param start: start time in seconds of the effect in the final render.
        :param end: end time in seconds of the effect in the final render.
        :return: self instance.
        """
        self.effects_queue.append((start, end, "shaky_cam"))
        return self

    def image(self, start: float, end: float, img: str, pos: Tuple[float, float, float, float]) -> 'VideoEditor':
        """
        Triggers the image event and puts it into the queue.
        :param start: start time in seconds of the effect in the final render.
        :param end: end time in seconds of the effect in the final render.
        :param img: image path of the image to display in the video over the given period.
        :param pos: coordinates of the image in the video.
        :return: self instance.
        """
        self.effects_queue.append((start, end, img, pos))
        return self

    def render(self, path: str, width: int, height: int, framerate: float, short: bool = False) -> 'VideoEditor':
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter('output.mp4', fourcc, 30.0, (width, height), False)

        # put all the videos together
        # pop current effect from the effect queue
        # convert start and end stamp to the frame index
        # iterate over all frames
        # check if the effect should be applied to the current frame
        # also display the current frame using cv.imshow to visualize the process

        return self


if __name__ == "__main__":
    (VideoEditor()
     .add_video("karlik_raw_footage.mp4")
     .cut(0, 11)
     .shaky_cam(11, 17)
     .grayscale(24, 34)
     .cut(34, 45)
     .chromakey(45, 50, "tux.jpg", (76, 120, 70), 70)
     .image(52, 60, "cat.png", (0, 0, 0.25, 0.25))
     .cut(60, 124)
     .image(120, math.inf, "tux.jpg", (0, 0, 0.5, 1))
     .shaky_cam(100, 130)
     .render("react_long.mp4", 426, 240, 25, False))
