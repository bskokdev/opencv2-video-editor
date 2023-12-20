import cv2 as cv
import numpy as np
from typing import Tuple, Union, List

# Frame related types
Frame = np.ndarray
Pixel = np.ndarray

# Effect types
Effect = Tuple[float, float, str, Union[None, List]]
Cut = Tuple[float, float]

# Effect names
EFFECT_GRAYSCALE = "grayscale"
EFFECT_CHROMAKEY = "chromakey"
EFFECT_SHAKY_CAM = "shaky_cam"
EFFECT_IMAGE = "image"


def apply_grayscale(frame: Frame) -> Frame:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


def apply_chromakey(frame: Frame, effect: Effect) -> Frame:
    # this will be done based on the pixel colorings somehow
    return frame


def apply_shaky_cam(frame: Frame) -> Frame:
    return frame


def apply_image(frame: Frame, effect: Effect) -> Frame:
    frame_width, frame_height = frame.shape[:2]

    args = effect[3]
    img_path, pos = args
    image = cv.imread(img_path)

    width_start, height_start, width_stop, height_stop = pos
    # pos tuple is given in percentage, so we convert to coordinates
    x_start = int(frame_width * width_start)
    y_start = int(frame_height * height_start)
    x_stop = int(frame_width * width_stop)
    y_stop = int(frame_height * height_stop)

    # resize, so the image fits the area
    resized_image = cv.resize(image, (x_stop - x_start, y_stop - y_start))
    frame[y_start:y_stop, x_start:x_stop] = resized_image
    return frame


# we map each effect name to the function that applies that effect to a frame
# some effects don't need all parameters, so we return a lambda function which ignores some of them
effect_callback_map = {
    EFFECT_GRAYSCALE: lambda frame, _: apply_grayscale(frame),
    EFFECT_CHROMAKEY: lambda frame, args: apply_chromakey(frame, args),
    EFFECT_SHAKY_CAM: lambda frame, _: apply_shaky_cam(frame),
    EFFECT_IMAGE: lambda frame, args: apply_image(frame, args)
}


def apply_effect(frame: Frame, effect: Effect) -> Frame:
    name = effect[2]
    # Calls the correct function which applies the effect on the given frame.
    return effect_callback_map[name](frame, effect)


def is_effect_active_in_frame(
        frame_index: int,
        effect_start_seconds: float,
        effect_end_seconds: float,
        framerate: float
) -> bool:
    # we have to convert start and end markers to frame indexes
    start_frame = int(effect_start_seconds * framerate) - 1
    end_frame = int(effect_end_seconds * framerate) - 1
    return start_frame <= frame_index <= end_frame


def are_frames_similar(frame_1: Frame, frame2: Frame) -> bool:
    if frame_1 is None or frame2 is None:
        return False
    # this will be done based on the pixel colorings
    return True


class VideoEditor:
    def __init__(self):
        self.videos = []
        self.effects: List[Effect] = []
        self.cuts: List[Cut] = []

    def add_video(self, path: str) -> "VideoEditor":
        self.videos.append(path)
        return self

    def grayscale(self, start: float, end: float) -> "VideoEditor":
        self.effects.append((start, end, EFFECT_GRAYSCALE, None))
        return self

    def chromakey(
            self,
            start: float,
            end: float,
            img_path: str,
            color: Tuple[int, int, int],
            similarity: int,
    ) -> "VideoEditor":
        self.effects.append((start, end, EFFECT_CHROMAKEY, [img_path, color, similarity]))
        return self

    def shaky_cam(self, start: float, end: float) -> "VideoEditor":
        self.effects.append((start, end, EFFECT_SHAKY_CAM, None))
        return self

    def image(
            self,
            start: float,
            end: float,
            img_path: str,
            pos: Tuple[float, float, float, float],
    ) -> "VideoEditor":
        self.effects.append((start, end, EFFECT_IMAGE, [img_path, pos]))
        return self

    def cut(self, start: float, end: float) -> "VideoEditor":
        self.cuts.append((start, end))
        return self

    def get_avg_project_framerate(self) -> float:
        framerate_sum = 0
        for video_path in self.videos:
            capture = cv.VideoCapture(video_path)
            framerate_sum += capture.get(cv.CAP_PROP_FPS)
            capture.release()
        return framerate_sum / len(self.videos)

    def should_write_frame(
            self,
            prev_frame: Frame,
            frame: Frame,
            framerate: float,
            frame_index: int,
            is_short_render: bool
    ) -> bool:
        should_write_short = is_short_render and not are_frames_similar(prev_frame, frame)
        is_frame_in_cut = any(is_effect_active_in_frame(frame_index, start, end, framerate) for start, end in self.cuts)

        # return (not is_frame_in_cut and should_write_short) or (not is_frame_in_cut and not is_short_render)
        return not is_frame_in_cut and not is_short_render

    def apply_active_effects(self, frame: Frame, frame_index: int, framerate: float) -> Frame:
        for effect in self.effects:
            start, end = effect[:2]
            if not is_effect_active_in_frame(frame_index, start, end, framerate):
                continue
            frame = apply_effect(frame, effect)
        return frame

    def render(self, output_path: str, width: int, height: int, framerate: float, short: bool = False) -> "VideoEditor":
        dim = (width, height)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, framerate, dim, True)

        project_avg_fps = self.get_avg_project_framerate()

        # for how many seconds each frame should be displayed
        project_frame_display_time = 1 / project_avg_fps
        render_frame_display_time = 1 / framerate

        prev_frame = None
        project_frame_index = elapsed_original_time = elapsed_render_time = 0
        for video_path in self.videos:
            capture = cv.VideoCapture(video_path)
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                frame = cv.resize(frame, dim)
                elapsed_original_time += project_frame_display_time
                while elapsed_render_time < elapsed_original_time:
                    elapsed_render_time += render_frame_display_time
                    frame = self.apply_active_effects(frame, project_frame_index, project_avg_fps)
                    if not self.should_write_frame(prev_frame, frame, project_avg_fps, project_frame_index, short):
                        continue

                    out.write(frame)
                    prev_frame = frame
                project_frame_index += 1

            capture.release()
        out.release()
        cv.destroyAllWindows()
        return self


if __name__ == "__main__":
    (VideoEditor()
     .add_video("test.mp4")
     .grayscale(0, 20)
     .cut(0, 10)
     .grayscale(25, 30)
     .image(25, 30, "cat.jpg", (0.5, 0, 1, 0.5))
     .render("output.mp4", 900, 600, 15))
