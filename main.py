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


def is_effect_active_in_frame(
        frame_index: int,
        effect_start_seconds: float,
        effect_end_seconds: float,
        framerate: float
) -> bool:
    # convert start and end markers to frame indexes
    start_frame = int(effect_start_seconds * framerate) - 1
    end_frame = int(effect_end_seconds * framerate) - 1
    return start_frame <= frame_index <= end_frame


def are_frames_similar(frame_1: Frame, frame2: Frame) -> bool:
    if frame_1 is None or frame2 is None:
        return False

    return True


def apply_grayscale(frame: Frame) -> Frame:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


def apply_chromakey(frame: Frame, effect: Effect) -> Frame:
    pass


def apply_shaky_cam(frame: Frame) -> Frame:
    pass


def apply_image(frame: Frame, effect: Effect) -> Frame:
    pass


# we map each effect name to the function that applies that effect to a frame
# some effects don't need all parameters, so we return a lambda function which ignores some of them
effect_callback_map = {
    EFFECT_GRAYSCALE: lambda frame, _: apply_grayscale(frame),
    EFFECT_CHROMAKEY: lambda frame, args: apply_chromakey,
    EFFECT_SHAKY_CAM: lambda frame, _: apply_shaky_cam,
    EFFECT_IMAGE: lambda frame, args: apply_image
}


def apply_effect(frame: Frame, effect: Effect) -> Frame:
    _, _, name, _ = effect
    # Calls the correct function which applies the effect on the given frame.
    return effect_callback_map[name](frame, effect)


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

    def get_project_frame_count(self) -> int:
        """Puts together frame count of the entire project"""
        total_frame_count = 0
        for video_path in self.videos:
            capture = cv.VideoCapture(video_path)
            total_frame_count += int(capture.get(cv.CAP_PROP_FRAME_COUNT))
            capture.release()
        return total_frame_count

    def get_avg_project_framerate(self) -> float:
        """Gets an average framerate of the current project"""
        fps_sum = 0
        for video_path in self.videos:
            capture = cv.VideoCapture(video_path)
            fps_sum += capture.get(cv.CAP_PROP_FPS)
            capture.release()
        return fps_sum / len(self.videos)

    def get_project_length_in_seconds(self) -> float:
        """Computes the length of the project in seconds"""
        return self.get_project_frame_count() / self.get_avg_project_framerate()

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
            start, end, _, _ = effect
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


# Example Usage
editor = VideoEditor()
editor.add_video("test.mp4").grayscale(0, 20).cut(0, 10).grayscale(25, 30).render("output.mp4", 900, 600, 15)
