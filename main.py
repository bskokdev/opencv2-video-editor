import cv2 as cv
import numpy as np
from typing import Tuple, Union, List

# Frame related types
Frame = np.ndarray
Pixel = np.ndarray

# Effect typing
Effect = Tuple[float, float, str, Union[None, Tuple]]
Cut = Tuple[float, float]

# Effect types
EFFECT_GRAYSCALE = "grayscale"
EFFECT_CHROMAKEY = "chromakey"
EFFECT_SHAKY_CAM = "shaky_cam"
EFFECT_IMAGE = "image"

# Flag for alternating the shaky cam rotation degree (positive || negative)
shaky_cam_flag = True


def apply_grayscale(frame: Frame) -> Frame:
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


def apply_chromakey(frame: Frame, args: Tuple[str, Tuple[int, int, int], int]) -> Frame:
    img_path, rgb_color, similarity = args
    width, height, _ = frame.shape

    # read img from path and resize to match the frame dimensions
    img = cv.imread(img_path)
    img = cv.resize(img, (height, width))

    # input color is in RGB format, so we convert to BRG
    bgr_color = np.array(rgb_color[::-1])

    # get total abs difference sum of img pixels and input color; axis=2 means color channel
    color_diff = np.sum(np.abs(img.astype(int) - bgr_color.astype(int)), axis=2)
    color_diff_percent = color_diff / (3 * 255) * 100

    # Replace pixels in frame where color difference is below threshold
    mask = color_diff_percent < similarity
    frame[mask] = img[mask]
    return frame


def apply_shaky_cam(frame: Frame, start_index: int, end_index: int, frame_index: int) -> Frame:
    if frame_index == start_index or frame_index == end_index:
        return frame

    rotation_degree = 5
    """
    Since we can't alternate based on the even/odd (because of possibly skipping frames we could get only odd frames),
    we keep the snaky_cam_flag to alternate between positive and negative degree, it's kind of hacky but w/e.
    """
    global shaky_cam_flag
    if shaky_cam_flag:
        rotation_degree = -rotation_degree

    # create a rotation matrix; GPT helped here quite a bit
    height, width, _ = frame.shape
    center_coordinates = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center_coordinates, rotation_degree, 1)

    # Perform the rotation
    rotated_frame = cv.warpAffine(frame, rotation_matrix, (width, height))
    shaky_cam_flag = not shaky_cam_flag
    return rotated_frame


def apply_image(frame: Frame, args: Tuple[str, Tuple[float, float, float, float]]) -> Frame:
    frame_width, frame_height, _ = frame.shape

    img_path, pos = args
    image = cv.imread(img_path)
    width_start, height_start, width_stop, height_stop = pos

    # pos tuple is given in percentage, so we convert to coordinates
    x_start = int(frame_width * width_start)
    y_start = int(frame_height * height_start)
    x_stop = int(frame_width * width_stop)
    y_stop = int(frame_height * height_stop)

    # resize, so the image fits the area
    dim = (x_stop - x_start, y_stop - y_start)
    resized_image = cv.resize(image, dim)
    frame[y_start:y_stop, x_start:x_stop] = resized_image
    return frame


# we map each effect name to the function that applies that effect to a frame
effect_callback_map = {
    EFFECT_GRAYSCALE: lambda frame, frame_index, start, end, args: apply_grayscale(frame),
    EFFECT_CHROMAKEY: lambda frame, frame_index, start, end, args: apply_chromakey(frame, args),
    EFFECT_SHAKY_CAM: lambda frame, frame_index, start, end, args: apply_shaky_cam(frame, start, end, frame_index),
    EFFECT_IMAGE: lambda frame, frame_index, start, end, args: apply_image(frame, args)
}


def apply_effect(frame: Frame, effect: Effect, frame_index: int, framerate: float) -> Frame:
    start, end, effect_type, args = effect
    start, end = convert_effect_period_to_frame_index(start, end, framerate)
    # Calls the correct function which applies the effect on the given frame.
    return effect_callback_map[effect_type](frame, frame_index, start, end, args)


def is_effect_active_in_frame(
        frame_index: int,
        effect_start_seconds: float,
        effect_end_seconds: float,
        framerate: float
) -> bool:
    start_frame, end_frame = convert_effect_period_to_frame_index(effect_start_seconds, effect_end_seconds, framerate)
    return start_frame <= frame_index <= end_frame


def convert_effect_period_to_frame_index(start_seconds: float, end_seconds: float, framerate: float) -> Tuple[int, int]:
    start_frame = int(start_seconds * framerate) - 1
    end_frame = int(end_seconds * framerate) - 1
    return start_frame, end_frame


def are_frames_similar(frame_1: Frame, frame_2: Frame, percentage_treshold: Union[int, float]) -> bool:
    if frame_1 is None or frame_2 is None:
        return False

    # Calculate color difference for the entire array
    color_diff = np.sum(np.abs(frame_1.astype(int) - frame_2.astype(int)), axis=2)
    avg_diff = np.mean(color_diff)

    # Calculate similarity in percent
    similarity = (1 - avg_diff / 255) * 100
    return similarity > percentage_treshold


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
        self.effects.append((start, end, EFFECT_CHROMAKEY, (img_path, color, similarity)))
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
        self.effects.append((start, end, EFFECT_IMAGE, (img_path, pos)))
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
        should_write_short = is_short_render and not are_frames_similar(prev_frame, frame, 90)
        is_frame_in_cut = any(is_effect_active_in_frame(frame_index, start, end, framerate) for start, end in self.cuts)

        return (not is_frame_in_cut and should_write_short) or (not is_frame_in_cut and not is_short_render)

    def apply_active_effects(self, frame: Frame, frame_index: int, framerate: float) -> Frame:
        for effect in self.effects:
            start, end = effect[:2]
            if not is_effect_active_in_frame(frame_index, start, end, framerate):
                continue
            frame = apply_effect(frame, effect, frame_index, framerate)
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
     .chromakey(0, 5, "cat.jpg", (0, 0, 0), 90)
     .grayscale(0, 20)
     .cut(0, 10)
     .grayscale(25, 30)
     .image(25, 35, "cat.jpg", (0.5, 0, 1, 0.5))
     .shaky_cam(30, 40)
     .render("output.mp4", 900, 600, 10, short=False))
