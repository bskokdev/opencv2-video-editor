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

# Flag for alternating the shaky cam rotation degree (for positive or negative degree)
shaky_cam_flag = True


def apply_grayscale(frame: Frame) -> Frame:
    """
    Applies grayscale effect on the frame.

    :param frame: Frame we apply the grayscale effect on.
    :return: Grayscale BGR frame to allow colorful display.
    """
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


def apply_chromakey(frame: Frame, args: Tuple[str, Tuple[int, int, int], int]) -> Frame:
    """
    Applies chromakey effect on the current frame. We replace all the pixels in frame for which the color difference
    between pixel and given color < similarity with the given image pixel.

    :param frame: Frame to perform the chromakey effect on.
    :param args: Tuple of (imp_path, rgb_color, similarity treshold) values
    :return: Chromakey-ed frame.
    """
    img_path, rgb_color, similarity = args
    width, height, channels = frame.shape

    # read img from path and resize to match the frame dimensions
    img = cv.imread(img_path)
    img = cv.resize(img, (height, width))

    # input color is in RGB format, so we convert the frame to RGB as well
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # get total abs difference sum of img pixels and input color; axis=2 means color channel
    color_diff = np.sum(np.abs(frame_rgb.astype(int) - np.array(rgb_color).astype(int)), axis=2)

    # Replace pixels in frame where color difference is below threshold; GPT helped with the mask hack
    mask = color_diff < similarity
    frame[mask] = img[mask]

    return frame


def are_frames_similar_in_color(frame_1: Frame, frame_2: Frame, similarity_treshold_percent: Union[int, float]) -> bool:
    """
    Checks color similarity of 2 given frames by using numpy operations for faste processing.
    We basically just take an average of the sum of absolute value from differences between 2 pixels.
    Then we check if such difference is < than the similarity treshold (in percent).

    :param frame_1: First frame to be compared.
    :param frame_2: Second frame to be compared.
    :param similarity_treshold_percent: Threshold of allowed similarity.
    :return: True if color similarity between 2 frames > threshold, else False.
    """
    if frame_1 is None or frame_2 is None:
        return False

    # Calculate the color difference between 2 frames
    color_diff = np.sum(np.abs(frame_1.astype(int) - frame_2.astype(int)), axis=2)
    avg_diff = np.mean(color_diff)

    # Calculate similarity in percent
    similarity = (1 - avg_diff / 255) * 100

    return similarity > similarity_treshold_percent


def apply_shaky_cam(frame: Frame, start_index: int, end_index: int, frame_index: int) -> Frame:
    """
    Applies shaking cam effect on a frame. This basically means we apply different rotations on the frame.
    The rotations are alternating based on the global shaky_cam_flag.
    If the flag is True -> rotation is -5 degrees, else 5 degrees.

    :param frame: Frame to apply the shaking cam effect on.
    :param start_index: Index of the frame on which the shaking effect starts.
    :param end_index: Index of the frame on which the shaking effect ends.
    :param frame_index: Index of the current frame we are modifying.
    :return: Frame with applied rotation depending on the shaky_cam_flag
    """

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


"""
We map each effect name to the function that applies that effect to a frame.
Each effect function has access to frame, frame index, effect start index, effect end index, and effect arguments.
"""
effect_callback_map = {
    EFFECT_GRAYSCALE: lambda frame, frame_index, start, end, args: apply_grayscale(frame),
    EFFECT_CHROMAKEY: lambda frame, frame_index, start, end, args: apply_chromakey(frame, args),
    EFFECT_SHAKY_CAM: lambda frame, frame_index, start, end, args: apply_shaky_cam(frame, start, end, frame_index),
    EFFECT_IMAGE: lambda frame, frame_index, start, end, args: apply_image(frame, args)
}


def apply_effect(frame: Frame, frame_index: int, effect: Effect, framerate: float) -> Frame:
    """
    Calls the correct function which applies the effect on the given frame.

    :param frame: Frame we want to apply an effect on.
    :param effect: Effect to be applied.
    :param frame_index: Current frame index.
    :param framerate: Project framerate.
    :return: Frame with applied effect.
    """
    start, end, effect_type, args = effect
    start, end = convert_effect_period_to_frame_index(start, end, framerate)
    return effect_callback_map[effect_type](frame, frame_index, start, end, args)


def is_effect_active_in_frame(
        frame_index: int,
        effect_start_seconds: float,
        effect_end_seconds: float,
        framerate: float
) -> bool:
    """
    Checks if the frame index is in an interval of an effect.
    The effect interval has to be converted to frame indexes first.

    :param frame_index: Current frame index.
    :param effect_start_seconds: Start of the effect in seconds.
    :param effect_end_seconds: End of the effect in seconds.
    :param framerate: Framerate of the project.
    :return: True if frame index is within the effect period, otherwise False.
    """
    start_frame, end_frame = convert_effect_period_to_frame_index(effect_start_seconds, effect_end_seconds, framerate)
    return start_frame <= frame_index <= end_frame


def convert_effect_period_to_frame_index(start_seconds: float, end_seconds: float, framerate: float) -> Tuple[int, int]:
    """
    Start and end effect periods are given in seconds, so this function converts them to frame indexes.

    :param start_seconds: Start of the effect in seconds.
    :param end_seconds: End of the effect in seconds.
    :param framerate: Project framerate.
    :return: (start, end) period as frame indexes.
    """
    start_frame = int(start_seconds * framerate) - 1
    end_frame = int(end_seconds * framerate) - 1
    return start_frame, end_frame


class VideoEditor:
    """
    Video editor class which allows adding videos to the project, applying effects, cutting intervals.
    The resulting video can be rendered in different framerates without changing the length of the video.
    """

    def __init__(self):
        self.videos = []
        self.effects: List[Effect] = []
        self.cuts: List[Cut] = []

    def add_video(self, path: str) -> "VideoEditor":
        """
        Add the video on specified path to the current video project. Video project is denoted as list of video paths.

        :param path: Path to the location of the video to be added to the project.
        :return: Self instance for builder.
        """
        self.videos.append(path)
        return self

    def grayscale(self, start: float, end: float) -> "VideoEditor":
        """
        Event dispatch which specifies there's grayscale effect applied on frame in (start, end) interval frames.

        :param start: Start of the grayscale effect in seconds.
        :param end: End of the grayscale effect in seconds.
        :return: Self instance for builder.
        """
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
        """
        Event dispatch which specifies there's chromakey effect applied on frame in (start, end) interval frames.

        :param start: Start of the chromakey effect in seconds.
        :param end: End of the chromakey effect in seconds.
        :param img_path: Path to the image we will display.
        :param color: RGB color to be chromakey-ed.
        :param similarity: We chromakey every pixel which color difference < similarity (in percent).
        :return: Self instance for builder.
        """
        self.effects.append((start, end, EFFECT_CHROMAKEY, (img_path, color, similarity)))
        return self

    def shaky_cam(self, start: float, end: float) -> "VideoEditor":
        """
        Event dispatch which specifies there's a shaking cam effect on frames in range of (start, end) interval.

        :param start: Start of the shaking in seconds.
        :param end: End of the shaking in seconds.
        :return: Self instance for builder.
        """
        self.effects.append((start, end, EFFECT_SHAKY_CAM, None))
        return self

    def image(
            self,
            start: float,
            end: float,
            img_path: str,
            pos: Tuple[float, float, float, float],
    ) -> "VideoEditor":
        """
        Event dispatch which specifies there's an image placed on frames in (start, end) interval.

        :param start: Start of the image display in seconds.
        :param end: End of the image display in seconds.
        :param img_path: Path to the image to be displayed on the frames.
        :param pos: Position in percent on the frames - (width_start, height_start, width_stop, height_stop)
        :return: Self instance for builder.
        """
        self.effects.append((start, end, EFFECT_IMAGE, (img_path, pos)))
        return self

    def cut(self, start: float, end: float) -> "VideoEditor":
        """
        Event dispatch which specifies there's a cut on (start, end) interval -> frame will be skipped.

        :param start: Cut start in seconds.
        :param end: Cut end in seconds.
        :return: Self instance for builder.
        """
        self.cuts.append((start, end))
        return self

    def get_avg_project_framerate(self) -> float:
        """
        Puts together fps sum of all the videos in the project and calculates the average.

        :return: The average framerate of the entire video project or 0 if no videos added
        """
        framerate_sum = 0
        video_count = len(self.videos)
        for video_path in self.videos:
            capture = cv.VideoCapture(video_path)
            framerate_sum += capture.get(cv.CAP_PROP_FPS)
            capture.release()
        return framerate_sum / video_count if video_count > 0 else 0

    def should_write_frame(
            self,
            prev_frame: Frame,
            frame: Frame,
            frame_index: int,
            framerate: float,
            is_short_render: bool
    ) -> bool:
        """
        Checks if the current frame should be written to the output video.
        We want to write frames which are either not in cuts and not similar to the previous one if the short == True,
        or the purely frames which are not in cuts if the short == False.

        :param prev_frame: Last written frame to the output video.
        :param frame: Current frame.
        :param framerate: Frames per second of the video project.
        :param frame_index: Index of the current frame.
        :param is_short_render: Is the render in short version.
        :return: True if we want the current frame to be written to the output, otherwise False.
        """
        should_write_short = is_short_render and not are_frames_similar_in_color(prev_frame, frame, 90)
        is_frame_in_cut = any(is_effect_active_in_frame(frame_index, start, end, framerate) for start, end in self.cuts)

        return (not is_frame_in_cut and should_write_short) or (not is_frame_in_cut and not is_short_render)

    def apply_active_effects(self, frame: Frame, frame_index: int, framerate: float) -> Frame:
        """
        Goes through all called effects and applies those which are active for the current frame index.

        :param frame: Frame to apply the effects on.
        :param frame_index: current frame index.
        :param framerate: Project fps.
        :return: Modified frame with applied effects.
        """
        for effect in self.effects:
            start, end = effect[:2]
            if not is_effect_active_in_frame(frame_index, start, end, framerate):
                continue
            modified_frame = apply_effect(frame, frame_index, effect, framerate)
            if modified_frame is not None:
                frame = modified_frame

        return frame

    def render(self, output_path: str, width: int, height: int, framerate: float, short: bool = False) -> "VideoEditor":
        """
        Renders the current video project in desired dimensions and framerate.
        The framerate is handled by keeping track of the elapsed time for both the project and rendered video.
        The rendered video has frame skipping based on the color similarity if short == True.

        :param output_path: Output video file location.
        :param width: Desired with of the rendered video.
        :param height: Desired height of the rendered video.
        :param framerate: Frames per second in the rendered video.
        :param short: Should the video skip frames based on similarity.
        :return: Self instance.
        """
        dim = (width, height)
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(output_path, fourcc, framerate, dim, True)

        project_avg_fps = self.get_avg_project_framerate()

        # for how many seconds each frame should be displayed
        project_frame_display_time = 1 / project_avg_fps
        render_frame_display_time = 1 / framerate

        # These handle the framerate matching
        elapsed_original_time = elapsed_render_time = 0

        prev_frame = None
        project_frame_index = 0
        for video_path in self.videos:
            capture = cv.VideoCapture(video_path)
            # Read ith video of the current project
            while True:
                ret, frame = capture.read()
                if not ret:
                    break

                frame = cv.resize(frame, dim)
                elapsed_original_time += project_frame_display_time
                while elapsed_render_time < elapsed_original_time:
                    elapsed_render_time += render_frame_display_time
                    frame = self.apply_active_effects(frame, project_frame_index, project_avg_fps)
                    if not self.should_write_frame(prev_frame, frame, project_frame_index, project_avg_fps, short):
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
     .chromakey(0, 5, "cat.jpg", (83, 137, 75), 70)
     .cut(10, 20)
     .grayscale(25, 30)
     .image(25, 35, "cat.jpg", (0.5, 0, 1, 0.5))
     .shaky_cam(30, 40)
     .render("output.mp4", 900, 600, 10, short=False))
