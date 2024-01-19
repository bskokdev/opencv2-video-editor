# OpenCV2 Video Editor

## Description
This demonstrates working with video using opencv2 library in Python.
It allows for applying cuts and several effects to the videos as well as rendering in different framerates.

## Functionality
- Creating a video project - You can add multiple videos to form the project.
- Grayscale - This performs a grayscale effect on the desired part of the video project.
- Chromakey - Performs a "greenscreen" effect on the desired color and the part of the video
- Shaky Camera - Rotates the frames to the left and right at certain speed for a period of time.
- Adding an image - Allows you to add an image to a time period of the video. The image overlays the video
- Cut - Removes the frames on the given time period.
- Rendering - Allow you to render the final project in different framerates from the original videos with applied effects.

### Example usage

```python
if __name__ == "__main__":
    (
        VideoEditor()
        .add_video("test.mp4")
        .chromakey(0, 30, "cat.jpg", (83, 137, 75), 70)
        .cut(10, 20)
        .grayscale(25, 30)
        .image(25, 35, "cat.jpg", (0.5, 0, 1, 0.5))
        .shaky_cam(30, 40)
        .render("output.mp4", 900, 600, 10, short=False)
    )
```