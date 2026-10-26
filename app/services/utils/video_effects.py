import math
import numpy as np
from moviepy import Clip, ColorClip, CompositeVideoClip, vfx
from PIL import Image


# FadeIn
def fadein_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeIn(t)])


# FadeOut
def fadeout_transition(clip: Clip, t: float) -> Clip:
    return clip.with_effects([vfx.FadeOut(t)])


# JumpCut (Instantaneous direct cut, zero transition latency)
def jump_cut_transition(clip: Clip) -> Clip:
    return clip


# CrossDissolve
def cross_dissolve_transition(clip: Clip, t: float = 0.8) -> Clip:
    dur = min(max(t, 0.1), clip.duration)
    return clip.with_effects([vfx.FadeIn(dur)])


# DipToBlack
def dip_to_black_transition(clip: Clip, t: float = 0.6) -> Clip:
    dur = min(max(t, 0.1), clip.duration / 2)
    return clip.with_effects([vfx.FadeIn(dur), vfx.FadeOut(dur)])


# DipToWhite (Flash cut)
def dip_to_white_transition(clip: Clip, t: float = 0.4) -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    white_overlay = (
        ColorClip(size=(width, height), color=(255, 255, 255))
        .with_duration(dur)
        .with_effects([vfx.FadeOut(dur)])
    )
    return CompositeVideoClip([clip, white_overlay], size=(width, height)).with_duration(
        clip.duration
    )


# RippleDissolve
def ripple_dissolve_transition(clip: Clip, t: float = 0.8) -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    y_coords = np.arange(height)
    freq = 0.04
    base_amp = min(25.0, width * 0.03)

    def ripple_effect(get_frame, current_time: float):
        frame = get_frame(current_time)
        if current_time >= dur:
            return frame
        progress = current_time / dur
        amp = base_amp * (1.0 - progress)
        shift = (amp * np.sin(freq * y_coords + current_time * 12.0)).astype(np.int32)
        grid_x = np.clip(np.arange(width)[None, :] + shift[:, None], 0, width - 1)
        rippled = frame[y_coords[:, None], grid_x]
        alpha = min(max(progress, 0.0), 1.0)
        return (rippled * alpha).astype(np.uint8)

    return clip.transform(ripple_effect)


# IrisIn (Round)
def iris_in_transition(clip: Clip, t: float = 0.8) -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    max_r = math.hypot(width / 2, height / 2)
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.hypot(x - width / 2, y - height / 2)

    def iris_in_effect(get_frame, current_time: float):
        frame = get_frame(current_time)
        if current_time >= dur:
            return frame
        progress = min(max(current_time / dur, 0.0), 1.0)
        r = max_r * progress
        edge = 10.0
        alpha = np.clip((r - dist_from_center) / edge + 0.5, 0.0, 1.0)
        return (frame * alpha[:, :, None]).astype(np.uint8)

    return clip.transform(iris_in_effect)


# IrisOut (Round)
def iris_out_transition(clip: Clip, t: float = 0.8) -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    max_r = math.hypot(width / 2, height / 2)
    y, x = np.ogrid[:height, :width]
    dist_from_center = np.hypot(x - width / 2, y - height / 2)
    start_time = max(0.0, clip.duration - dur)

    def iris_out_effect(get_frame, current_time: float):
        frame = get_frame(current_time)
        if current_time < start_time:
            return frame
        progress = min(max((clip.duration - current_time) / dur, 0.0), 1.0)
        r = max_r * progress
        edge = 10.0
        alpha = np.clip((r - dist_from_center) / edge + 0.5, 0.0, 1.0)
        return (frame * alpha[:, :, None]).astype(np.uint8)

    return clip.transform(iris_out_effect)


# IrisBox
def iris_box_transition(clip: Clip, t: float = 0.8) -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    y, x = np.ogrid[:height, :width]
    dx = np.abs(x - width / 2)
    dy = np.abs(y - height / 2)

    def iris_box_effect(get_frame, current_time: float):
        frame = get_frame(current_time)
        if current_time >= dur:
            return frame
        progress = min(max(current_time / dur, 0.0), 1.0)
        half_w = (width / 2) * progress
        half_h = (height / 2) * progress
        mask = (dx <= half_w) & (dy <= half_h)
        return (frame * mask[:, :, None]).astype(np.uint8)

    return clip.transform(iris_box_effect)


# LinearWipe
def linear_wipe_transition(clip: Clip, t: float = 0.8, direction: str = "left") -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    y, x = np.ogrid[:height, :width]

    def wipe_effect(get_frame, current_time: float):
        frame = get_frame(current_time)
        if current_time >= dur:
            return frame
        progress = min(max(current_time / dur, 0.0), 1.0)
        edge = 12.0
        if direction == "left":
            boundary = width * progress
            alpha = np.clip((boundary - x) / edge + 0.5, 0.0, 1.0)
        elif direction == "right":
            boundary = width * (1.0 - progress)
            alpha = np.clip((x - boundary) / edge + 0.5, 0.0, 1.0)
        elif direction == "top":
            boundary = height * progress
            alpha = np.clip((boundary - y) / edge + 0.5, 0.0, 1.0)
        else:
            boundary = height * (1.0 - progress)
            alpha = np.clip((y - boundary) / edge + 0.5, 0.0, 1.0)
        return (frame * alpha[:, :, None]).astype(np.uint8)

    return clip.transform(wipe_effect)


# ClockWipe (Radial angle wipe)
def clock_wipe_transition(clip: Clip, t: float = 0.8) -> Clip:
    width, height = clip.size
    dur = min(max(t, 0.1), clip.duration)
    y, x = np.ogrid[:height, :width]
    angles = (np.arctan2(x - width / 2, -(y - height / 2)) + 2 * np.pi) % (2 * np.pi)

    def clock_wipe_effect(get_frame, current_time: float):
        frame = get_frame(current_time)
        if current_time >= dur:
            return frame
        progress = min(max(current_time / dur, 0.0), 1.0)
        sweep_angle = 2 * np.pi * progress
        mask = angles <= sweep_angle
        return (frame * mask[:, :, None]).astype(np.uint8)

    return clip.transform(clock_wipe_effect)


# CrossZoom
def cross_zoom_transition(clip: Clip, t: float = 0.6) -> Clip:
    dur = min(max(t, 0.1), clip.duration)

    def cross_zoom_effect(get_frame, current_time: float):
        if current_time >= dur:
            return get_frame(current_time)
        progress = min(max(current_time / dur, 0.0), 1.0)
        ease = 1.0 - math.pow(1.0 - progress, 3)
        scale = 1.6 - 0.6 * ease
        return _zoom_frame(get_frame(current_time), scale)

    return clip.transform(cross_zoom_effect, apply_to=["mask"])


# SlideIn
def slidein_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size

    # MoviePy's built-in SlideIn is unstable for full-screen material in this pipeline,
    # causing situations where the transition is logically applied but visually imperceptible.
    # Replaced here with an explicit black background + displacement animation for visible, controllable results.
    def position(current_time: float):
        progress = min(max(current_time / max(t, 0.001), 0), 1)

        if side == "left":
            return (-width + width * progress, 0)
        if side == "right":
            return (width - width * progress, 0)
        if side == "top":
            return (0, -height + height * progress)
        if side == "bottom":
            return (0, height - height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# SlideOut
def slideout_transition(clip: Clip, t: float, side: str) -> Clip:
    width, height = clip.size
    transition_duration = min(t, clip.duration)
    transition_start = max(clip.duration - transition_duration, 0)

    # SlideOut is also replaced with explicit displacement to ensure smooth slide-out at clip end.
    def position(current_time: float):
        if current_time <= transition_start:
            return (0, 0)

        progress = min(
            max((current_time - transition_start) / max(transition_duration, 0.001), 0), 1
        )

        if side == "left":
            return (-width * progress, 0)
        if side == "right":
            return (width * progress, 0)
        if side == "top":
            return (0, -height * progress)
        if side == "bottom":
            return (0, height * progress)
        return (0, 0)

    background = ColorClip(size=(width, height), color=(0, 0, 0)).with_duration(
        clip.duration
    )
    moving_clip = clip.with_position(position)
    return CompositeVideoClip([background, moving_clip], size=(width, height)).with_duration(
        clip.duration
    )


# Retain the original 20% zoom range so even ~3s short clips have a noticeable Ken Burns motion feel.
# Zoom stability is ensured by subpixel center sampling below, rather than weakening zoom range to mask source flicker.
_ZOOM_MAX_SCALE = 1.2


def _zoom_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """Achieve borderless, stable zoom using subpixel center cropping.

    Crop dimensions cannot be cast to integers first: as the zoom ratio continuously changes,
    integer boundaries jump at varying step sizes and shift half-pixel sampling phases on parity
    changes, resulting in jitter. Pillow's EXTENT transform directly accepts floating-point boundaries,
    performing subpixel sampling on a fixed output canvas centered symmetrically around a float center.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be greater than zero")

    # Return the original frame directly at 1x zoom to avoid unnecessary resampling blur on the first frame.
    if abs(scale_factor - 1.0) < 1e-9:
        return frame

    height, width = frame.shape[:2]
    crop_width = width / scale_factor
    crop_height = height / scale_factor
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = left + crop_width
    bottom = top + crop_height

    image = Image.fromarray(frame)
    transformed = image.transform(
        (width, height),
        Image.Transform.EXTENT,
        (left, top, right, bottom),
        # Continuous video zoom prioritizes inter-frame consistency. While BICUBIC/LANCZOS are sharper
        # per frame, high-frequency textures cause ringing and shimmer across sampling grids; BILINEAR
        # is smoother, trading slight sharpness for stable dynamic motion.
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(transformed)


def zoomin_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zoom in from original frame to 1.2x across the entire clip."""
    # t is retained for signature consistency with other transition functions; zoom must cover
    # the entire clip, otherwise the frame abruptly freezes after a brief zoom.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = 1 + (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])


def zoomout_transition(clip: Clip, t: float) -> Clip:
    """Smoothly zoom out from 1.2x to original frame across the entire clip."""
    # Consistent with zoomin_transition, t is retained for interface compatibility.
    _ = t
    duration = max(clip.duration, 0.001)

    def scale_effect(get_frame, current_time: float):
        progress = min(max(current_time / duration, 0), 1)
        scale_factor = _ZOOM_MAX_SCALE - (_ZOOM_MAX_SCALE - 1) * progress
        return _zoom_frame(get_frame(current_time), scale_factor)

    return clip.transform(scale_effect, apply_to=["mask"])
