"""Synchronized silent GIF recording for a measured robot episode."""

from __future__ import annotations

import bisect
import copy
import io
import json
import math
import pathlib
import time
from typing import Any

import numpy as np
from PIL import Image as pil_image
from sensor_msgs import msg as sensor

from hrl_tl.robot_demo import pose
from hrl_tl.robot_demo.world import arena, native_view


class EpisodeRecorder:
    """Gazebo camera and native-board samples for one measured episode."""

    def __init__(
        self,
        layout: arena.Arena,
        output: pathlib.Path,
        *,
        duration: float = 20.0,
        fps: int = 5,
    ) -> None:
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("GIF duration must be finite and positive")
        if fps <= 0:
            raise ValueError("GIF fps must be positive")
        self._layout = layout
        self._output = output
        self._duration = duration
        self._fps = fps
        self._interval = 1 / fps
        self._board: list[tuple[float, tuple[pose.Pose2D, dict[str, Any]]]] = []
        self._gazebo: list[tuple[float, bytes]] = []
        self._last_board_at = float("-inf")
        self._last_gazebo_at = float("-inf")

    def capture_board(
        self,
        measured: pose.Pose2D,
        state: dict[str, Any] | None,
        received_at: float,
        *,
        force: bool = False,
    ) -> None:
        """Store a measured board state at the bounded recording rate."""
        if not force and received_at - self._last_board_at < self._interval:
            return
        self._board.append(
            (
                received_at,
                (measured.model_copy(), copy.deepcopy(state or {})),
            )
        )
        self._last_board_at = received_at

    def capture_gazebo(
        self,
        message: sensor.Image,
        *,
        received_at: float | None = None,
    ) -> None:
        """Compress one Gazebo camera image at the bounded recording rate."""
        sample_at = time.monotonic() if received_at is None else received_at
        if sample_at - self._last_gazebo_at < self._interval:
            return
        frame = _decode_image(message)
        encoded = io.BytesIO()
        pil_image.fromarray(frame).save(encoded, format="JPEG", quality=80)
        self._gazebo.append((sample_at, encoded.getvalue()))
        self._last_gazebo_at = sample_at

    def finalize(self, *, require_gazebo: bool) -> dict[str, Any]:
        """Write synchronized, fixed-duration board and Gazebo GIFs.

        Args:
            require_gazebo: Whether absence of Gazebo camera frames is fatal.

        Returns:
            Recording metadata also saved next to the GIF files.

        Raises:
            RuntimeError: No board samples, or required Gazebo frames are absent.
        """
        if not self._board:
            raise RuntimeError(
                "Cannot create GIFs without measured robot poses"
            )
        if require_gazebo and not self._gazebo:
            raise RuntimeError(
                "Cannot create gazebo.gif: no Gazebo camera frames received"
            )
        all_times = [sample[0] for sample in self._board + self._gazebo]
        targets = np.linspace(
            min(all_times), max(all_times), self._frame_count, dtype=np.float64
        )
        board_path = self._output / "board.gif"
        self._write_board(board_path, targets)
        gazebo_path: pathlib.Path | None = None
        if self._gazebo:
            gazebo_path = self._output / "gazebo.gif"
            self._write_gazebo(gazebo_path, targets)
        metadata = {
            "silent": True,
            "duration_seconds": self._duration,
            "fps": self._fps,
            "frames": self._frame_count,
            "source_duration_seconds": max(all_times) - min(all_times),
            "source_board_samples": len(self._board),
            "source_gazebo_samples": len(self._gazebo),
            "board": str(board_path),
            "gazebo": str(gazebo_path) if gazebo_path is not None else None,
        }
        with (self._output / "recording.json").open("x") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        return metadata

    @property
    def _frame_count(self) -> int:
        """Number of frames required for the requested playback duration."""
        return max(1, round(self._duration * self._fps))

    def _write_board(
        self, path: pathlib.Path, targets: np.ndarray[Any, np.dtype[np.float64]]
    ) -> None:
        """Render selected measured states through the native Zone renderer."""
        view = native_view.NativeZoneView(self._layout, render_mode="rgb_array")
        frames: list[pil_image.Image] = []
        try:
            for measured, state in _select(self._board, targets):
                view.update_pose(measured)
                if state:
                    view.update_state(state)
                rendered = view.render()
                if rendered is None:
                    raise RuntimeError(
                        "Native board renderer returned no image"
                    )
                frames.append(pil_image.fromarray(rendered).convert("RGB"))
        finally:
            view.close()
        self._save(path, frames)

    def _write_gazebo(
        self, path: pathlib.Path, targets: np.ndarray[Any, np.dtype[np.float64]]
    ) -> None:
        """Decode only the Gazebo samples selected for the final GIF."""
        frames = [
            pil_image.open(io.BytesIO(encoded)).convert("RGB")
            for encoded in _select(self._gazebo, targets)
        ]
        self._save(path, frames)

    def _save(self, path: pathlib.Path, frames: list[pil_image.Image]) -> None:
        """Write a looping GIF whose frame delays total the exact duration."""
        total_ms = round(self._duration * 1000)
        duration_by_frame = [total_ms // len(frames)] * len(frames)
        for index in range(total_ms % len(frames)):
            duration_by_frame[index] += 1
        with path.open("xb") as stream:
            frames[0].save(
                stream,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=duration_by_frame,
                loop=0,
                optimize=False,
                disposal=2,
            )


def _select[T](samples: list[tuple[float, T]], targets: np.ndarray) -> list[T]:
    """Select the source sample nearest each shared wall-clock target."""
    timestamps = [sample[0] for sample in samples]
    selected: list[T] = []
    for target in targets:
        right = bisect.bisect_left(timestamps, float(target))
        if right == 0:
            index = 0
        elif right == len(samples):
            index = len(samples) - 1
        else:
            before = timestamps[right - 1]
            after = timestamps[right]
            index = right - 1 if target - before <= after - target else right
        selected.append(samples[index][1])
    return selected


def _decode_image(message: sensor.Image) -> np.ndarray:
    """Decode the uint8 image encodings produced by ros_gz_bridge."""
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
    }
    encoding = message.encoding.lower()
    try:
        channels = channels_by_encoding[encoding]
    except KeyError as error:
        raise ValueError(
            f"Unsupported Gazebo image encoding: {encoding}"
        ) from error
    row_bytes = message.width * channels
    if message.height <= 0 or message.width <= 0 or message.step < row_bytes:
        raise ValueError("Gazebo camera image has invalid dimensions")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < message.height * message.step:
        raise ValueError("Gazebo camera image data is truncated")
    rows = raw[: message.height * message.step].reshape(
        message.height, message.step
    )
    frame = rows[:, :row_bytes].reshape(message.height, message.width, channels)
    if encoding in ("bgr8", "bgra8"):
        order = [2, 1, 0, *([3] if channels == 4 else [])]
        frame = frame[:, :, order]
    if channels == 4:
        frame = frame[:, :, :3]
    elif channels == 1:
        frame = np.repeat(frame, 3, axis=2)
    return np.ascontiguousarray(frame)
