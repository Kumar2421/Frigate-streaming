import numpy as np

from frigate.camera import PTZMetrics
from frigate.config import CameraConfig
from frigate.track.deepocsort_tracker import DeepOCSORTTracker


class NorfairTracker(DeepOCSORTTracker):
    def __init__(self, config: CameraConfig, ptz_metrics: PTZMetrics):
        super().__init__(config=config, ptz_metrics=ptz_metrics)
        self.untracked_object_boxes: list[list[int]] = []

    def update_frame_times(self, frame_name: str, frame_time: float) -> None:
        return

    def match_and_update(
        self,
        frame_name: str,
        frame_time: float,
        detections,
        frame: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        self.untracked_object_boxes = []
        return super().match_and_update(
            frame_name=frame_name,
            frame_time=frame_time,
            detections=detections,
            frame=frame,
        )
