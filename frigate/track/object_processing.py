import base64
import datetime
import json
import logging
import queue
import threading
from collections import defaultdict
from enum import Enum
from multiprocessing import Queue as MpQueue
from multiprocessing.synchronize import Event as MpEvent
from typing import Any
from frigate.track.deepocsort_tracker import DeepOCSORTTracker

import cv2
import numpy as np
import os
import uuid
from peewee import SQL, DoesNotExist

from frigate.camera.state import CameraState
from frigate.comms.detections_updater import DetectionPublisher, DetectionTypeEnum
from frigate.comms.dispatcher import Dispatcher
from frigate.comms.event_metadata_updater import (
    EventMetadataSubscriber,
    EventMetadataTypeEnum,
)
from frigate.comms.events_updater import EventEndSubscriber, EventUpdatePublisher
from frigate.comms.inter_process import InterProcessRequestor
from frigate.config import (
    CameraMqttConfig,
    FrigateConfig,
    RecordConfig,
    SnapshotsConfig,
)
try:
    from frigate.config.camera.updater import (
        CameraConfigUpdateEnum,
        CameraConfigUpdateSubscriber,
    )
except Exception:  # noqa: BLE001
    CameraConfigUpdateEnum = Enum("CameraConfigUpdateEnum", "add enabled remove zones")

    class CameraConfigUpdateSubscriber:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.camera_configs = {}

        def check_for_updates(self) -> dict[str, list[str]]:
            return {}
from frigate.const import (
    FAST_QUEUE_TIMEOUT,
    UPDATE_CAMERA_ACTIVITY,
    UPSERT_REVIEW_SEGMENT,
)
from frigate.events.types import EventStateEnum, EventTypeEnum
from frigate.models import Event, ReviewSegment, Timeline
from frigate.ptz.autotrack import PtzAutoTrackerThread
from frigate.track.tracked_object import TrackedObject
from frigate.util.image import SharedMemoryFrameManager, calculate_region
from frigate.const import BASE_DIR
from frigate.camera import PTZMetrics

logger = logging.getLogger(__name__)


class ManualEventState(str, Enum):
    complete = "complete"
    start = "start"
    end = "end"


class TrackedObjectProcessor(threading.Thread):
    def __init__(
        self,
        config: FrigateConfig,
        dispatcher: Dispatcher,
        tracked_objects_queue: MpQueue,
        ptz_autotracker_thread: PtzAutoTrackerThread,
        stop_event: MpEvent,
    ) -> None:
        super().__init__(name="detected_frames_processor")
        self.config = config
        self.dispatcher = dispatcher
        self.tracked_objects_queue = tracked_objects_queue
        self.stop_event: MpEvent = stop_event
        self.camera_states: dict[str, CameraState] = {}
        self.frame_manager = SharedMemoryFrameManager()
        self.last_motion_detected: dict[str, float] = {}
        self.ptz_autotracker_thread = ptz_autotracker_thread
        # Testing flags to bypass person gating
        self._fire_always_run = os.environ.get("FIRE_ALWAYS_RUN", "0") in ("1", "true", "True")
        self._face_always_run = os.environ.get("FACE_ALWAYS_RUN", "0") in ("1", "true", "True")
        # Optional overlay
        self._draw_clip_overlays = os.environ.get("DRAW_CLIP_OVERLAYS", "0") in ("1", "true", "True")

        self._person_output_dir = os.environ.get(
            "FIRST_PERSON_OUTPUT_DIR", os.path.join(BASE_DIR, "clips", "first-person")
        )
        try:
            os.makedirs(self._person_output_dir, exist_ok=True)
        except Exception:
            pass
        self._face_crop_output_dir = os.environ.get(
            "FIRST_FACE_OUTPUT_DIR", os.path.join(BASE_DIR, "clips", "first-face")
        )
        try:
            os.makedirs(self._face_crop_output_dir, exist_ok=True)
        except Exception:
            pass

        self._best_person_min_score = float(os.environ.get("PERSON_CROP_MIN_SCORE", "0.8"))
        self._best_face_min_score = float(os.environ.get("FACE_CROP_MIN_SCORE", "0.8"))

        # Crop filename behavior
        # - If enabled, saved crops include a timestamp suffix so they won't overwrite.
        # - For "best", this allows you to keep a history of improvements over time.
        self._crop_add_timestamp = os.environ.get("CROP_ADD_TIMESTAMP", "1") in (
            "1",
            "true",
            "True",
        )
        self._crop_save_best_history = os.environ.get("CROP_SAVE_BEST_HISTORY", "1") in (
            "1",
            "true",
            "True",
        )

        self._first_saved: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self._best_saved: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
        self._display_track_id_map = defaultdict(dict)
        self._display_track_next = defaultdict(lambda: 1)
        # Optional fire detector configuration via environment
        self._fire_model = None
        self._fire_threshold = float(os.environ.get("FIRE_THRESHOLD", "0.5"))
        fire_model_path = os.environ.get("FIRE_MODEL_PATH")
        if fire_model_path and os.path.exists(fire_model_path):
            try:
                from ultralytics import YOLO  # type: ignore
                self._fire_model = YOLO(fire_model_path)
                logger.info(f"Fire model loaded: {fire_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load fire model at {fire_model_path}: {e}")

        self._person_model = None
        self._person_threshold = float(os.environ.get("PERSON_THRESHOLD", "0.5"))
        self._person_infer_failures = 0
        try:
            self._person_infer_max_failures = int(
                os.environ.get("PERSON_MODEL_MAX_FAILURES", "3")
            )
        except Exception:
            self._person_infer_max_failures = 3
        person_model_path = os.environ.get("PERSON_MODEL_PATH")
        try:
            person_model_path = person_model_path.strip() if person_model_path else person_model_path
        except Exception:
            pass
        try:
            logger.info(
                f"Person model init: PERSON_MODEL_PATH={repr(person_model_path)}, exists={bool(person_model_path and os.path.exists(person_model_path))}"
            )
        except Exception:
            pass
        if person_model_path and os.path.exists(person_model_path):
            try:
                from ultralytics import YOLO  # type: ignore

                device: str | int = "cpu"
                use_half = False
                try:
                    import torch  # type: ignore

                    if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
                        device = 0
                        use_half = True
                except Exception:
                    device = "cpu"
                    use_half = False

                self._person_model_device = device
                self._person_model_half = use_half
                self._person_model = YOLO(person_model_path)
                logger.info(f"Person model loaded: {person_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load person model at {person_model_path}: {e}")

        # Optional face detector configuration via environment
        self._face_model = None
        self._face_threshold = float(os.environ.get("FACE_THRESHOLD", "0.5"))
        face_model_path = os.environ.get("FACE_MODEL_PATH")
        if face_model_path and os.path.exists(face_model_path):
            try:
                from ultralytics import YOLO  # type: ignore

                device: str | int = "cpu"
                use_half = False
                try:
                    import torch  # type: ignore

                    if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available():
                        device = 0
                        use_half = True
                except Exception:
                    device = "cpu"
                    use_half = False

                self._face_model_device = device
                self._face_model_half = use_half
                self._face_model = YOLO(face_model_path)
                logger.info(f"Face model loaded: {face_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load face model at {face_model_path}: {e}")

        self.camera_config_subscriber = CameraConfigUpdateSubscriber(
            self.config,
            self.config.cameras,
            [
                CameraConfigUpdateEnum.add,
                CameraConfigUpdateEnum.enabled,
                CameraConfigUpdateEnum.remove,
                CameraConfigUpdateEnum.zones,
            ],
        )

        self.requestor = InterProcessRequestor()
        self.detection_publisher = DetectionPublisher(DetectionTypeEnum.all)
        self.event_sender = EventUpdatePublisher()
        self.event_end_subscriber = EventEndSubscriber()
        self.sub_label_subscriber = EventMetadataSubscriber(EventMetadataTypeEnum.all)

        self.camera_activity: dict[str, dict[str, Any]] = {}
        self.ongoing_manual_events: dict[str, str] = {}

        # {
        #   'zone_name': {
        #       'person': {
        #           'camera_1': 2,
        #           'camera_2': 1
        #       }
        #   }
        # }
        self.zone_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.active_zone_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: defaultdict(dict)
        )

        for camera in self.config.cameras.keys():
            self.create_camera_state(camera)

        self.reid_trackers_person: dict[str, DeepOCSORTTracker] = {}
        self.reid_trackers_face: dict[str, DeepOCSORTTracker] = {}
        try:
            tracker_cfg = getattr(self.config, "tracker", None)
            reid_cfg = None
            if tracker_cfg is not None:
                reid_cfg = getattr(tracker_cfg, "deepsortrealtime", None) or getattr(
                    tracker_cfg, "deepocsort", None
                )
        except Exception:
            reid_cfg = None

        try:
            for camera_name, cam_cfg in self.config.cameras.items():
                try:
                    ptz_metrics = PTZMetrics(autotracker_enabled=False)
                    kwargs: dict[str, Any] = {"config": cam_cfg, "ptz_metrics": ptz_metrics}
                    try:
                        if reid_cfg is not None:
                            if getattr(reid_cfg, "reid_model_path", None) is not None:
                                kwargs["reid_model_path"] = getattr(reid_cfg, "reid_model_path")
                            if getattr(reid_cfg, "reid_device", None) is not None:
                                kwargs["device"] = getattr(reid_cfg, "reid_device")
                            if getattr(reid_cfg, "max_age", None) is not None:
                                kwargs["max_age"] = int(getattr(reid_cfg, "max_age"))
                            if getattr(reid_cfg, "n_init", None) is not None:
                                kwargs["n_init"] = int(getattr(reid_cfg, "n_init"))
                            if getattr(reid_cfg, "max_iou_distance", None) is not None:
                                kwargs["max_iou_distance"] = float(getattr(reid_cfg, "max_iou_distance"))
                            if getattr(reid_cfg, "max_cosine_distance", None) is not None:
                                kwargs["max_cosine_distance"] = float(getattr(reid_cfg, "max_cosine_distance"))
                            if getattr(reid_cfg, "nn_budget", None) is not None:
                                kwargs["nn_budget"] = int(getattr(reid_cfg, "nn_budget"))
                            if getattr(reid_cfg, "reid_threshold", None) is not None:
                                kwargs["reid_threshold"] = float(getattr(reid_cfg, "reid_threshold"))
                    except Exception:
                        pass

                    # Sidecar ReID trackers should confirm quickly so we can save a "first" crop.
                    # If not explicitly configured, default to n_init=1.
                    if "n_init" not in kwargs:
                        kwargs["n_init"] = 1

                    # Option B: use DeepSort track_id as the identity for crop saving.
                    # This ensures every distinct track gets its own first/best crops (no ReID merges).
                    kwargs["use_track_id_as_reid"] = True

                    self.reid_trackers_person[camera_name] = DeepOCSORTTracker(**kwargs)
                    self.reid_trackers_face[camera_name] = DeepOCSORTTracker(**kwargs)
                except Exception:
                    continue
        except Exception:
            self.reid_trackers_person = {}
            self.reid_trackers_face = {}

    def _ts_ms(self, frame_time: float) -> int:
        try:
            return int(float(frame_time) * 1000)
        except Exception:
            return 0

    def _crop_path(
        self,
        base_dir: str,
        camera: str,
        kind: str,
        reid_id: str,
        which: str,
        frame_time: float,
        include_ts: bool,
    ) -> str:
        # which: "first" or "best"
        if include_ts:
            return os.path.join(
                base_dir,
                camera,
                f"{camera}-{kind}-{reid_id}-{which}-{self._ts_ms(frame_time)}.jpg",
            )
        return os.path.join(base_dir, camera, f"{camera}-{kind}-{reid_id}-{which}.jpg")

    def _save_padded_crop_to_path(
        self,
        out_path: str,
        box: list[int],
        frame_time: float,
        frame: np.ndarray,
        kind: str | None = None,
        reid_id: str | None = None,
        score: float | None = None,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            return

        region = calculate_region(
            frame.shape,
            x1,
            y1,
            x2,
            y2,
            300,
            multiplier=1.1,
        )
        crop = frame[region[1] : region[3], region[0] : region[2]].copy()
        if crop.size == 0:
            return

        try:
            rx1, ry1, rx2, ry2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
            cx1 = max(0, min(int(x1 - rx1), crop.shape[1] - 1))
            cy1 = max(0, min(int(y1 - ry1), crop.shape[0] - 1))
            cx2 = max(0, min(int(x2 - rx1), crop.shape[1] - 1))
            cy2 = max(0, min(int(y2 - ry1), crop.shape[0] - 1))
            if cx2 > cx1 and cy2 > cy1:
                color = (0, 165, 255)
                cv2.rectangle(crop, (cx1, cy1), (cx2, cy2), color, 2)
                label = ""
                try:
                    parts: list[str] = []
                    if kind:
                        parts.append(str(kind))
                    if reid_id:
                        parts.append(str(reid_id))
                    if score is not None:
                        parts.append(f"{float(score):.2f}")
                    label = " ".join(parts)
                except Exception:
                    label = ""
                if label:
                    try:
                        (tw, th), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        pad = 3
                        y_text = max(th + pad, cy1 - 6)
                        x0 = cx1
                        y0 = max(0, y_text - th - pad)
                        x1b = min(crop.shape[1] - 1, x0 + tw + pad * 2)
                        y1b = min(crop.shape[0] - 1, y_text + baseline + pad)
                        cv2.rectangle(crop, (x0, y0), (x1b, y1b), color, -1)
                        cv2.putText(
                            crop,
                            label,
                            (x0 + pad, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
        except Exception:
            pass
        try:
            ok = cv2.imwrite(out_path, crop)
            if not ok:
                logger.warning("Failed to write crop: path=%s", out_path)
                return
        except Exception:
            logger.exception("Failed to write crop: path=%s", out_path)
            return

        try:
            which: str | None = None
            name = os.path.basename(out_path).lower()
            if "-first-" in name:
                which = "first"
            elif "-best-" in name:
                which = "best"

            camera = os.path.basename(os.path.dirname(out_path))

            payload = {
                "label": kind,
                "reid_id": reid_id,
                "score": float(score) if score is not None else None,
                "camera": camera,
                "frame_time": float(frame_time),
                "timestamp_ms": self._ts_ms(frame_time),
                "which": which,
                "file": os.path.basename(out_path),
            }

            json_path = os.path.splitext(out_path)[0] + ".json"
            tmp_path = json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_path, json_path)
        except Exception:
            pass




    def create_camera_state(self, camera: str) -> None:
        """Creates a new camera state."""

        def start(camera: str, obj: TrackedObject, frame_name: str) -> None:
            self.event_sender.publish(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.start,
                    camera,
                    frame_name,
                    obj.to_dict(),
                )
            )

        def update(camera: str, obj: TrackedObject, frame_name: str) -> None:
            obj.has_snapshot = self.should_save_snapshot(camera, obj)
            obj.has_clip = self.should_retain_recording(camera, obj)
            after = obj.to_dict()
            message = {
                "before": obj.previous,
                "after": after,
                "type": "new" if obj.previous["false_positive"] else "update",
            }
            self.dispatcher.publish("events", json.dumps(message), retain=False)
            obj.previous = after
            self.event_sender.publish(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.update,
                    camera,
                    frame_name,
                    obj.to_dict(),
                )
            )

        def autotrack(camera: str, obj: TrackedObject, frame_name: str) -> None:
            self.ptz_autotracker_thread.ptz_autotracker.autotrack_object(camera, obj)

        def end(camera: str, obj: TrackedObject, frame_name: str) -> None:
            # populate has_snapshot
            obj.has_snapshot = self.should_save_snapshot(camera, obj)
            obj.has_clip = self.should_retain_recording(camera, obj)

            # write thumbnail to disk if it will be saved as an event
            if obj.has_snapshot or obj.has_clip:
                obj.write_thumbnail_to_disk()

            # write the snapshot to disk
            if obj.has_snapshot:
                obj.write_snapshot_to_disk()

            if not obj.false_positive:
                message = {
                    "before": obj.previous,
                    "after": obj.to_dict(),
                    "type": "end",
                }
                self.dispatcher.publish("events", json.dumps(message), retain=False)
                self.ptz_autotracker_thread.ptz_autotracker.end_object(camera, obj)

            self.event_sender.publish(
                (
                    EventTypeEnum.tracked_object,
                    EventStateEnum.end,
                    camera,
                    frame_name,
                    obj.to_dict(),
                )
            )

        def snapshot(camera: str, obj: TrackedObject) -> bool:
            mqtt_config: CameraMqttConfig = self.config.cameras[camera].mqtt
            if mqtt_config.enabled and self.should_mqtt_snapshot(camera, obj):
                jpg_bytes = obj.get_img_bytes(
                    ext="jpg",
                    timestamp=mqtt_config.timestamp,
                    bounding_box=mqtt_config.bounding_box,
                    crop=mqtt_config.crop,
                    height=mqtt_config.height,
                    quality=mqtt_config.quality,
                )

                if jpg_bytes is None:
                    logger.warning(
                        f"Unable to send mqtt snapshot for {obj.obj_data['id']}."
                    )
                else:
                    self.dispatcher.publish(
                        f"{camera}/{obj.obj_data['label']}/snapshot",
                        jpg_bytes,
                        retain=True,
                    )

                    if obj.obj_data.get("sub_label"):
                        sub_label = obj.obj_data["sub_label"][0]

                        if sub_label in self.config.model.all_attribute_logos:
                            self.dispatcher.publish(
                                f"{camera}/{sub_label}/snapshot",
                                jpg_bytes,
                                retain=True,
                            )

                    return True

            return False

        def camera_activity(camera: str, activity: dict[str, Any]) -> None:
            last_activity = self.camera_activity.get(camera)

            if not last_activity or activity != last_activity:
                self.camera_activity[camera] = activity
                self.requestor.send_data(UPDATE_CAMERA_ACTIVITY, self.camera_activity)

        camera_state = CameraState(
            camera, self.config, self.frame_manager, self.ptz_autotracker_thread
        )
        camera_state.on("start", start)
        camera_state.on("autotrack", autotrack)
        camera_state.on("update", update)
        camera_state.on("end", end)
        camera_state.on("snapshot", snapshot)
        camera_state.on("camera_activity", camera_activity)
        self.camera_states[camera] = camera_state

    def should_save_snapshot(self, camera: str, obj: TrackedObject) -> bool:
        if obj.false_positive:
            return False

        snapshot_config: SnapshotsConfig = self.config.cameras[camera].snapshots

        if not snapshot_config.enabled:
            return False

        # object never changed position
        if obj.obj_data["position_changes"] == 0:
            return False

        # if there are required zones and there is no overlap
        required_zones = snapshot_config.required_zones
        if len(required_zones) > 0 and not set(obj.entered_zones) & set(required_zones):
            # Object did not enter required zones - skip snapshot
            return False

        return True

    def should_retain_recording(self, camera: str, obj: TrackedObject) -> bool:
        if obj.false_positive:
            return False

        record_config: RecordConfig = self.config.cameras[camera].record

        # Recording is disabled
        if not record_config.enabled:
            return False

        # object never changed position
        if obj.obj_data["position_changes"] == 0:
            return False

        # If the object is not considered an alert or detection
        if obj.max_severity is None:
            return False

        return True

    def should_mqtt_snapshot(self, camera: str, obj: TrackedObject) -> bool:
        # object never changed position
        if obj.is_stationary():
            return False

        # if there are required zones and there is no overlap
        required_zones = self.config.cameras[camera].mqtt.required_zones
        if len(required_zones) > 0 and not set(obj.entered_zones) & set(required_zones):
            # Object did not enter required zones - skip MQTT
            return False

        return True

    def update_mqtt_motion(
        self, camera: str, frame_time: float, motion_boxes: list
    ) -> None:
        # publish if motion is currently being detected
        if motion_boxes:
            # only send ON if motion isn't already active
            if self.last_motion_detected.get(camera, 0) == 0:
                self.dispatcher.publish(
                    f"{camera}/motion",
                    "ON",
                    retain=False,
                )

            # always updated latest motion
            self.last_motion_detected[camera] = frame_time
        elif self.last_motion_detected.get(camera, 0) > 0:
            mqtt_delay = self.config.cameras[camera].motion.mqtt_off_delay

            # If no motion, make sure the off_delay has passed
            if frame_time - self.last_motion_detected.get(camera, 0) >= mqtt_delay:
                self.dispatcher.publish(
                    f"{camera}/motion",
                    "OFF",
                    retain=False,
                )
                # reset the last_motion so redundant `off` commands aren't sent
                self.last_motion_detected[camera] = 0

    def get_best(self, camera: str, label: str) -> dict[str, Any]:
        # TODO: need a lock here
        camera_state = self.camera_states[camera]
        if label in camera_state.best_objects:
            best_obj = camera_state.best_objects[label]

            if not best_obj.thumbnail_data:
                return {}

            best = best_obj.thumbnail_data.copy()
            best["frame"] = camera_state.frame_cache.get(
                best_obj.thumbnail_data["frame_time"]
            )
            return best
        else:
            return {}

    def get_best_by_id(self, camera: str, object_id: str) -> dict[str, Any]:
        """Returns the best thumbnail data for a given object id."""
        try:
            return self.camera_states[camera].best_objects[object_id].thumbnail_data
        except Exception:
            return {}

    def get_current_frame(
        self, camera: str, draw_options: dict[str, Any] | None = None
    ) -> np.ndarray | None:
        """Returns the latest frame for a given camera."""
        if draw_options is None:
            draw_options = {}
        try:
            return self.camera_states[camera].get_current_frame(draw_options)
        except Exception:
            return None

    def get_current_frame_time(self, camera: str) -> float:
        """Returns the latest frame time for a given camera."""
        return self.camera_states[camera].current_frame_time

    def set_sub_label(
        self, event_id: str, sub_label: str | None, score: float | None
    ) -> None:
        """Update sub label for given event id."""
        tracked_obj: TrackedObject | None = None

        for state in self.camera_states.values():
            tracked_obj = state.tracked_objects.get(event_id)

            if tracked_obj is not None:
                break

        try:
            event: Event | None = Event.get(Event.id == event_id)
        except DoesNotExist:
            event = None

        if not tracked_obj and not event:
            return

        if tracked_obj:
            tracked_obj.obj_data["sub_label"] = (sub_label, score)

        if event:
            event.sub_label = sub_label  # type: ignore[assignment]
            data = event.data
            if sub_label is None:
                data["sub_label_score"] = None  # type: ignore[index]
            elif score is not None:
                data["sub_label_score"] = score  # type: ignore[index]
            event.data = data
            event.save()

            # update timeline items
            Timeline.update(
                data=Timeline.data.update({"sub_label": (sub_label, score)})
            ).where(Timeline.source_id == event_id).execute()

            # only update ended review segments
            # manually updating a sub_label from the UI is only possible for ended tracked objects
            try:
                review_segment = ReviewSegment.get(
                    (
                        SQL(
                            "json_extract(data, '$.detections') LIKE ?",
                            [f'%"{event_id}"%'],
                        )
                    )
                    & (ReviewSegment.end_time.is_null(False))
                )

                segment_data = review_segment.data
                detection_ids = segment_data.get("detections", [])

                # Rebuild objects list and sync sub_labels
                objects_list = []
                sub_labels = set()
                events = Event.select(Event.id, Event.label, Event.sub_label).where(
                    Event.id.in_(detection_ids)  # type: ignore[call-arg, misc]
                )
                for det_event in events:
                    if det_event.sub_label:
                        sub_labels.add(det_event.sub_label)
                        objects_list.append(
                            f"{det_event.label}-verified"
                        )  # eg, "bird-verified"
                    else:
                        objects_list.append(det_event.label)  # eg, "bird"

                segment_data["sub_labels"] = list(sub_labels)
                segment_data["objects"] = objects_list

                updated_data = {
                    ReviewSegment.id.name: review_segment.id,
                    ReviewSegment.camera.name: review_segment.camera,
                    ReviewSegment.start_time.name: review_segment.start_time,
                    ReviewSegment.end_time.name: review_segment.end_time,
                    ReviewSegment.severity.name: review_segment.severity,
                    ReviewSegment.thumb_path.name: review_segment.thumb_path,
                    ReviewSegment.data.name: segment_data,
                }

                self.requestor.send_data(UPSERT_REVIEW_SEGMENT, updated_data)
                # Sub-label updated in review segment

            except DoesNotExist:
                # No review segment found for event
                pass

    def set_object_attribute(
        self,
        event_id: str,
        field_name: str,
        field_value: str | None,
        score: float | None,
    ) -> None:
        """Update attribute for given event id."""
        tracked_obj: TrackedObject | None = None

        for state in self.camera_states.values():
            tracked_obj = state.tracked_objects.get(event_id)

            if tracked_obj is not None:
                break

        try:
            event: Event | None = Event.get(Event.id == event_id)
        except DoesNotExist:
            event = None

        if not tracked_obj and not event:
            return

        if tracked_obj:
            tracked_obj.obj_data[field_name] = (
                field_value,
                score,
            )

        if event:
            data = event.data
            data[field_name] = field_value  # type: ignore[index]
            if field_value is None:
                data[f"{field_name}_score"] = None  # type: ignore[index]
            elif score is not None:
                data[f"{field_name}_score"] = score  # type: ignore[index]
            event.data = data
            event.save()

    def save_lpr_snapshot(self, payload: tuple) -> None:
        # save the snapshot image
        (frame, event_id, camera) = payload

        img = cv2.imdecode(
            np.frombuffer(base64.b64decode(frame), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        self.camera_states[camera].save_manual_event_image(
            img, event_id, "license_plate", {}
        )

    def create_manual_event(self, payload: tuple) -> None:
        (
            frame_time,
            camera_name,
            label,
            event_id,
            include_recording,
            score,
            sub_label,
            duration,
            source_type,
            draw,
        ) = payload

        # save the snapshot image
        self.camera_states[camera_name].save_manual_event_image(
            None, event_id, label, draw
        )
        end_time = frame_time + duration if duration is not None else None

        # send event to event maintainer
        self.event_sender.publish(
            (
                EventTypeEnum.api,
                EventStateEnum.start,
                camera_name,
                "",
                {
                    "id": event_id,
                    "label": label,
                    "sub_label": sub_label,
                    "score": score,
                    "camera": camera_name,
                    "start_time": frame_time
                    - self.config.cameras[camera_name].record.event_pre_capture,
                    "end_time": end_time,
                    "has_clip": self.config.cameras[camera_name].record.enabled
                    and include_recording,
                    "has_snapshot": True,
                    "type": source_type,
                },
            )
        )

        if source_type == "api":
            self.ongoing_manual_events[event_id] = camera_name
            self.detection_publisher.publish(
                (
                    camera_name,
                    frame_time,
                    {
                        "state": (
                            ManualEventState.complete
                            if end_time
                            else ManualEventState.start
                        ),
                        "label": f"{label}: {sub_label}" if sub_label else label,
                        "event_id": event_id,
                        "end_time": end_time,
                    },
                ),
                DetectionTypeEnum.api.value,
            )

    def create_lpr_event(self, payload: tuple) -> None:
        (
            frame_time,
            camera_name,
            label,
            event_id,
            include_recording,
            score,
            sub_label,
            plate,
        ) = payload

        # send event to event maintainer
        self.event_sender.publish(
            (
                EventTypeEnum.api,
                EventStateEnum.start,
                camera_name,
                "",
                {
                    "id": event_id,
                    "label": label,
                    "sub_label": sub_label,
                    "score": score,
                    "camera": camera_name,
                    "start_time": frame_time
                    - self.config.cameras[camera_name].record.event_pre_capture,
                    "end_time": None,
                    "has_clip": self.config.cameras[camera_name].record.enabled
                    and include_recording,
                    "has_snapshot": True,
                    "type": "api",
                    "recognized_license_plate": plate,
                    "recognized_license_plate_score": score,
                },
            )
        )

        self.ongoing_manual_events[event_id] = camera_name
        self.detection_publisher.publish(
            (
                camera_name,
                frame_time,
                {
                    "state": ManualEventState.start,
                    "label": f"{label}: {sub_label}" if sub_label else label,
                    "event_id": event_id,
                    "end_time": None,
                },
            ),
            DetectionTypeEnum.lpr.value,
        )

    def end_manual_event(self, payload: tuple) -> None:
        (event_id, end_time) = payload

        self.event_sender.publish(
            (
                EventTypeEnum.api,
                EventStateEnum.end,
                None,
                "",
                {"id": event_id, "end_time": end_time},
            )
        )

        if event_id in self.ongoing_manual_events:
            self.detection_publisher.publish(
                (
                    self.ongoing_manual_events[event_id],
                    end_time,
                    {
                        "state": ManualEventState.end,
                        "event_id": event_id,
                        "end_time": end_time,
                    },
                ),
                DetectionTypeEnum.api.value,
            )
            self.ongoing_manual_events.pop(event_id)

    def force_end_all_events(self, camera: str, camera_state: CameraState) -> None:
        """Ends all active events on camera when disabling."""
        last_frame_name = camera_state.previous_frame_id
        for obj_id, obj in list(camera_state.tracked_objects.items()):
            if "end_time" not in obj.obj_data:
                # Camera disabled - ending active event
                obj.obj_data["end_time"] = datetime.datetime.now().timestamp()
                # end callbacks
                for callback in camera_state.callbacks["end"]:
                    callback(camera, obj, last_frame_name)

                # camera activity callbacks
                for callback in camera_state.callbacks["camera_activity"]:
                    callback(
                        camera,
                        {"enabled": False, "motion": 0, "objects": []},
                    )

    def run(self) -> None:
        while not self.stop_event.is_set():
            # check for config updates
            updated_topics = self.camera_config_subscriber.check_for_updates()

            if "enabled" in updated_topics:
                for camera in updated_topics["enabled"]:
                    if self.camera_states[camera].prev_enabled is None:
                        self.camera_states[camera].prev_enabled = self.config.cameras[
                            camera
                        ].enabled
            elif "add" in updated_topics:
                for camera in updated_topics["add"]:
                    self.config.cameras[camera] = (
                        self.camera_config_subscriber.camera_configs[camera]
                    )
                    self.create_camera_state(camera)
            elif "remove" in updated_topics:
                for camera in updated_topics["remove"]:
                    camera_state = self.camera_states[camera]
                    camera_state.shutdown()
                    self.camera_states.pop(camera)

            # manage camera disabled state
            for camera, config in self.config.cameras.items():
                if not config.enabled_in_config:
                    continue

                current_enabled = config.enabled
                camera_state = self.camera_states[camera]

                if camera_state.prev_enabled and not current_enabled:
                    # Camera disabled - skipping object processing
                    self.force_end_all_events(camera, camera_state)

                camera_state.prev_enabled = current_enabled

                if not current_enabled:
                    continue

            # check for sub label updates
            while True:
                update = self.sub_label_subscriber.check_for_update(timeout=0)

                if not update:
                    break

                (raw_topic, payload) = update

                if not raw_topic or not payload:
                    break

                topic = str(raw_topic)

                if topic.endswith(EventMetadataTypeEnum.sub_label.value):
                    (event_id, sub_label, score) = payload
                    self.set_sub_label(event_id, sub_label, score)
                if topic.endswith(EventMetadataTypeEnum.attribute.value):
                    (event_id, field_name, field_value, score) = payload
                    self.set_object_attribute(event_id, field_name, field_value, score)
                elif topic.endswith(EventMetadataTypeEnum.lpr_event_create.value):
                    self.create_lpr_event(payload)
                elif topic.endswith(EventMetadataTypeEnum.save_lpr_snapshot.value):
                    self.save_lpr_snapshot(payload)
                elif topic.endswith(EventMetadataTypeEnum.manual_event_create.value):
                    self.create_manual_event(payload)
                elif topic.endswith(EventMetadataTypeEnum.manual_event_end.value):
                    self.end_manual_event(payload)

            try:
                (
                    camera,
                    frame_name,
                    frame_time,
                    current_tracked_objects,
                    motion_boxes,
                    regions,
                ) = self.tracked_objects_queue.get(True, 1)
            except queue.Empty:
                continue

            try:
                debug_enabled = os.environ.get("OBJECT_PROCESSING_DEBUG", "0") in (
                    "1",
                    "true",
                    "True",
                )
            except Exception:
                debug_enabled = False

            if debug_enabled:
                try:
                    every_n = int(os.environ.get("OBJECT_PROCESSING_DEBUG_EVERY_N", "30"))
                except Exception:
                    every_n = 30
                try:
                    if not hasattr(self, "_objproc_debug_counter"):
                        self._objproc_debug_counter = 0
                    self._objproc_debug_counter += 1
                    if every_n <= 1 or (self._objproc_debug_counter % every_n) == 0:
                        det_count = -1
                        labels: dict[str, int] = {}
                        try:
                            if isinstance(current_tracked_objects, dict):
                                det_count = len(current_tracked_objects)
                                for o in current_tracked_objects.values():
                                    if isinstance(o, dict):
                                        l = str(o.get("label") or o.get("name") or "unknown")
                                    else:
                                        l = str(getattr(o, "label", "unknown"))
                                    labels[l] = labels.get(l, 0) + 1
                            elif isinstance(current_tracked_objects, list):
                                det_count = len(current_tracked_objects)
                                for o in current_tracked_objects:
                                    if isinstance(o, dict):
                                        l = str(o.get("label") or o.get("name") or "unknown")
                                    elif isinstance(o, (tuple, list)) and len(o) > 0:
                                        l = str(o[0])
                                    else:
                                        l = str(getattr(o, "label", "unknown"))
                                    labels[l] = labels.get(l, 0) + 1
                        except Exception:
                            det_count = -1
                            labels = {}

                        logger.info(
                            "ObjectProcessing debug: camera=%s frame=%.6f detections=%s labels=%s motion_boxes=%s regions=%s",
                            camera,
                            float(frame_time),
                            det_count,
                            labels,
                            len(motion_boxes) if isinstance(motion_boxes, list) else -1,
                            len(regions) if isinstance(regions, list) else -1,
                        )
                except Exception:
                    pass

            if not self.config.cameras[camera].enabled:
                # Camera disabled - skipping update
                continue

            camera_state = self.camera_states[camera]

            camera_state.update(
                frame_name, frame_time, current_tracked_objects, motion_boxes, regions
            )

            self.update_mqtt_motion(camera, frame_time, motion_boxes)

            frame_yuv = None
            frame_bgr = None
            try:
                cam_cfg = self.config.cameras[camera]
                frame_yuv = self.frame_manager.get(
                    frame_name,
                    cam_cfg.frame_shape_yuv,
                )
            except Exception:
                frame_yuv = None

            if debug_enabled and frame_yuv is None:
                try:
                    logger.warning(
                        "Raw frame fetch returned None: camera=%s frame=%.6f frame_name=%s expected_shape=%s",
                        camera,
                        float(frame_time),
                        str(frame_name),
                        str(getattr(self.config.cameras[camera], "frame_shape_yuv", None)),
                    )
                except Exception:
                    pass

            if frame_yuv is not None:
                try:
                    frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_I420)
                except Exception:
                    frame_bgr = None

            if debug_enabled and frame_yuv is not None and frame_bgr is None:
                try:
                    logger.warning(
                        "Raw frame convert failed (YUV->BGR): camera=%s frame=%.6f frame_name=%s yuv_shape=%s",
                        camera,
                        float(frame_time),
                        str(frame_name),
                        str(getattr(frame_yuv, "shape", None)),
                    )
                except Exception:
                    pass

            person_dets: list[dict[str, Any]] = []
            face_dets: list[dict[str, Any]] = []

            if frame_bgr is not None:
                if self._person_model is not None:
                    try:
                        person_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        # Use .predict for consistency with face model and to avoid
                        # accidental differences in inference defaults.
                        person_device = getattr(self, "_person_model_device", "cpu")
                        person_half = bool(getattr(self, "_person_model_half", False))
                        results = self._person_model.predict(
                            person_rgb,
                            verbose=False,
                            save=False,
                            device=person_device,
                            half=person_half,
                        )
                        for r in results:
                            names = getattr(r, "names", {}) or {}
                            boxes = getattr(r, "boxes", None)
                            if boxes is None:
                                continue
                            for b in boxes:
                                try:
                                    conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                                    if conf < float(self._person_threshold):
                                        continue
                                    cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                                    label = names.get(cls_id, str(cls_id))
                                    try:
                                        label_str = str(label)
                                    except Exception:
                                        label_str = ""
                                    if cls_id != 0 and "person" not in label_str.lower():
                                        continue
                                    arr = b.xyxy[0].tolist() if hasattr(b, "xyxy") else None
                                    if not arr or len(arr) < 4:
                                        continue
                                    x1, y1, x2, y2 = [int(v) for v in arr[:4]]
                                    person_dets.append(
                                        {"label": "person", "score": conf, "box": [x1, y1, x2, y2]}
                                    )
                                except Exception:
                                    continue
                    except Exception:
                        if debug_enabled:
                            logger.exception(
                                "Person model inference failed: camera=%s frame=%.6f",
                                camera,
                                float(frame_time),
                            )
                        self._person_infer_failures += 1
                        if (
                            self._person_infer_max_failures > 0
                            and self._person_infer_failures >= self._person_infer_max_failures
                        ):
                            logger.error(
                                "Disabling person model after %s failures (likely incompatible model/Ultralytics).",
                                self._person_infer_failures,
                            )
                            self._person_model = None
                        person_dets = []
                elif debug_enabled:
                    logger.info(
                        "Person model not loaded; skipping person detections: camera=%s",
                        camera,
                    )

                if self._face_model is not None:
                    try:
                        face_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        face_device = getattr(self, "_face_model_device", "cpu")
                        face_half = bool(getattr(self, "_face_model_half", False))
                        results = self._face_model.predict(
                            face_rgb,
                            verbose=False,
                            save=False,
                            device=face_device,
                            half=face_half,
                        )
                        for r in results:
                            names = getattr(r, "names", {}) or {}
                            boxes = getattr(r, "boxes", None)
                            if boxes is None:
                                continue
                            for b in boxes:
                                try:
                                    conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                                    if conf < float(self._face_threshold):
                                        continue
                                    cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                                    label = names.get(cls_id, "face")
                                    arr = b.xyxy[0].tolist() if hasattr(b, "xyxy") else None
                                    if not arr or len(arr) < 4:
                                        continue
                                    x1, y1, x2, y2 = [int(v) for v in arr[:4]]
                                    face_dets.append(
                                        {"label": "face", "score": conf, "box": [x1, y1, x2, y2]}
                                    )
                                except Exception:
                                    continue
                    except Exception:
                        face_dets = []

            if debug_enabled and frame_bgr is not None and self._person_model is not None:
                if len(person_dets) == 0:
                    try:
                        logger.info(
                            "Person dets are 0 after filtering: camera=%s frame=%.6f threshold=%.3f",
                            camera,
                            float(frame_time),
                            float(self._person_threshold),
                        )
                    except Exception:
                        pass

            person_tracker = self.reid_trackers_person.get(camera)
            face_tracker = self.reid_trackers_face.get(camera)

            if person_tracker is not None:
                try:
                    person_tracker.match_and_update(
                        frame_name=frame_name,
                        frame_time=frame_time,
                        detections=person_dets,
                        frame=frame_yuv,
                    )
                except Exception:
                    if debug_enabled:
                        logger.exception(
                            "ReID person match_and_update failed: camera=%s frame=%.6f dets=%s",
                            camera,
                            float(frame_time),
                            len(person_dets),
                        )

            if face_tracker is not None:
                try:
                    face_tracker.match_and_update(
                        frame_name=frame_name,
                        frame_time=frame_time,
                        detections=face_dets,
                        frame=frame_yuv,
                    )
                except Exception:
                    if debug_enabled:
                        logger.exception(
                            "ReID face match_and_update failed: camera=%s frame=%.6f dets=%s",
                            camera,
                            float(frame_time),
                            len(face_dets),
                        )

            if debug_enabled:
                try:
                    p_tracks = (
                        len(person_tracker.get_tracked_objects())
                        if person_tracker is not None
                        else 0
                    )
                    f_tracks = (
                        len(face_tracker.get_tracked_objects())
                        if face_tracker is not None
                        else 0
                    )
                    logger.info(
                        "ReID debug: camera=%s frame=%.6f person_dets=%s face_dets=%s person_tracks=%s face_tracks=%s out_person=%s out_face=%s",
                        camera,
                        float(frame_time),
                        len(person_dets),
                        len(face_dets),
                        p_tracks,
                        f_tracks,
                        self._person_output_dir,
                        self._face_crop_output_dir,
                    )
                except Exception:
                    pass

            if frame_bgr is not None:
                try:
                    if person_tracker is not None:
                        for obj in person_tracker.get_tracked_objects().values():
                            if not isinstance(obj, dict):
                                continue
                            rid = obj.get("reid_id")
                            box = obj.get("box")
                            if not rid or not isinstance(box, list) or len(box) != 4:
                                continue
                            try:
                                score = float(obj.get("score") or 0.0)
                            except Exception:
                                score = 0.0
                            if score < float(self._best_person_min_score):
                                continue

                            key = ("person", str(rid))
                            if key not in self._first_saved[camera]:
                                out_path = self._crop_path(
                                    self._person_output_dir,
                                    camera,
                                    "person",
                                    str(rid),
                                    "first",
                                    frame_time,
                                    self._crop_add_timestamp,
                                )
                                self._save_padded_crop_to_path(
                                    out_path=out_path,
                                    box=[int(v) for v in box],
                                    frame_time=frame_time,
                                    frame=frame_bgr,
                                    kind="person",
                                    reid_id=str(rid),
                                    score=score,
                                )
                                self._first_saved[camera].add(key)
                                if debug_enabled:
                                    logger.info(
                                        "Saved FIRST person crop: camera=%s reid_id=%s score=%.3f path=%s",
                                        camera,
                                        str(rid),
                                        float(score),
                                        out_path,
                                    )

                            prev_best = float(self._best_saved[camera].get(key, 0.0) or 0.0)
                            if score > prev_best:
                                out_path = self._crop_path(
                                    self._person_output_dir,
                                    camera,
                                    "person",
                                    str(rid),
                                    "best",
                                    frame_time,
                                    self._crop_add_timestamp and self._crop_save_best_history,
                                )
                                self._save_padded_crop_to_path(
                                    out_path=out_path,
                                    box=[int(v) for v in box],
                                    frame_time=frame_time,
                                    frame=frame_bgr,
                                    kind="person",
                                    reid_id=str(rid),
                                    score=score,
                                )
                                self._best_saved[camera][key] = float(score)
                                if debug_enabled:
                                    logger.info(
                                        "Saved BEST person crop: camera=%s reid_id=%s score=%.3f path=%s",
                                        camera,
                                        str(rid),
                                        float(score),
                                        out_path,
                                    )
                except Exception:
                    if debug_enabled:
                        logger.exception(
                            "Person crop saving failed: camera=%s frame=%.6f",
                            camera,
                            float(frame_time),
                        )

                try:
                    if face_tracker is not None:
                        for obj in face_tracker.get_tracked_objects().values():
                            if not isinstance(obj, dict):
                                continue
                            rid = obj.get("reid_id")
                            box = obj.get("box")
                            if not rid or not isinstance(box, list) or len(box) != 4:
                                continue
                            try:
                                score = float(obj.get("score") or 0.0)
                            except Exception:
                                score = 0.0
                            if score < float(self._best_face_min_score):
                                if debug_enabled and key not in self._first_saved[camera]:
                                    logger.info(
                                        "Skip face crop (score<threshold): camera=%s reid_id=%s score=%.3f threshold=%.3f",
                                        camera,
                                        str(rid),
                                        float(score),
                                        float(self._best_face_min_score),
                                    )
                                continue

                            key = ("face", str(rid))
                            if key not in self._first_saved[camera]:
                                out_path = self._crop_path(
                                    self._face_crop_output_dir,
                                    camera,
                                    "face",
                                    str(rid),
                                    "first",
                                    frame_time,
                                    self._crop_add_timestamp,
                                )
                                self._save_padded_crop_to_path(
                                    out_path=out_path,
                                    box=[int(v) for v in box],
                                    frame_time=frame_time,
                                    frame=frame_bgr,
                                    kind="face",
                                    reid_id=str(rid),
                                    score=score,
                                )
                                self._first_saved[camera].add(key)
                                if debug_enabled:
                                    logger.info(
                                        "Saved FIRST face crop: camera=%s reid_id=%s score=%.3f path=%s",
                                        camera,
                                        str(rid),
                                        float(score),
                                        out_path,
                                    )

                            prev_best = float(self._best_saved[camera].get(key, 0.0) or 0.0)
                            if score > prev_best:
                                out_path = self._crop_path(
                                    self._face_crop_output_dir,
                                    camera,
                                    "face",
                                    str(rid),
                                    "best",
                                    frame_time,
                                    self._crop_add_timestamp and self._crop_save_best_history,
                                )
                                self._save_padded_crop_to_path(
                                    out_path=out_path,
                                    box=[int(v) for v in box],
                                    frame_time=frame_time,
                                    frame=frame_bgr,
                                    kind="face",
                                    reid_id=str(rid),
                                    score=score,
                                )
                                self._best_saved[camera][key] = float(score)
                                if debug_enabled:
                                    logger.info(
                                        "Saved BEST face crop: camera=%s reid_id=%s score=%.3f path=%s",
                                        camera,
                                        str(rid),
                                        float(score),
                                        out_path,
                                    )
                except Exception:
                    if debug_enabled:
                        logger.exception(
                            "Face crop saving failed: camera=%s frame=%.6f",
                            camera,
                            float(frame_time),
                        )

            # Keep core Frigate tracked objects pipeline unchanged.

            tracked_objects = [
                o.to_dict() for o in camera_state.tracked_objects.values()
            ]

            if self._draw_clip_overlays:
                if frame_bgr is not None and tracked_objects is not None:
                    try:
                        annotated = frame_bgr.copy()
                        try:
                            import time as _time
                            sec = _time.strftime('%S', _time.localtime(frame_time))
                        except Exception:
                            sec = f"{int(frame_time)%60:02d}"
                        overlay_objects = tracked_objects
                        for obj in overlay_objects:
                            raw_tid = obj.get("id") or obj.get("track_id") or obj.get("tracking_id")
                            tid = None
                            if raw_tid is not None:
                                try:
                                    tid_key = str(raw_tid)
                                    cam_map = self._display_track_id_map[camera]
                                    if tid_key not in cam_map:
                                        cam_map[tid_key] = self._display_track_next[camera]
                                        self._display_track_next[camera] += 1
                                    tid = cam_map[tid_key]
                                except Exception:
                                    tid = None
                            # Try multiple bbox formats
                            x1 = y1 = x2 = y2 = None
                            if isinstance(obj.get("box"), (list, tuple)) and len(obj["box"]) == 4:
                                x1, y1, x2, y2 = obj["box"]
                            elif isinstance(obj.get("bbox"), dict):
                                bx = obj["bbox"]
                                if all(k in bx for k in ("x", "y", "w", "h")):
                                    x1, y1 = int(bx["x"]), int(bx["y"])
                                    x2, y2 = x1 + int(bx["w"]), y1 + int(bx["h"])
                            elif all(k in obj for k in ("x1", "y1", "x2", "y2")):
                                x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]

                            if None not in (x1, y1, x2, y2):
                                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                                label = obj.get("label")
                                if label is None and isinstance(obj.get("obj_data"), dict):
                                    label = obj["obj_data"].get("label")
                                if label is None:
                                    label = "person"
                                score = obj.get("score")
                                if score is None:
                                    score = obj.get("confidence")
                                if score is None:
                                    score = obj.get("conf")
                                if score is None and isinstance(obj.get("obj_data"), dict):
                                    score = obj["obj_data"].get("score")
                                conf_txt = None
                                try:
                                    if score is not None:
                                        conf_txt = f"{int(float(score) * 100)}"
                                except Exception:
                                    conf_txt = None
                                parts = []
                                if conf_txt:
                                    parts.append(conf_txt)
                                parts.append(f"{sec}s")
                                if tid is not None:
                                    parts.append(str(tid))
                                text = f"{label}: " + ", ".join(parts)

                                bbox_color = (0, 120, 0)
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 2)

                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.6
                                thickness = 2
                                pad = 3

                                (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                                ih, iw = annotated.shape[:2]
                                rect_left = max(0, x1)
                                rect_right = min(iw - 1, x1 + tw + pad * 2)

                                if y1 - (th + baseline + pad * 2) >= 0:
                                    rect_top = y1 - (th + baseline + pad * 2)
                                    rect_bottom = y1
                                    text_y = y1 - pad - baseline
                                else:
                                    rect_top = y1
                                    rect_bottom = min(ih - 1, y1 + th + baseline + pad * 2)
                                    text_y = rect_top + th + pad

                                cv2.rectangle(
                                    annotated,
                                    (rect_left, rect_top),
                                    (rect_right, rect_bottom),
                                    bbox_color,
                                    -1,
                                )
                                cv2.putText(
                                    annotated,
                                    text,
                                    (rect_left + pad, text_y),
                                    font,
                                    font_scale,
                                    (0, 0, 0),
                                    thickness,
                                    cv2.LINE_AA,
                                )
                    except Exception:
                        pass


            # publish info on this frame
            self.detection_publisher.publish(
                (
                    camera,
                    frame_name,
                    frame_time,
                    tracked_objects,
                    motion_boxes,
                    regions,
                ),
                DetectionTypeEnum.video.value,
            )

            # cleanup event finished queue
            while not self.stop_event.is_set():
                update = self.event_end_subscriber.check_for_update(
                    timeout=FAST_QUEUE_TIMEOUT
                )

                if not update:
                    break

                event_id, camera, _ = update
                self.camera_states[camera].finished(event_id)

        # shut down camera states
        for state in self.camera_states.values():
            state.shutdown()

        if hasattr(self.requestor, "stop") and callable(getattr(self.requestor, "stop")):
            self.requestor.stop()
        if hasattr(self.detection_publisher, "stop") and callable(
            getattr(self.detection_publisher, "stop")
        ):
            self.detection_publisher.stop()
        if hasattr(self.event_sender, "stop") and callable(getattr(self.event_sender, "stop")):
            self.event_sender.stop()
        if hasattr(self.event_end_subscriber, "stop") and callable(
            getattr(self.event_end_subscriber, "stop")
        ):
            self.event_end_subscriber.stop()
        if hasattr(self.sub_label_subscriber, "stop") and callable(
            getattr(self.sub_label_subscriber, "stop")
        ):
            self.sub_label_subscriber.stop()
        if hasattr(self.camera_config_subscriber, "stop") and callable(
            getattr(self.camera_config_subscriber, "stop")
        ):
            self.camera_config_subscriber.stop()

        logger.info("Exiting object processor...")
