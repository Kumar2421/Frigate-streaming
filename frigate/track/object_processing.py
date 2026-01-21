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
from frigate.util.image import SharedMemoryFrameManager

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
        # Directory to save YOLO annotated frames (optional)
        self._yolo_save_dir = os.environ.get("YOLO_SAVE_DIR", "/media/frigate/yolo")
        try:
            os.makedirs(self._yolo_save_dir, exist_ok=True)
        except Exception:
            # Fallback to /tmp if media is not writable
            self._yolo_save_dir = "/tmp/Ultralytics"
            os.makedirs(self._yolo_save_dir, exist_ok=True)
        # Testing flags to bypass person gating
        self._fire_always_run = os.environ.get("FIRE_ALWAYS_RUN", "0") in ("1", "true", "True")
        self._face_always_run = os.environ.get("FACE_ALWAYS_RUN", "0") in ("1", "true", "True")
        # Optional overlay for clips
        self._draw_clip_overlays = os.environ.get("DRAW_CLIP_OVERLAYS", "0") in ("1", "true", "True")
        self._clips_output_dir = os.environ.get("CLIPS_OUTPUT_DIR", "/media/frigate/clips")
        try:
            os.makedirs(self._clips_output_dir, exist_ok=True)
        except Exception:
            pass
        self._face_output_dir = os.environ.get(
            "FACE_OUTPUT_DIR", os.path.join(self._clips_output_dir, "faces")
        )
        try:
            os.makedirs(self._face_output_dir, exist_ok=True)
        except Exception:
            pass
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

        # Optional face detector configuration via environment
        self._face_model = None
        self._face_threshold = float(os.environ.get("FACE_THRESHOLD", "0.5"))
        face_model_path = os.environ.get("FACE_MODEL_PATH")
        if face_model_path and os.path.exists(face_model_path):
            try:
                from ultralytics import YOLO  # type: ignore
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

        # Initialize DeepOCSORT tracker safely (only if enabled and available)
        self.deep_tracker = None
        try:
            track_cfg = getattr(self.config, "track", None)
            if getattr(track_cfg, "type", None) == "deepocsort":
                detect_cfg = getattr(self.config, "detect", self.config)
                self.deep_tracker = DeepOCSORTTracker(detect_cfg)
        except Exception as e:
            logger.debug(f"Deep tracker disabled: {e}")

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

    def get_current_frame(
        self, camera: str, draw_options: dict[str, Any] = {}
    ) -> np.ndarray | None:
        if camera == "birdseye":
            return self.frame_manager.get(
                "birdseye",
                (self.config.birdseye.height * 3 // 2, self.config.birdseye.width),
            )

        if camera not in self.camera_states:
            return None

        return self.camera_states[camera].get_current_frame(draw_options)

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

            if not self.config.cameras[camera].enabled:
                # Camera disabled - skipping update
                continue

            camera_state = self.camera_states[camera]

            camera_state.update(
                frame_name, frame_time, current_tracked_objects, motion_boxes, regions
            )

            self.update_mqtt_motion(camera, frame_time, motion_boxes)

            frame = None
            if self._draw_clip_overlays or self._face_model:
                frame = self.get_current_frame(camera)

            face_extra = None
            face_save_score = None
            face_event_id = None
            # Secondary face detection on frames containing a person (optional)
            try:
                if self._face_model and current_tracked_objects is not None:
                    has_person = any(
                        (getattr(o, "label", None) == "person") or
                        (isinstance(o, dict) and o.get("label") == "person") or
                        (hasattr(o, "obj_data") and isinstance(getattr(o, "obj_data"), dict) and o.obj_data.get("label") == "person")
                        for o in current_tracked_objects
                    )
                    # Run face model whenever frame is available (no person gating)
                    if True or self._face_always_run:
                        logger.info(
                            f"Face inference: camera={camera}, has_person={has_person}, always_run={self._face_always_run}"
                        )
                        if frame is None:
                            logger.info(f"Face detection skipped: no frame available for camera {camera}")
                            # clear any pending face dets for tracker if we have no frame
                            self._pending_face_detections = []
                            face_extra = []
                        else:
                            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = self._face_model.predict(
                                img_rgb,
                                verbose=False,
                                save=False,
                            )
                            # Normalize face detections for DeepSORT (xyxy format per object)
                            face_norm_dets = []
                            try:
                                for r in results:
                                    names = getattr(r, "names", {}) or {}
                                    boxes = getattr(r, "boxes", None)
                                    if boxes is None:
                                        continue
                                    for b in boxes:
                                        try:
                                            # coords
                                            if hasattr(b, "xyxy"):
                                                arr = b.xyxy[0].tolist()
                                                x1, y1, x2, y2 = [int(v) for v in arr[:4]]
                                            elif hasattr(b, "xywh"):
                                                arr = b.xywh[0].tolist()
                                                cx, cy, w, h = arr[:4]
                                                x1, y1 = int(cx - w / 2), int(cy - h / 2)
                                                x2, y2 = int(cx + w / 2), int(cy + h / 2)
                                            else:
                                                continue
                                            # class/conf
                                            cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                                            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                                            label = names.get(cls_id, "face")
                                            face_norm_dets.append({
                                                "label": label,
                                                "score": conf,
                                                "x1": x1,
                                                "y1": y1,
                                                "x2": x2,
                                                "y2": y2,
                                            })
                                        except Exception:
                                            continue
                            except Exception:
                                face_norm_dets = []
                            # store for merging into DeepSORT tracker update
                            self._pending_face_detections = face_norm_dets
                            face_extra = face_norm_dets
                            face_score = 0.0
                            for r in results:
                                names = getattr(r, "names", {}) or {}
                                boxes = getattr(r, "boxes", None)
                                if boxes is None:
                                    continue
                                for b in boxes:
                                    try:
                                        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                                        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                                    except Exception:
                                        continue
                                    label = names.get(cls_id, str(cls_id))
                                    # common face models label faces as 'face', otherwise accept class id 0
                                    if label == "face" or cls_id == 0:
                                        if conf > face_score:
                                            face_score = conf
                            face_save_score = face_score
                            try:
                                logger.info(
                                    f"Face detection results: camera={camera}, dets={len(face_norm_dets)}, best_score={face_score:.3f}, threshold={self._face_threshold:.3f}"
                                )
                            except Exception:
                                pass
                            if face_score >= self._face_threshold:
                                event_id = uuid.uuid4().hex
                                face_event_id = event_id
                                include_recording = True
                                sub_label = None
                                duration = None
                                source_type = "api"
                                draw = {"boxes": []}
                                try:
                                    detect_cfg = self.config.cameras[camera].detect
                                    w = float(detect_cfg.width)
                                    h = float(detect_cfg.height)
                                    for fobj in face_norm_dets:
                                        fx1 = fobj.get("x1"); fy1 = fobj.get("y1"); fx2 = fobj.get("x2"); fy2 = fobj.get("y2")
                                        if None in (fx1, fy1, fx2, fy2):
                                            continue
                                        fx1, fy1, fx2, fy2 = map(float, (fx1, fy1, fx2, fy2))
                                        bw = max(0.0, (fx2 - fx1))
                                        bh = max(0.0, (fy2 - fy1))
                                        if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
                                            continue
                                        draw["boxes"].append(
                                            {
                                                "box": [fx1 / w, fy1 / h, bw / w, bh / h],
                                                "score": int(float(fobj.get("score", 0.0)) * 100),
                                                "color": (0, 120, 0),
                                            }
                                        )
                                except Exception:
                                    draw = {}
                                try:
                                    logger.info(
                                        f"Face event created: camera={camera}, event_id={event_id}, score={face_score:.3f}"
                                    )
                                except Exception:
                                    pass
                                self.create_manual_event(
                                    (
                                        frame_time,
                                        camera,
                                        "face",
                                        event_id,
                                        include_recording,
                                        face_score,
                                        sub_label,
                                        duration,
                                        source_type,
                                        draw,
                                    )
                                )
            except Exception as e:
                logger.debug(f"Face detection step skipped due to error: {e}")
                face_extra = getattr(self, "_pending_face_detections", None)

            # -------------- tracking --------------
            # IMPORTANT: do not replace Frigate's tracked_objects pipeline.
            # If we overwrite tracked_objects here, core event snapshot/clip saving breaks.
            deep_tracked_objects = None
            if self.deep_tracker:
                # Feed detections into DeepOCSORT (for overlay/track id purposes only)
                detections_for_tracker = current_tracked_objects if current_tracked_objects else []
                if isinstance(detections_for_tracker, tuple):
                    detections_for_tracker = list(detections_for_tracker)
                if face_extra:
                    try:
                        detections_for_tracker = list(detections_for_tracker) + list(face_extra)
                    except Exception:
                        pass
                self.deep_tracker.match_and_update(
                    frame_name=frame_name,
                    frame_time=frame_time,
                    detections=detections_for_tracker,
                )
                deep_tracked_objects = list(self.deep_tracker.get_tracked_objects().values())

            tracked_objects = [
                o.to_dict() for o in camera_state.tracked_objects.values()
            ]

            if self._draw_clip_overlays:
                if frame is not None and tracked_objects is not None:
                    try:
                        annotated = frame.copy()
                        try:
                            import time as _time
                            sec = _time.strftime('%S', _time.localtime(frame_time))
                        except Exception:
                            sec = f"{int(frame_time)%60:02d}"
                        overlay_objects = deep_tracked_objects if deep_tracked_objects is not None else tracked_objects
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
                                    x2, y2 = int(bx["x"] + bx["w"]), int(bx["y"] + bx["h"])
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

        self.requestor.stop()
        self.detection_publisher.stop()
        self.event_sender.stop()
        self.event_end_subscriber.stop()
        self.sub_label_subscriber.stop()
        self.camera_config_subscriber.stop()

        logger.info("Exiting object processor...")
