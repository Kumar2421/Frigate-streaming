"""Facilitates communication between processes."""

import logging
import multiprocessing as mp
import threading
from multiprocessing.synchronize import Event as MpEvent
from typing import Any, Callable

import zmq

from frigate.comms.base_communicator import Communicator

logger = logging.getLogger(__name__)

import os
import socket
import time
# Windows doesn't support IPC sockets, use TCP instead
if os.name == "nt":  # Windows
    BASE_PORT = 5555
    SOCKET_REP_REQ = f"tcp://127.0.0.1:{BASE_PORT}"
else:
    SOCKET_REP_REQ = "ipc:///tmp/cache/comms"


def _is_port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def _find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if not _is_port_in_use("127.0.0.1", port):
            return port
    return start_port  # Fallback to original port


class InterProcessCommunicator(Communicator):
    def __init__(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # Set linger to 0 for quick socket closure on Windows
        if os.name == "nt":  # Windows
            self.socket.setsockopt(zmq.LINGER, 0)
            # Try to find an available port if default is in use
            actual_port = _find_available_port(BASE_PORT)
            if actual_port != BASE_PORT:
                logger.warning(f"Port {BASE_PORT} is in use, using port {actual_port} instead")
            socket_addr = f"tcp://127.0.0.1:{actual_port}"
            # Update the global socket address for requestors
            global SOCKET_REP_REQ
            SOCKET_REP_REQ = socket_addr
        else:
            socket_addr = SOCKET_REP_REQ
        
        # Retry binding with delay if port is in use
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.socket.bind(socket_addr)
                logger.debug(f"Successfully bound to {socket_addr}")
                break
            except zmq.ZMQError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to bind to {socket_addr} (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                else:
                    logger.error(f"Failed to bind to {socket_addr} after {max_retries} attempts: {e}")
                    raise
        self.stop_event: MpEvent = mp.Event()

    def publish(self, topic: str, payload: Any, retain: bool = False) -> None:
        """There is no communication back to the processes."""
        pass

    def subscribe(self, receiver: Callable) -> None:
        self._dispatcher = receiver
        self.reader_thread = threading.Thread(target=self.read)
        self.reader_thread.start()

    def read(self) -> None:
        while not self.stop_event.is_set():
            while True:  # load all messages that are queued
                has_message, _, _ = zmq.select([self.socket], [], [], 1)

                if not has_message:
                    break

                try:
                    raw = self.socket.recv_json(flags=zmq.NOBLOCK)

                    if isinstance(raw, list):
                        (topic, value) = raw
                        response = self._dispatcher(topic, value)
                    else:
                        logging.warning(
                            f"Received unexpected data type in ZMQ recv_json: {type(raw)}"
                        )
                        response = None

                    if response is not None:
                        self.socket.send_json(response)
                    else:
                        self.socket.send_json([])
                except zmq.ZMQError:
                    break

    def stop(self) -> None:
        self.stop_event.set()
        self.reader_thread.join()
        self.socket.close()
        self.context.destroy()


class InterProcessRequestor:
    """Simplifies sending data to InterProcessCommunicator and getting a reply."""

    def __init__(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        if os.name == "nt":  # Windows
            self.socket.setsockopt(zmq.LINGER, 0)
        # Use the global SOCKET_REP_REQ which may have been updated by InterProcessCommunicator
        self.socket.connect(SOCKET_REP_REQ)

    def send_data(self, topic: str, data: Any) -> Any:
        """Sends data and then waits for reply."""
        try:
            self.socket.send_json((topic, data))
            return self.socket.recv_json()
        except zmq.ZMQError:
            return ""

    def stop(self) -> None:
        self.socket.close()
        self.context.destroy()
