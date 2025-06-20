import threading
from typing import Any, Self

import paho.mqtt.client as mqtt

# see https://stevessmarthomeguide.com/adding-an-mqtt-device-to-home-assistant/


class MQTTClient:
    """
    Context-managed MQTT client for publishing and subscribing messages.
    Usage:
        with MQTTClient(host, port, ...) as client:
            client.publish(topic, payload)
            client.subscribe(topic)
            msg = client.wait_for_message(timeout=5)
    """

    broker: str
    port: int
    client: mqtt.Client
    _connect_event: threading.Event | None = None

    def __init__(
        self,
        broker: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self.broker = broker
        self.port = port

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username and password:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

        self._message_event: threading.Event = threading.Event()

        self._received_message: str | None = None
        self._subscribe_topic: str | None = None

    def __enter__(self) -> Self:
        # Create the event before connecting
        self._connect_event = threading.Event()
        self.client.loop_start()
        self.client.connect(self.broker, self.port, 60)
        # Wait until connected
        if not self._connect_event.wait(timeout=5):
            raise TimeoutError("MQTT client could not connect to broker in time.")
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.client.loop_stop()
        self.client.disconnect()
        self._connect_event = None

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: dict[str, Any],
        rc: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        """Callback when the internal client has connected"""
        if self._connect_event is None:
            raise ValueError("Client not awaiting connection")
        self._connect_event.set()

    def _on_message(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        """Callback when the internal client has received a message"""
        if self._subscribe_topic is None or msg.topic == self._subscribe_topic:
            self._received_message = msg.payload.decode()
            self._message_event.set()

    def subscribe(self, topic: str) -> None:
        if self._subscribe_topic is not None:
            raise ValueError("Client can only subscribe to one topic")
        self._subscribe_topic = topic
        error, _ = self.client.subscribe(topic)
        if error != mqtt.MQTTErrorCode.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"MQTT error {error}")

    def publish(
        self,
        topic: str,
        payload: str,
        qos: int = 1,
        retain: bool = False,
        timeout: float | None = 5,
    ) -> None:
        info = self.client.publish(topic, payload=payload, qos=qos, retain=retain)
        info.wait_for_publish(timeout)

    def wait_for_message(self, timeout: float = 5) -> str | None:
        """
        Wait (blocking) for a message to arrive on the subscribed topic.
        Returns the message payload as string, or None if timeout.
        """
        self._message_event.clear()
        self._received_message = None
        if self._message_event.wait(timeout):
            return self._received_message
        return None
