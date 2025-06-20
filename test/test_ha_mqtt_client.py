from typing import Any
import random
import string

from camvidlog2.ha_mqtt_client import MQTTClient

def test_basic(mqtt_broker: dict[str, Any]) -> None:
    """Test basic MQTT publish and subscribe functionality.
    
    Verifies that a message can be published to a topic and subsequently
    received through subscription to the same topic.
    """
    # Given: A topic, payload, and MQTT client connection
    topic = "pytest/mqtt_fixture"
    payloads = [
        "".join(random.choices(string.ascii_letters + string.digits, k=8))
        for _ in range(10)
    ]
    receiveds = []

    with MQTTClient(
        broker=mqtt_broker["host"],
        port=mqtt_broker["port"],
    )
    
    # When: We subscribe to a topic, publish messages, and read them
    with client:
        client.subscribe(topic)

        for payload in payloads:
            client.publish(topic, payload)
            receiveds.append(client.wait_for_message(timeout=2))

    # Then: The received messages should match what we published
    assert payloads == receiveds
