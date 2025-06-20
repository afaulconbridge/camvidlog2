from typing import Any

from camvidlog2.ha_mqtt_client import MQTTClient

def test_basic(mqtt_broker: dict[str, Any]) -> None:
    """Test basic MQTT publish and subscribe functionality.
    
    Verifies that a message can be published to a topic and subsequently
    received through subscription to the same topic.
    """
    # Given: A topic, payload, and MQTT client connection
    topic = "pytest/mqtt_fixture"
    payload = "fixture-test"
    client = MQTTClient(
        broker=mqtt_broker["host"],
        port=mqtt_broker["port"],
    )
    
    # When: We subscribe to a topic, publish a message, and wait for it
    with client:
        client.subscribe(topic)
        client.publish(topic, payload)
        received = client.wait_for_message(timeout=2)
    
    # Then: The received message should match what we published
    assert received == payload