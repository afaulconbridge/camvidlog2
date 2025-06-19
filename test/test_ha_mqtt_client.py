from camvidlog2.ha_mqtt_client import MQTTClient


def test_basic(mqtt_broker):
    topic = "pytest/mqtt_fixture"
    payload = "fixture-test"
    with MQTTClient(
        broker=mqtt_broker["host"],
        port=mqtt_broker["port"],
    ) as client:
        client.subscribe(topic)
        client.publish(topic, payload)
        received = client.wait_for_message(timeout=2)
    assert received == payload
