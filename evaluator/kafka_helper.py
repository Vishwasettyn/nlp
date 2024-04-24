from kafka import KafkaProducer
from kafka import KafkaConsumer
import streamlit as st

bootstrap_servers = "localhost:9092"
topic = "transcribe.data"


def produce_message(message):
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    try:
        producer.send(topic, value=message.encode('utf-8'))
        producer.flush()
        print(f"Produced message to topic '{topic}': {message}")
    except Exception as e:
        print(f"Error producing message: {e}")
    finally:
        producer.close()


def consume_messages():
    result_list = []  # List to store consumed messages

    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                             group_id='transcribe_consumer_group',
                             auto_offset_reset='earliest')

    try:
        while True:
            messages = consumer.poll(timeout_ms=1000)  # Poll for messages, waiting at most 1 second.

            if not messages:
                continue

            for partition, message_list in messages.items():
                for message in message_list:
                    result = message.value.decode('utf-8')
                    result_list.append(result)
                    print("result_list", result_list)
                    st.write(f"Spoken words are: {result}")

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

    return result_list
