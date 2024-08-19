import sys
import asyncio
from threading import Thread
from confluent_kafka import Consumer, KafkaException, KafkaError


class AIOConsumer:

    def __init__(self, conf, topics: list[str], loop=None, on_message=None):
        self.conf = conf
        self.topics = topics
        self.on_message = on_message
        self.loop = loop or asyncio.get_event_loop()
        self.consumer = Consumer(conf)
        self._cancelled = False
        self.loop.create_task(self._consume_loop()) 

    async def _consume_loop(self):
        self.consumer.subscribe(self.topics)
        while not self._cancelled:
            msg = self.consumer.poll(timeout=0.1)
           
            if msg is None: continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    sys.stderr.write('%% %s [%d] reached end at offset %d\n' %
                                     (msg.topic(), msg.partition(), msg.offset()))
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                print('Received message: {}'.format(msg.value().decode('utf-8')))
                if self.on_message: # check if on_message is a function
                    await self.on_message(msg)
                    
    
    def close(self):
        self.consumer.close()
        self._cancelled = True
