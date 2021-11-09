# -*- coding: utf-8 -*-
'''
Local Communications library (Flask Server)
'''
__author__ = "IBM Research, Ireland"

import logging
import requests
import json
import time
import pickle
import base64

#logger = logging.getLogger(__name__)


def is_jsonable(x):
    """
    Checks whether a given object can be serialized via json.dumps.

    :param x: Object to be serialized
    :type x: arbitrary (typically a dictionary)
    :return: Whether or not the object can be seralized
    :rtype: boolean
    """
    try:
        json.dumps(x)
        return True
    except:
        return False


def serialize(x):
    """
    Serialize a given object.

    :param x: Object to be serialized
    :type x: arbitrary (typically a dictionary)
    :return: Serialized object
    :rtype: string
    """
    return base64.b64encode(pickle.dumps(x)).decode('utf-8')


def unserialize(x):
    """
    Unserialize a serialized object.

    :param x: Object to be unserialized
    :type x: string
    :return: Unserialized object
    :rtype: arbitrary (typically a dictionary)
    """
    return pickle.loads(base64.b64decode(x.encode()))


class Comms:
    """
    This class implements basic communication functionality for sending and receiving 
    messages e.g. to be used in a Federated ML context. 
    """

    def __init__(self, workers_ids=None, my_id=None, url='http://localhost', port=5000, wait=0.1, timeout=60.):
        """
        Create a :class:`Comms` instance.

        :param my_id: Identifier for the sender of messages via this instance.
        :type my_id: int
        :param url: URL of the Musketeer platform instance providing the backend.
        :type url: string
        :param port: Port via which to send/receive messages.
        :type port: int
        :param wait: How many seconds to wait between pollings of received messages.
        :type wait: float
        :param timeout: How many seconds to maximally wait for received messages.
        :type timeout: float
        """
        self.id = my_id
        self.url = url
        self.port = port
        self.path = self.url + ':' + str(self.port) + '/'
        self.wait = wait
        self.timeout = timeout
        self.workers_ids = workers_ids
        self.name = 'localflask'

    def send(self, message, receiver):
        """
        Send message for designated receiver.

        :param receiver: Id of designated receiver
        :type receiver: int
        :param message: Message to be sent
        :type message: arbitrary (typically a dictionary)
        """

        if is_jsonable(message):
            message = {'serialized': False, 'arg': message.copy()}
        else:
            message = {'serialized': True, 'arg': serialize(message.copy())}

        payload = {'sender': self.id, 'receiver': receiver, 'message': json.dumps(message)}
        r = requests.post(self.path + 'send/', params=payload)

        if r.status_code != requests.codes.ok:
            raise Exception('Unexpected status code when sending message: %i' % r.status_code)

        #logger.info('Sent message (sender=%s, receiver=%s, serialized=%r).' % (str(self.id), str(receiver), message['serialized']))

    def receive(self, sender, timeout=None):
        """
        Receive message from designated sender.

        :param sender: Id of designated sender
        :type sender: int
        :param timeout: How many seconds to maximally wait for message to be received. 
                        If not specified, self.timeout will be used.
        :type timeout: float
        :return: Received message
        :rtype: arbitrary (typically a dictionary)
        """
        if timeout is None:
            timeout = self.timeout
        payload = {"sender": sender, "receiver": self.id}
        start = time.time()

        while time.time() - start < timeout:
            r = requests.get(self.path + 'receive/', params=payload)

            if r.status_code == requests.codes.ok:
                message = json.loads(json.loads(r.text)['message'])
                #logger.info('Received message (sender=%s, receiver=%s, Len. serialized=%r) after %f seconds.'
                #            % (str(sender), str(self.id), message['serialized'], time.time()-start))
                if message['serialized']:
                    message = unserialize(message['arg'])
                else:
                    message = message['arg']
                return message

            if r.status_code != requests.codes.no_content:
                raise Exception('Unexpected status code when receiving message: %i' % r.status_code)

            time.sleep(self.wait)

        raise Exception('Timeout when receiving data (%f over %f seconds)' % ((time.time()-start), timeout))

    def broadcast(self, message, receivers_list):
        """
        Send message for designated receivers.

        :param message: Message to be sent
        :type message: arbitrary (typically a dictionary)
        :param receivers_list: Ids of designated receivers
        :type receivers_list: list of int
        """
        if is_jsonable(message):
            message = {'serialized': False, 'arg': message.copy()}
        else:
            message = {'serialized': True, 'arg': serialize(message.copy())}

        for addr in receivers_list:
            payload = {'sender': self.id, 'receiver': addr, 'message': json.dumps(message)}
            r = requests.post(self.path + 'send/', params=payload)
            if r.status_code != requests.codes.ok:
                raise Exception('Unexpected status code when sending message: %i' % r.status_code)

        #logger.info('Broadcasted message to %d receivers (sender=%s, serialized=%r).' % (len(receivers_list), str(self.id), message['serialized']))

    def roundrobin(self, message, receivers_list):
        text = 'Not implemented yet.'
        raise Exception(text)
        print(text)
        return
