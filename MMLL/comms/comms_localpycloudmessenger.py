# -*- coding: utf-8 -*-
'''
Comms interface to pycloudmessenger
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
try:
    import pycloudmessenger.ffl.abstractions as ffl
    import pycloudmessenger.ffl.fflapi as fflapi
    import pycloudmessenger.serializer as serializer
except:
    print("pycloudmessenger is not installed, use:")
    print("pip install https://github.com/IBM/pycloudmessenger/archive/v0.3.0.tar.gz")

    def get_current_task_name(self):
        task_available = False
        while not task_available:
            try:
                with open('current_taskname.txt', 'r') as f:
                    self.task_name = f.read()
                task_available = True
            except:
                print('No available task yet...')
                time.sleep(1)
                pass
        return self.task_name


class Comms_master:
    """
    """

    def __init__(self, context_master, task_name):
        """
        """
        #self.comms = Comms_master(commsffl)
        self.context_master = context_master
        self.task_name = task_name
        self.name = 'pycloudmessenger'
        self.commsffl = ffl.Factory.aggregator(self.context_master, task_name=task_name)

    def send(self, message, destiny):

        with self.commsffl:
            # self.send_to maps between worker_id and pseudo_id 
            self.commsffl.send(message, destiny)

    def broadcast(self, message, receivers_list=None):
        # receivers_list are not used here, pycloudmessenger already knows all the recipients
        with self.commsffl:
            self.commsffl.send(message)

    def receive(self, timeout=1):
        try:
            with self.commsffl:
                packet = self.commsffl.receive(timeout)
            message = packet.content
            pseudo_id = packet.notification['participant']
            #sender_ = str(self.workers_addresses_cloud.index(pseudo_id))
            #sender = message['sender']
            #message.update({'pseudo_id': pseudo_id})
            #message.update({'sender_': sender})
            message.update({'sender': pseudo_id})
        except:
            message = None
            pass

        return message

class Comms_worker:
    """
    """

    def __init__(self, context_w, task_name, worker_real_name):
        """
        """
        self.id = worker_real_name  # unused by now...
        self.task_name = task_name
        self.commsffl = ffl.Factory.participant(context_w, task_name=self.task_name)
        self.name = 'pycloudmessenger'

    def send(self, message, address=None):
        # address is not used here, a worker can only send to the master        
        with self.commsffl:
            self.commsffl.send(message)

    def receive(self, timeout=1):
        try:
            with self.commsffl:
                packet = self.commsffl.receive(timeout)
            message = packet.content
        except:
            message = None
            pass
        return message
