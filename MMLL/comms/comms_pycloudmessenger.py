# -*- coding: utf-8 -*-
'''
Comms interface to pycloudmessenger with roundrobin functionality.
@author:  Angel Navia Vázquez, Marcos Fernández Díaz.
'''
__author__ = "Angel Navia Vázquez, Marcos Fernández Díaz"

import random, string
import time
'''
try:
    import pycloudmessenger.ffl.abstractions as ffl
    import pycloudmessenger.ffl.fflapi as fflapi
    import pycloudmessenger.serializer as serializer
except:
    print("pycloudmessenger is not installed, use:")
    print("pip install https://github.com/IBM/pycloudmessenger/archive/v0.3.0.tar.gz")
'''

def get_current_task_name(self):
    """
    Function to retrieve the current task name from local disk.

    Parameters
    ----------
    None

    Returns
    -------
    task_name: string
        The current task name currently created from the master.
    """
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
    This class provides an interface with the communication object, run at Master node.
    """

    def __init__(self, commsffl):
        """
        Create a :class:`Comms_master` instance.

        Parameters
        ----------
        commsffl: :class:`ffl.Factory.aggregator`
            Object providing communication functionalities at Master for pycloudmessenger.
        """
        #self.comms = Comms_master(commsffl)
        #self.context_master = context_master
        #self.task_name = task_name
        self.name = 'pycloudmessenger'
        #self.commsffl = ffl.Factory.aggregator(self.context_master, task_name=task_name)
        self.commsffl = commsffl
        workers = self.commsffl.get_participants()
        self.workers_ids = list(workers.keys())


    def send(self, message, destiny):
        """
        Send a packet to a given destination.

        Parameters
        ----------
        message: dict
            Packet to be sent.
        destiny: string
            Address of the recipient for the message.
        """
        try:
            with self.commsffl:
                # self.send_to maps between worker_id and pseudo_id 
                self.commsffl.send(message, destiny, topology='STAR') 
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at send: %s' % err)
            print('*' * 80)
            print('\n')
            raise


    def broadcast(self, message, receivers_list=None):
        """
        Send a packet to a set of workers.

        Parameters
        ----------
        message: dict
            Packet to be sent.
        receivers_list: list of strings
            Addresses of the recipients for the message.
        """
        # receivers_list are not used here, pycloudmessenger already knows all the recipients
        try:
            if receivers_list is None:
                with self.commsffl:
                    self.commsffl.send(message, topology='STAR')
            else:
                for destiny in receivers_list:
                    with self.commsffl:
                        self.commsffl.send(message, destiny, topology='STAR')
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at broadcast: %s' % err)
            print('*' * 80)
            print('\n')
            raise


    def roundrobin(self, message, receivers_list=None):
        """
        Send a packet to a set of workers using the roundrobin protocol (ring communications).

        Parameters
        ----------
        message: dict
            Packet to be sent.
        receivers_list: list of strings
            Addresses of the recipients for the message.
        """
        # receivers_list are not used here, pycloudmessenger already knows all the recipients
        try:
            with self.commsffl:
                self.commsffl.send(message, topology='RING')
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at roundrobin: %s' % err)
            print('*' * 80)
            print('\n')
            raise


    def receive(self, timeout=1):
        """
        Wait for a packet to arrive or until timeout expires.

        Parameters
        ----------
        timeout: float
            Time to wait for a packet in seconds.

        Returns
        -------
        message: dict
            Received packet.
        """
        try:
            message = None
            packet = None
            with self.commsffl:
                packet = self.commsffl.receive(timeout)

            if packet is not None:
                if packet.content is not None:
                    message = packet.content
                    pseudo_id = packet.notification['participant']
                    #sender_ = str(self.workers_addresses_cloud.index(pseudo_id))
                    #sender = message['sender']
                    #message.update({'pseudo_id': pseudo_id})
                    #message.update({'sender_': sender})
                    '''
                    print('----------- message update')
                    print(packet)
                    print('---------------------------')
                    print(message)
                    print('------------ message update')
                    '''
                    message.update({'sender': pseudo_id})
        except Exception as err:
            if 'pycloudmessenger.ffl.fflapi.TimedOutException' not in str(type(err)): # we skip the normal timeouts
                print('\n')
                print('*' * 80)
                print('Pycloudmessenger ERROR at receive: %s' % err)
                print('*' * 80)
                print('\n')
                print('STOP AT comms_pycloudmessenger')
                import code
                code.interact(local=locals())
            else:
                message = None
            raise
        return message


    def receive_poms_123(self, timeout=10):
        """
        Wait for a packet to arrive or until timeout expires. Used in POMs 1, 2 and 3.

        Parameters
        ----------
        timeout: float
            Time to wait for a packet in seconds.

        Returns
        -------
        packet: dict
            Received packet.
        """
        with self.commsffl:
            packet = self.commsffl.receive(timeout)            
        return packet



class Comms_worker:
    """
    This class provides an interface with the communication object, run at Worker node.
    """

    def __init__(self, commsffl, worker_real_name='Anonymous'):
        """
        Create a :class:`Comms_worker` instance.

        Parameters
        ----------
        commsffl: :class:`ffl.Factory.participant`
            Object providing communication functionalities for pycloudmessenger.
        worker_real_name: string
            Real name of the worker.
        """
        self.id = worker_real_name  # unused by now...
        #self.task_name = task_name
        #self.commsffl = ffl.Factory.participant(context_w, task_name=self.task_name)
        self.name = 'pycloudmessenger'
        self.commsffl = commsffl


    def send(self, message, address=None):
        """
        Send a packet to the master.

        Parameters
        ----------
        message: dict
            Packet to be sent.
        address: string
            Address of the recipient for the message.
        """
        try:
            # address is not used here, a worker can only send to the master        
            with self.commsffl:
                self.commsffl.send(message)
        except Exception as err:
            print('\n')
            print('*' * 80)
            print('Pycloudmessenger ERROR at send: %s' % err)
            print('*' * 80)
            print('\n')
            raise


    def receive(self, timeout=0.1):
        """
        Wait for a packet to arrive or until timeout expires.

        Parameters
        ----------
        timeout: float
            Time to wait for a packet in seconds.

        Returns
        -------
        message: dict
            Received packet.
        """
        message = None
        try:
            with self.commsffl:
                packet = self.commsffl.receive(timeout)
            
            if packet is not None:
                '''
                print('=======================  RECEIVED PACKET ========================')
                print(packet)
                print('=================================================================')
                '''
                if packet.content is not None:
                    message = packet.content
        except Exception as err:
            #print(err)
            if 'pycloudmessenger.ffl.fflapi.TimedOutException' not in str(type(err)): # we skip the normal timeouts
                #if 'pika.exceptions.ChannelClosed' not in str(type(err)):
                print('\n')
                print('*' * 80)
                print('Pycloudmessenger ERROR at receive: %s' % err)
                print('*' * 80)
                print('\n')
                print('STOP AT comms_pycloudmessenger')
                import code
                code.interact(local=locals())
            else:
                message = None
        return message


    def receive_poms_123(self, timeout=10):
        """
        Wait for a packet to arrive or until timeout expires. Used in POMs 1, 2 and 3.

        Parameters
        ----------
        timeout: float
            Time to wait for a packet in seconds.

        Returns
        -------
        packet: dict
            Received packet.
        """
        with self.commsffl:
            packet = self.commsffl.receive(timeout)            
        return packet
