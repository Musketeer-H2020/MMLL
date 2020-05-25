# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM1

'''

__author__ = "Marcos Fernandez Diaz"
__date__ = "May 2020"

import sys

from MMLL.models.POM1.CommonML.POM1_ML import POM1ML



class POM1_CommonML_Master(POM1ML):
    """
    This class implements the Common ML operations, run at Master node. It inherits from POM1ML.
    """

    def __init__(self, workers_addresses, comms, logger, verbose=False):
        """
        Create a :class:`POM1_CommonML_Master` instance.

        Parameters
        ----------
        master_address: string
            address of the master node

        workers_addresses: list of strings
            list of the addresses of the workers

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        """
        self.name = 'POM1_CommonML_Master'          # Name
        self.workers_addresses = workers_addresses
        self.logger = logger                        # logger
        self.comms = comms                          # comms lib
        self.verbose = verbose                      # print on screen when true

        self.platform = comms.name                  # Type of comms to use (either 'pycloudmessenger' or 'localflask')



    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        users_addresses_terminate: list of strings
            addresses of the workers to be terminated

        """
        packet = {'action': 'STOP'}
        # Broadcast packet to all workers
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ' sent STOP to all Workers')



    def reset(self):
        """
        Create/reset some empty variables needed by the Master Node
        """
        self.display(self.name + ': Resetting local data')
        self.list_centroids = []
        self.list_counts = []
        self.list_dists = []
        self.list_public_keys = []
        self.list_gradients = []
    
    

    def checkAllStates(self, condition, state_dict):
        """
        Checks if all worker states satisfy a given condition
        """
        all_active = True
        for worker in self.workers_addresses:
            if state_dict[worker] != condition:
                all_active = False
                break
        return all_active



    def train_Master(self):
        """
        This is the main training loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Process the received packets
            - Perform actions according to the state

        Parameters
        ----------
        None
        """        
        self.state_dict.update({'CN': 'START_TRAIN'})
        self.display(self.name + ': Starting training')

        while self.state_dict['CN'] != 'END':
            self.Update_State_Master()
            self.TakeAction_Master()
            self.CheckNewPacket_Master()
            
        self.display(self.name + ': Training is done')
    
    
    '''
    def broadcast(self, packet):
        # Broadcast packet to all workers
        if self.platform == 'pycloudmessenger':
            with self.comms:
                self.comms.send(packet)
        else:
            self.comms.broadcast(packet, self.workers_addresses)'''



    def CheckNewPacket_Master(self):
        """
        Checks if there is a new message in the Master queue

        Parameters
        ----------
            None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(10) # We only receive a dictionary at a time even if there are more than 1 workers
                try:  # For the pycloudmessenger cloud
                    sender = packet.notification['participant']
                except Exception: # For the pycloudmessenger local
                    self.counter = (self.counter + 1) % self.Nworkers
                    sender = self.workers_addresses[self.counter]
                    
                packet = packet.content
                self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                self.ProcessReceivedPacket_Master(packet, sender)
            except KeyboardInterrupt:
                self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                sys.exit()
            except Exception:
                raise
                #pass
        else: # Local flask
            packet = None
            sender = None
            for sender in self.workers_addresses:
                try:
                    packet = self.comms.receive(sender, timeout=10)
                    self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                    self.ProcessReceivedPacket_Master(packet, sender)
                except KeyboardInterrupt:
                    self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                    sys.exit()
                except Exception as err:
                    if str(err).startswith('Timeout when receiving data'): # TimedOutException
                        pass
                    else:
                        raise

       
        

#===============================================================
#                 Worker   
#===============================================================

class POM1_CommonML_Worker(POM1ML):
    '''
    Class implementing the POM1 Common operations, run at Worker

    '''

    def __init__(self, logger, verbose=False):
        """
        Create a :class:`POM1_CommonML_Worker` instance.

        Parameters
        ----------
        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen
        """
        self.name = 'POM1_CommonML_Worker'      # Name
        self.logger = logger                    # logger
        self.verbose = verbose                  # print on screen when true



    def run_worker(self):
        """
        This is the training executed at every Worker

        Parameters
        ----------
            None
        """
        self.display(self.name + ' %s: READY and waiting instructions' %(self.worker_address))
        self.terminate = False

        while not self.terminate:
            self.CheckNewPacket_worker()



    def CheckNewPacket_worker(self):
        """
        Checks if there is a new message in the Worker queue

        Parameters
        ----------
            None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(timeout=10)
                packet = packet.content
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))
                self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception:
                pass
        else: # Local flask
            packet = None
            sender = None
            try:
                packet = self.comms.receive(self.master_address, timeout=10)
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))
                self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception as err:
                if str(err).startswith('Timeout when receiving data'): # TimedOutException
                    pass
                else:
                    raise


'''
    def send_master(self, packet):
        if self.platform == 'pycloudmessenger':
            with self.comms:
                self.comms.send(packet)
        else:
            self.comms.send(packet, self.master_address)'''
