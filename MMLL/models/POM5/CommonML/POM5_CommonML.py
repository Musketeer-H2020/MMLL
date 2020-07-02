# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM5

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "May 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
import pickle
from transitions import State
from transitions.extensions import GraphMachine
import time

class POM5_CommonML_Master(Common_to_all_POMs):
    """
    This class implements ML operations common to POM5 algorithms, run at Master node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, comms, logger, verbose=False, **kwargs):
        """
        Create a :class:`POM5_CommonML_Master` instance.

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
        **kwargs: Arbitrary keyword arguments.


        -----------------------------------------------------------------
        Optional or POM dependant arguments

        -----------------------------------------------------------------

        Parameters
        ---------------------
        cr: encryption object instance
            the encryption library to be used in POMs 4 and 5

        cryptonode_address: string
            address of the crypto node

        Nmaxiter: integer
            Maximum number of iterations during learning

        NC: integer
            Number of centroids

        regularization: float
            Regularization parameter

        classes: list of strings
            Possible class values in a multiclass problem

        balance_classes: Boolean
            If True, the algorithm takes into account unbalanced datasets

        C: array of floats
            Centroids matrix

        nf: integer
            Number of bits for the floating part

        N: integer
            Number of

        fsigma: float
            factor to multiply standard sigma value = sqrt(Number of inputs)

        normalize_data: Boolean
            If True, data normalization is applied, irrespectively if it has been previously normalized

        """
        self.name = 'POM5_CommonML_Master'               # Name
        self.master_address = master_address
        self.cryptonode_address = 'ca'

        # Convert workers_addresses -> '0', '1', + send_to dict
        self.broadcast_addresses = workers_addresses
        self.Nworkers = len(workers_addresses)                    # Nworkers
        self.workers_addresses = list(range(self.Nworkers))
        self.workers_addresses = [str(x) for x in self.workers_addresses]
        
        self.send_to = {}
        self.receive_from = {}
        for k in range(self.Nworkers):
            self.send_to.update({str(k): workers_addresses[k]})
            self.receive_from.update({workers_addresses[k]: str(k)})

        self.cryptonode_address = None

        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.verbose = verbose                      # print on screen when true

        self.process_kwargs(kwargs)
        self.encrypter = self.cr.get_encrypter()  # to be shared
        self.decrypter = self.cr.get_decrypter()  # to be kept as secret

        self.state_dict = None                      # State of the main script
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.NI_dict = {}
        self.Data_encr_dict = {}  # X data encrypted with PK from every user
        self.Xq_prodpk_dict = {}  # X data reencypted with PRodPK
        self.yq_prodpk_dict = {}  # y data reencypted with PRodPK

        for addr in self.workers_addresses:
            self.state_dict.update({addr: ''})
        self.create_FSM_master()
        self.message_counter = 0    # used to number the messages
        self.worker_names = {} # dictionary with the mappings worker_id -> pseudo_id


    def create_FSM_master(self):
        """
        Creates a Finite State Machine to be run at the Master Node

        Parameters
        ----------
        None
        """
        self.display(self.name + ': creating Common ML FSM, POM4')

        '''
        with open('../MMLL/models/POM5/CommonML/POM5_CommonML_FSM_master.pkl', 'rb') as f:
            [states_master, transitions_master] = pickle.load(f)
        '''
        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),

            State(name='sending_prep_object', on_enter=['while_sending_prep_object']),
            State(name='asking_local_prep', on_enter=['while_asking_local_prep']),

            State(name='bcasting_encrypter', on_enter=['while_bcasting_encrypter']),

            State(name='Exit', on_enter=['while_Exit'])
        ]

        transitions_master = [
            ['go_sending_prep_object', 'waiting_order', 'sending_prep_object'],
            ['go_asking_local_prep', 'sending_prep_object', 'asking_local_prep'],
            ['go_waiting_order', 'asking_local_prep', 'waiting_order'],
            ['go_bcasting_encrypter', 'waiting_order', 'bcasting_encrypter'],
            ['go_waiting_order', 'bcasting_encrypter', 'waiting_order'],
            ['go_Exit', 'waiting_order', 'Exit'],
            ['go_waiting_order', 'Exit', 'waiting_order']
        ]

        class FSM_master(object):

            def while_waiting_order(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' is waiting...')
                except:
                    print('STOP AT while_waiting_order')
                    import code
                    code.interact(local=locals())
                return

            def while_sending_prep_object(self, MLmodel):
                try:
                    action = 'store_prep'
                    data = {'prep_object': MLmodel.prep}
                    packet = {'action': action, 'to': 'CommonML', 'data': data}
                    MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    '''
                    for k in range(0, MLmodel.Nworkers):
                        MLmodel.comms.send(packet, MLmodel.workers_addresses[k])
                    '''
                    MLmodel.display(MLmodel.name + ' sent preprocessing object to all Workers')
                except:
                    print('STOP AT while_sending_prep_object')
                    import code
                    code.interact(local=locals())
                return

            def while_asking_local_prep(self, MLmodel):
                try:
                    packet = {'action': 'do_local_prep', 'to': 'CommonML'}
                    MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    '''
                    for k in range(0, MLmodel.Nworkers):
                        MLmodel.comms.send(packet, MLmodel.workers_addresses[k])
                    '''
                    MLmodel.display(MLmodel.name + ' sent do_local_prep to all Workers')
                except:
                    print('STOP AT while_asking_local_prep')
                    import code
                    code.interact(local=locals())
                return

            def while_bcasting_encrypter(self, MLmodel):
                try:
                    # Communicating encrypter to workers
                    data = {'encrypter': MLmodel.cr.encrypter}
                    # For checking, REMOVE
                    message_id = MLmodel.master_address + str(MLmodel.message_counter)
                    MLmodel.message_counter += 1
                    #data.update({'decrypter': MLmodel.cr.decrypter})
                    packet = {'action': 'send_encrypter', 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address, 'message_id': message_id}
                    MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    MLmodel.display(MLmodel.name + ' sent encrypter to all Workers')
                except:
                    print('ERROR AT while_bcasting_encrypter')
                    import code
                    code.interact(local=locals())
                return

            def while_Exit(self, MLmodel):
                try:
                    print(MLmodel.name + 'while_Exit')
                except:
                    print('STOP AT while_Exit')
                    import code
                    code.interact(local=locals())
                return

        self.FSMmaster = FSM_master()
        self.grafmachine_master = GraphMachine(model=self.FSMmaster,
            states=states_master,
            transitions=transitions_master,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling CommonML at master",
            show_conditions=False)
        return

    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        users_addresses_terminate: list of strings
            addresses of the workers to be terminated

        """
        message_id = self.master_address + str(self.message_counter)
        self.message_counter += 1            
        packet = {'action': 'STOP', 'to': 'CommonML', 'sender': self.master_address, 'message_id': message_id}
        if workers_addresses_terminate is None:  # We terminate all of them
            workers_addresses_terminate = self.workers_addresses
            self.display('CommonML_Master sent STOP to all Workers')
        else:
            self.display('CommonML_Master sent STOP to %d Workers' % len(workers_addresses_terminate))

        '''
        self.comms.broadcast(packet, MLmodel.workers_addresses)
        for address in workers_addresses_terminate:
            self.comms.send(address, packet)
        '''
        self.comms.broadcast(packet, workers_addresses_terminate)

        # Updating the list of active users
        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)
        #self.FSMmaster.go_Exit(self)
        #self.FSMmaster.go_waiting_order(self)

    ######## CHECKED




    ######## UNCHECKED

    def send_encrypter(self):
        """
        Send encrypter to all nodes

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_bcasting_encrypter(self)
        self.run_Master()



    '''
    def get_params(self):
        """
        Obtain common cryptographic parameters to be shared, under POM4

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_asking_encrypt_params(self)
        self.run_Master()
    '''


    def get_cryptdata(self):
        """
        Get encrypted data from workers, under POM4

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_asking_encrypted_data(self)
        self.run_Master()

    def reencrypt_data(self):  # Includes removing blinding
        """
        Ask cryptonode to reencrypt the data to a common key

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_reencrypting_data(self)
        self.run_Master()

    def reset(self, NI):
        """
        Create some empty variables needed by the Master Node

        Parameters
        ----------
        NI: integer
            Number of input features
        """

        #self.w_decr = np.random.normal(0, 0.001, (self.NI + 1, 1))      # weights in plaintext, first value is bias
        self.NI = NI
        self.w_decr = None      # weights in plaintext, first value is bias
        self.R_central_decr = np.zeros((self.NI + 1, self.NI + 1))    # Cov. matrix in plaintext
        self.r_central_decr = np.zeros((self.NI + 1, 1))              # Cov. matrix in plaintext
        self.preds_dict = {}                                           # dictionary storing the prediction errors
        self.AUCs_dict = {}                                           # dictionary storing the prediction errors
        self.R_dict = {}
        self.r_dict = {}
        self.Pk_dict = {}
        self.Data_dict = {}
        self.display('CommonML_Master: Resetting local data')

    def local_prep_Master(self, prep_object):
        """
        This is the local preprocessing loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Process the received packets
            - Perform actions according to the state

        Parameters
        ----------
            None
        """
        self.prep = prep_object
        self.FSMmaster.go_sending_prep_object(self)
        self.display('CommonML_Master: Sending Preprocessing object')
        self.run_Master()
        self.display('CommonML_Master: Local Preprocessing is done')

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_stored_prep'):
            self.FSMmaster.go_asking_local_prep(self)

        if self.chekAllStates('ACK_local_prep'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sent_encrypter'):
            self.FSMmaster.go_waiting_order(self)

    def ProcessReceivedPacket_Master(self, packet, sender):
        """
        Process the received packet at Master and take some actions, possibly changing the state

        Parameters
        ----------
            packet: packet object 
                packet received (usually a dict with various content)

            sender: string
                id of the sender
        """
        if packet['action'][0:3] == 'ACK':
            self.display('Master received ACK from %s: %s' % (str(sender), packet['action']))
            self.state_dict[sender] = packet['action']

        if packet['action'] == 'ACK_sent_encrypter':
            self.NI_dict.update({sender: packet['data']['NI']})
            # This part could be moved to a more general first step retrieving the feature characteristics...
            #self.send_to.update({sender: packet['pseudo_id']})
            #self.worker_names.update({sender: packet['sender_']})
        return


class POM5_CommonML_Worker(Common_to_all_POMs):
    '''
    This class implements ML operations common to POM5 algorithms, run at Worker node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`POM4_CommonML_Worker` instance.

        Parameters
        ----------
        master_address: string
            address of the master node

        worker_address: string
            id of the worker

        model_type: string
            type of ML model

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen

        Xtr_b: ndarray
            2-D numpy array containing the input training patterns

        ytr: ndarray
            1-D numpy array containing the target training values
    
        **kwargs: Arbitrary keyword arguments.


        -----------------------------------------------------------------
        Optional or POM dependant arguments

        -----------------------------------------------------------------

        Parameters
        ---------------------

        cryptonode_address: string
            address of the crypto node

        Nmaxiter: integer
            Maximum number of iterations during learning

        NC: integer
            Number of centroids

        regularization: float
            Regularization parameter

        classes: list of strings
            Possible class values in a multiclass problem

        balance_classes: Boolean
            If True, the algorithm takes into account unbalanced datasets

        C: array of floats
            Centroids matrix

        nf: integer
            Number of bits for the floating part

        N: integer
            Number of

        fsigma: float
            factor to multiply standard sigma value = sqrt(Number of inputs)

        normalize_data: Boolean
            If True, data normalization is applied, irrespectively if it has been previously normalized
        """
        self.master_address = master_address
        self.worker_address = worker_address                    # The id of this Worker
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.comms = comms                      # The comms library
        self.model_type = model_type
        #self.cr = cr
        self.logger = logger                    # logger
        self.name = 'POM5_CommonML_Worker'           # Name
        self.verbose = verbose                  # print on screen when true
        self.Xtr_b = Xtr_b
        self.ytr = ytr
        self.create_FSM_worker()

    def create_FSM_worker(self):
        """
        Creates a Finite State Machine to be run at the Worker Node

        Parameters
        ----------
        None
        """
        self.display(self.name + ' %s: creating FSM' % (str(self.worker_address)))

        class FSM_worker(object):

            # Enter/exit callbacks are defined here

            def while_waiting_order(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' %s is waiting...' % (str(MLmodel.worker_address)))
                except:
                    print('STOP AT while_waiting_order')
                    import code
                    code.interact(local=locals())
                return

            def while_storing_prep_object(self, MLmodel, packet):
                try:
                    MLmodel.prep = packet['data']['prep_object']
                    MLmodel.display(MLmodel.name + ' %s: stored preprocessing object' % (str(MLmodel.worker_address)))
                    action = 'ACK_stored_prep'
                    packet = {'action': action}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_stored_prep' % (str(MLmodel.worker_address)))
                except:
                    print('STOP AT while_storing_prep_object')
                    import code
                    code.interact(local=locals())
                return

            def while_local_preprocessing(self, MLmodel):
                try:
                    X = np.copy(MLmodel.Xtr_b)
                    new_Xtr_b = MLmodel.prep.transform(X)
                    MLmodel.Xtr_b = np.copy(new_Xtr_b)
                    MLmodel.display(MLmodel.name + ' %s: locally preprocessing data...' % (str(MLmodel.worker_address)))
                    action = 'ACK_local_prep'
                    packet = {'action': action}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep' % (str(MLmodel.worker_address)))
                except:
                    print('STOP AT while_local_preprocessing')
                    import code
                    code.interact(local=locals())
                return

            def while_storing_encrypter(self, MLmodel, packet):
                try:
                    MLmodel.encrypter = packet['data']['encrypter']
                    #MLmodel.decrypter = packet['data']['decrypter']
                    action = 'ACK_sent_encrypter'
                    NI = MLmodel.Xtr_b.shape[1]
                    data = {'NI': NI}
                    packet = {'action': action, 'data': data, 'sender': str(MLmodel.worker_address)}
                    # Sending params to Master
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_encrypter')
                except:
                    print('STOP AT while_storing_encrypter')
                    import code
                    code.interact(local=locals())
                return

        '''
        with open('../MMLL/models/POM5/CommonML/POM5_CommonML_FSM_worker.pkl', 'rb') as f:
            [states_worker, transitions_worker] = pickle.load(f)
        '''

        states_worker = [
            'Exit',
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='storing_prep_object', on_enter=['while_storing_prep_object']),
            State(name='local_preprocessing', on_enter=['while_local_preprocessing']),
            State(name='storing_encrypter', on_enter=['while_storing_encrypter']),
        ]

        transitions_worker = [
            ['go_Exit', 'waiting_order', 'Exit'],

            ['go_storing_prep_object', 'waiting_order', 'storing_prep_object'],
            ['done_storing_prep_object', 'storing_prep_object', 'waiting_order'],

            ['go_local_preprocessing', 'waiting_order', 'local_preprocessing'],
            ['done_local_preprocessing', 'local_preprocessing', 'waiting_order'],

            ['go_storing_encrypter', 'waiting_order', 'storing_encrypter'],
            ['done_storing_encrypter', 'storing_encrypter', 'waiting_order'],
        ]


        self.FSMworker = FSM_worker()
        self.grafmachine_worker = GraphMachine(model=self.FSMworker,
            states=states_worker,
            transitions=transitions_worker,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of worker No. %s" % str(self.worker_address),
            show_conditions=False)
        return

    def ProcessReceivedPacket_Worker(self, packet, sender):
        """
        Take an action after receiving a packet

        Parameters
        ----------
            packet: packet object 
                packet received (usually a dict with various content)

            sender: string
                id of the sender
        """
        self.terminate = False

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
            self.display('EXIT WORKER')
            time.sleep(2)
            self.terminate = True

        if packet['action'] == 'store_prep':
            self.FSMworker.go_storing_prep_object(self, packet)
            self.FSMworker.done_storing_prep_object(self)

        if packet['action'] == 'do_local_prep':
            self.FSMworker.go_local_preprocessing(self)
            self.FSMworker.done_local_preprocessing(self)

        if packet['action'] == 'send_encrypter':
            self.FSMworker.go_storing_encrypter(self, packet)
            self.FSMworker.done_storing_encrypter(self)

        return self.terminate

