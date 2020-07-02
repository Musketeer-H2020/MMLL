# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM4

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "May 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
import pickle
from transitions import State
from transitions.extensions import GraphMachine
import time

class POM4_CommonML_Master(Common_to_all_POMs):
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


        """
        self.name = 'POM4_CommonML_Master'               # Name
        self.pom = 4
        self.master_address = master_address
        self.cryptonode_address = None
        
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

        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.verbose = verbose                      # print on screen when true

        self.process_kwargs(kwargs)

        self.state_dict = None                      # State of the main script
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.NI_dict = {}
        self.X_encr_dict = {}
        self.y_encr_dict = {}

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

        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='asking_encrypter', on_enter=['while_asking_encrypter']),
            State(name='asking_encr_data', on_enter=['while_asking_encr_data']),
            State(name='sending_bl_data', on_enter=['while_sending_bl_data']),
            State(name='terminating_workers', on_enter=['while_terminating_workers']),
            State(name='Exit', on_enter=['while_Exit']),
        ]

        transitions_master = [
            ['go_asking_encrypter', 'waiting_order', 'asking_encrypter'],
            ['go_waiting_order', 'asking_encrypter', 'waiting_order'],

            ['go_asking_encr_data', 'waiting_order', 'asking_encr_data'],
            ['go_waiting_order', 'asking_encr_data', 'waiting_order'],

            ['go_sending_bl_data', 'waiting_order', 'sending_bl_data'],
            ['go_waiting_order', 'sending_bl_data', 'waiting_order'],

            ['go_terminating_workers', 'waiting_order', 'terminating_workers'],
            ['go_waiting_order', 'terminating_workers', 'waiting_order'],

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

            '''
            def while_sending_prep_object(self, MLmodel):
                try:
                    action = 'store_prep'
                    data = {'prep_object': MLmodel.prep}
                    packet = {'action': action, 'to': 'CommonML', 'data': data}
                    MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    #for k in range(0, MLmodel.Nworkers):
                    #    MLmodel.comms.send(packet, MLmodel.workers_addresses[k])
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
                    #for k in range(0, MLmodel.Nworkers):
                    #    MLmodel.comms.send(packet, MLmodel.workers_addresses[k])
                    MLmodel.display(MLmodel.name + ' sent do_local_prep to all Workers')
                except:
                    print('STOP AT while_asking_local_prep')
                    import code
                    code.interact(local=locals())
                return
            '''

            def while_asking_encrypter(self, MLmodel):
                try:
                    action = 'ask_encrypter'
                    data = {}
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    #MLmodel.comms.send(packet, MLmodel.cryptonode_address)
                    # We dont know the address of the cryptonode, we boradcast.
                    MLmodel.comms.broadcast(packet, MLmodel.broadcast_addresses)
                    
                    MLmodel.display(MLmodel.name + ' asking encrypter to cryptonode')
                except:
                    print('ERROR AT while_asking_encrypter')
                    import code
                    code.interact(local=locals())
                return

            def while_asking_encr_data(self, MLmodel):
                try:
                    # Communicating encrypter to workers
                    data = {'encrypter': MLmodel.encrypter}
                    # For checking, REMOVE
                    data.update({'decrypter': MLmodel.decrypter})
                    packet = {'action': 'ask_encr_data', 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.broadcast(packet, MLmodel.broadcast_addresses)
                    MLmodel.display(MLmodel.name + ' sent encrypter to all Workers and asked encr_data')
                except:
                    print('ERROR AT while_bcasting_encrypter')
                    import code
                    code.interact(local=locals())
                return


            '''
            def while_bcasting_encrypter(self, MLmodel):
                try:
                    # Communicating encrypter to workers
                    data = {'encrypter': MLmodel.cr.encrypter}
                    # For checking, REMOVE
                    data.update({'decrypter': MLmodel.cr.decrypter})
                    packet = {'action': 'send_encrypter', 'to': 'CommonML', 'data': data}
                    MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    MLmodel.display(MLmodel.name + ' sent encrypter to all Workers')
                except:
                    print('ERROR AT while_bcasting_encrypter')
                    import code
                    code.interact(local=locals())
                return
            '''

            def while_sending_bl_data(self, MLmodel):
                try:                    
                    # Encrypted data at MLmodel.X_encr_dict, MLmodel.y_encr_dict
                    # To store at MLmodel
                    MLmodel.BX_dict = {}
                    MLmodel.By_dict = {}
                    # To send to crypto
                    MLmodel.X_bl_dict = {}
                    MLmodel.y_bl_dict = {}

                    
                    for waddr in MLmodel.workers_addresses:
                        X = MLmodel.X_encr_dict[waddr]
                        y = MLmodel.y_encr_dict[waddr]
                        NP, NI = X.shape
                        BX = np.random.normal(0, 1, (NP, NI))
                        By = np.random.normal(0, 1, (NP, 1))
                        MLmodel.BX_dict.update({waddr: BX})
                        MLmodel.By_dict.update({waddr: By})
                        MLmodel.X_bl_dict.update({waddr: X + BX})
                        MLmodel.y_bl_dict.update({waddr: y + By})
                        '''
                        try:
                            aux = MLmodel.decrypter.decrypt(X + BX)
                            aux = MLmodel.decrypter.decrypt(y + By)
                            print('DECRYPT OK')
                        except:
                            print('STOP AT while_sending_bl_data')
                            import code
                            code.interact(local=locals())
                        '''

                    action = 'send_Xy_bl'
                    data = {'X_bl_dict': MLmodel.X_bl_dict, 'y_bl_dict': MLmodel.y_bl_dict}
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.display(MLmodel.name + ' sending Xy data blinded to cryptonode...')
                    MLmodel.comms.send(packet,  MLmodel.send_to[MLmodel.cryptonode_address])

                except:
                    print('ERROR AT while_sending_bl_data')
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

    '''
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

    def ask_encrypter(self):
        """
        Obtain encrypter from crypto, under POM4. The crypto also shares with the workers.

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_asking_encrypter(self)
        # We wait for the answer from the cryptonode
        self.run_Master()

    def get_cryptdata(self):
        """
        Get encrypted data from workers, under POM4

        Parameters
        ----------
        None
        """
        # Le mandamos al estado para activar el run_Master, pero no hace nada.
        self.FSMmaster.go_asking_encr_data(self)
        print('---------- waiting for cryptdata')
        self.run_Master()

        # Add blinding and share with cryptonode
        self.FSMmaster.go_sending_bl_data(self)
        self.FSMmaster.go_waiting_order(self)


    '''
    def reencrypt_data(self):  # Includes removing blinding
        """
        Ask cryptonode to reencrypt the data to a common key

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_reencrypting_data(self)
        self.run_Master()
    '''
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

        if self.chekAllStates('ACK_send_encr_data'):
           self.FSMmaster.go_waiting_order(self)

        #if self.chekAllStates('ACK_sent_encrypter'):
        #    self.FSMmaster.go_waiting_order(self)

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
            self.display('Master received ACK from %s: %s at common' % (str(sender), packet['action']))
            if sender != self.cryptonode_address:
                self.state_dict[sender] = packet['action']

        if packet['action'] == 'ACK_sent_encrypter':
            self.display('Storing encrypter')
            self.encrypter = packet['data']['encrypter']
            print('#### WARNING delete decrypter, CommonML #####')
            self.decrypter = packet['data']['decrypter']

            self.broadcast_addresses = list(set(self.workers_addresses) -set([sender]))
            
            self.workers_addresses = list(set(self.workers_addresses) -  set([sender]))
            self.display('Identified workers: ' + str(self.workers_addresses))
            self.cryptonode_address = sender
            self.display('Identified cryptonode as worker %s: address %s'% (str(self.cryptonode_address), str(self.send_to[sender])))

            #we update self.state_dict with the new list of workers_addresses
            self.state_dict = {}
            for waddr in self.workers_addresses:
                self.state_dict.update({waddr: ''})

            self.FSMmaster.go_waiting_order(self)

        if packet['action'] == 'ACK_send_encr_data':                  
            self.X_encr_dict.update({sender: packet['data']['Xtr_b_encr']})
            self.y_encr_dict.update({sender: packet['data']['ytr_encr']})

        return


class POM4_CommonML_Worker(Common_to_all_POMs):
    '''
    This class implements ML operations common to POM5 algorithms, run at Worker node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None, cryptonode_address=None):
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
        self.cryptonode_address = cryptonode_address
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.comms = comms                      # The comms library
        self.model_type = model_type
        #self.cr = cr
        self.logger = logger                    # logger
        self.name = 'POM4_CommonML_Worker'           # Name
        self.verbose = verbose                  # print on screen when true
        #self.Xtr_b = Xtr_b
        self.Xtr_b = self.add_bias(Xtr_b)
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
                    packet = {'action': action, 'sender': MLmodel.worker_address}
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
                    packet = {'action': action, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep' % (str(MLmodel.worker_address)))
                except:
                    print('STOP AT while_local_preprocessing')
                    import code
                    code.interact(local=locals())
                return
            
            def while_sending_encr_data(self, MLmodel, packet):

                try:
                    MLmodel.encrypter = packet['data']['encrypter']

                    MLmodel.display(MLmodel.name + ': stored encrypter, encrypting data...')
                    
                    # Encrypting data
                    MLmodel.Xtr_b_encr = MLmodel.encrypter.encrypt(MLmodel.Xtr_b)
                    MLmodel.ytr_encr = MLmodel.encrypter.encrypt(MLmodel.ytr)

                    action = 'ACK_send_encr_data'
                    data = {'Xtr_b_encr': MLmodel.Xtr_b_encr, 'ytr_encr': MLmodel.ytr_encr}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_send_encr_data')
                except:
                    print('ERROR AT while_storing_encrypt')
                    import code
                    code.interact(local=locals())
                return

            '''
            def while_sending_encr_data(self, MLmodel):
                try:
                    action = 'ACK_send_encr_data'
                    data = {'Xtr_b_encr': MLmodel.Xtr_b_encr, 'ytr_encr': MLmodel.ytr_encr}
                    packet = {'action': action, 'data': data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_send_encr_data')
                except:
                    print('ERROR AT while_sending_encr_data')
                    import code
                    code.interact(local=locals())
                return
            '''

        states_worker = [
            'Exit',
            State(name='waiting_order', on_enter=['while_waiting_order']),

            #State(name='storing_prep_object', on_enter=['while_storing_prep_object']),
            #State(name='local_preprocessing', on_enter=['while_local_preprocessing']),

            State(name='storing_encrypt', on_enter=['while_storing_encrypt']),

            State(name='sending_encr_data', on_enter=['while_sending_encr_data']),
        ]

        transitions_worker = [
            ['go_Exit', 'waiting_order', 'Exit'],

            #['go_storing_prep_object', 'waiting_order', 'storing_prep_object'],
            #['done_storing_prep_object', 'storing_prep_object', 'waiting_order'],

            #['go_local_preprocessing', 'waiting_order', 'local_preprocessing'],
            #['done_local_preprocessing', 'local_preprocessing', 'waiting_order'],

            #['go_storing_encrypt', 'waiting_order', 'storing_encrypt'],
            #['done_storing_encrypt', 'storing_encrypt', 'waiting_order'],

            ['go_sending_encr_data', 'waiting_order', 'sending_encr_data'],
            ['done_sending_encr_data', 'sending_encr_data', 'waiting_order'],
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
            self.terminate = True

        if packet['action'] == 'store_prep':
            self.FSMworker.go_storing_prep_object(self, packet)
            self.FSMworker.done_storing_prep_object(self)

        if packet['action'] == 'do_local_prep':
            self.FSMworker.go_local_preprocessing(self)
            self.FSMworker.done_local_preprocessing(self)

        if packet['action'] == 'ask_encr_data':
            #self.encrypter = packet['data']['encrypter']
            #self.display('Worker stored encrypter')
            self.FSMworker.go_sending_encr_data(self, packet)
            self.FSMworker.done_sending_encr_data(self)

        '''
        if packet['action'] == 'ask_encrypted_data':
            self.FSMworker.go_sending_encr_data(self)
            self.FSMworker.done_sending_encr_data(self)
        '''
        return self.terminate


class POM4_CommonML_Crypto(Common_to_all_POMs):
    '''
    This class implements ML operations common to POM4 algorithms, run at Crypto node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.

    '''

    def __init__(self, cryptonode_address, master_address, model_type, comms, logger, verbose=False, cr=None):
        """
        Create a :class:`POM4_CommonML_Crypto` instance.

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

   
        **kwargs: Arbitrary keyword arguments.

        """
        self.pom = 4
        self.master_address = master_address
        self.cryptonode_address = cryptonode_address                    # The id of this Worker
        self.comms = comms                      # The comms library
        self.model_type = model_type
        self.cr = cr
        self.encrypter = self.cr.get_encrypter()  # to be shared
        self.decrypter = self.cr.get_decrypter()  # to be kept as secret

        self.logger = logger                    # logger
        self.name = 'POM4_CommonML_Crypto'           # Name
        self.verbose = verbose                  # print on screen when true
        self.create_FSM_crypto()

    def create_FSM_crypto(self):
        """
        Creates a Finite State Machine to be run at the Worker Node

        Parameters
        ----------
        None
        """
        self.display(self.name + ': creating FSM')

        class FSM_crypto(object):

            # Enter/exit callbacks are defined here

            def while_waiting_order(self, MLmodel):
                try:
                    MLmodel.display('Crypto is waiting...')
                except:
                    print('ERROR AT while_waiting_order')
                    import code
                    code.interact(local=locals())
                return

            '''
            def while_storing_prep_object(self, MLmodel, packet):
                try:
                    MLmodel.prep = packet['data']['prep_object']
                    MLmodel.display(MLmodel.name + ' %s: stored preprocessing object' % (str(MLmodel.worker_address)))
                    action = 'ACK_stored_prep'
                    packet = {'action': action, 'sender': MLmodel.cryptonode_address}
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
                    packet = {'action': action, 'sender': MLmodel.cryptonode_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep' % (str(MLmodel.worker_address)))
                except:
                    print('STOP AT while_local_preprocessing')
                    import code
                    code.interact(local=locals())
                return
            '''
            def while_sending_encrypter(self, MLmodel, packet):
                try:

                    '''
                    key_size = packet['data']['key_size']
                    crypt_library = packet['data']['crypt_library']
                    if crypt_library == 'phe':
                        from MMLL.crypto.crypt_PHE import Crypto as CR
                        MLmodel.cr = CR(key_size=key_size)
                        MLmodel.encrypter = MLmodel.cr.encrypter
                        MLmodel.decrypter = MLmodel.cr.decrypter
                    '''
                    # sending encrypter to workers and Master
                    action = 'ACK_sent_encrypter'
                    ############################################
                    # WARNING, remove decrypter, only for testing
                    print('##### WARNING, sending decrypter, only for testing ####')
                    ##########################################
                    data = {'decrypter': MLmodel.decrypter, 'encrypter': MLmodel.encrypter, 'sender': MLmodel.cryptonode_address}
                    packet = {'action': action, 'data': data, 'to': 'CommonML'}
                    # Sending encrypter to Master
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    
                    # Sending params to Master
                    #MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_encrypter')
                except:
                    print('ERROR AT while_sending_encrypter')
                    import code
                    code.interact(local=locals())
                return

            def while_storing_Xy_bl(self, MLmodel, packet):
                try:

                    MLmodel.X_bl_dict = {}
                    MLmodel.y_bl_dict = {}
                    keys =  list(packet['data']['X_bl_dict'].keys())
                    for waddr in keys:
                        MLmodel.X_bl_dict.update({waddr: MLmodel.decrypter.decrypt(packet['data']['X_bl_dict'][waddr])})   
                        MLmodel.y_bl_dict.update({waddr: MLmodel.decrypter.decrypt(packet['data']['y_bl_dict'][waddr])})   
                        MLmodel.display('Decrypting blinded data from %s OK' % waddr)
                    MLmodel.display(MLmodel.name + ': stored decrypted blinded data')

                except:
                    print('ERROR AT while_storing_Xy_bl')
                    import code
                    code.interact(local=locals())
                return

            def while_multiplying_XB(self, MLmodel, packet):

                try:
                    # Result in:
                    XB_bl_encr_dict = {}
                    MLmodel.display(MLmodel.name + ' is multiplying...')
                    # Warning, Bq_prodpk_bl can be a value or a dictionary...
                    B_bl_encr = packet['data']['B_bl']
                    is_dictionary = type(B_bl_encr) is dict

                    if is_dictionary:
                        MLmodel.workers_addresses = list(B_bl_encr.keys())

                    if not is_dictionary:
                        # Same value of B for all X
                        B_bl = MLmodel.decrypter.decrypt(B_bl_encr)
                        MQ, NQ = B_bl.shape

                    for waddr in MLmodel.workers_addresses:
                        if is_dictionary:
                            # Different values of B for every X
                            ### Overflow???
                            try:
                                B_bl = MLmodel.decrypter.decrypt(B_bl_encr[waddr])
                            except:
                                print('STOP AT while_multiplying_XB  overflow at crypto???--------')
                                import code
                                code.interact(local=locals())
                                pass
                            MQ, NQ = B_bl.shape

                        X_bl = MLmodel.X_bl_dict[waddr]

                        MX, NX = X_bl.shape
                        if (MX == MQ and NQ == 1) or (MX == MQ and NQ == NX):
                            # B is of size MP, e.g., errors
                            XB_bl = X_bl * B_bl
                           
                        if (NX == NQ and MQ == 1):
                            # B is of size 1xNI, e.g., weights
                            XB_bl = B_bl * X_bl

                        XB_bl_encr = MLmodel.encrypter.encrypt(XB_bl)
                        XB_bl_encr_dict.update({waddr: XB_bl_encr})
                        
                    action = 'ACK_sent_XB_bl_encr_dict'
                    data = {'XB_bl_encr_dict': XB_bl_encr_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_XB_bl_encr_dict')

                except:
                    print('ERROR AT while_multiplying_XB ############')
                    import code
                    code.interact(local=locals())
                return

            def while_decrypting_model(self, MLmodel, packet):
                try:
                    model_encr_bl = packet['data']['model_bl']
                    model_decr_bl = {}
                    for key in list(model_encr_bl.keys()):
                        model_decr_bl.update({key: MLmodel.cr.decrypter.decrypt(model_encr_bl[key])})

                    action = 'ACK_sent_decr_bl_model'
                    data = {'model_decr_bl': model_decr_bl}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_decr_bl_model')
                except:
                    print('ERROR AT while_decr_model pom4commonml crypto')
                    import code
                    code.interact(local=locals())
                return

            def while_compute_exp(self, MLmodel, packet):
                try:
                    s_encr_bl_dict = packet['data']['s_encr_bl_dict']
                    exps_bl_dict = {}
                    for waddr in s_encr_bl_dict.keys():
                        s_bl = MLmodel.decrypter.decrypt(s_encr_bl_dict[waddr])
                        exp_s_bl = np.exp(-s_bl)
                        exp_s_bl_encr = MLmodel.encrypter.encrypt(exp_s_bl)
                        exps_bl_dict.update({waddr: exp_s_bl_encr})

                    action = 'ACK_exp_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'exps_bl_dict': exps_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    print('ERROR AT Common crypto while_compute_exp')
                    import code
                    code.interact(local=locals())
                    pass
                return

            def while_compute_div(self, MLmodel, packet):
                try:
                    num_bl_dict = packet['data']['num_bl_dict']
                    den_bl_dict = packet['data']['den_bl_dict']

                    sigm_encr_bl_dict = {}
                    for waddr in den_bl_dict.keys():
                        num_bl = MLmodel.decrypter.decrypt(num_bl_dict[waddr])
                        den_bl = MLmodel.decrypter.decrypt(den_bl_dict[waddr])
                        sigm_bl = num_bl / den_bl
                        sigm_encr_bl = MLmodel.encrypter.encrypt(sigm_bl)
                        sigm_encr_bl_dict.update({waddr: sigm_encr_bl})

                    action = 'ACK_div_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'sigm_encr_bl_dict': sigm_encr_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    print('ERROR AT Common crypto while_compute_div')
                    import code
                    code.interact(local=locals())
                    pass
                return

            def while_compute_argmin(self, MLmodel, packet):
                try:
                    c2_2XTC_bl_dict = packet['data']['c2_2XTC_bl_dict']
                    argmin_dict = {}
                    for waddr in c2_2XTC_bl_dict.keys():
                        distXC_bl = MLmodel.decrypter.decrypt(c2_2XTC_bl_dict[waddr])
                        argmin_dict.update({waddr: np.argmin(distXC_bl, axis=1)})

                    action = 'ACK_compute_argmin'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'argmin_dict': argmin_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    print('ERROR AT Common crypto while_compute_argmin')
                    import code
                    code.interact(local=locals())
                    pass
                return


        states_crypto = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='sending_encrypter', on_enter=['while_sending_encrypter']),
            State(name='storing_Xy_bl', on_enter=['while_storing_Xy_bl']),
            State(name='multiplying_XB', on_enter=['while_multiplying_XB']),
            State(name='decrypting_model', on_enter=['while_decrypting_model']),
            State(name='compute_exp', on_enter=['while_compute_exp']),
            State(name='compute_div', on_enter=['while_compute_div']),
            State(name='compute_argmin', on_enter=['while_compute_argmin']),
            'Exit']

        transitions_crypto = [
            ['go_exit', 'waiting_order', 'Exit'],
            ['go_sending_encrypter', 'waiting_order', 'sending_encrypter'],
            ['go_waiting_order', 'sending_encrypter', 'waiting_order'],
            ['go_storing_Xy_bl', 'waiting_order', 'storing_Xy_bl'],
            ['go_waiting_order', 'storing_Xy_bl', 'waiting_order'],
            ['go_multiplying_XB', 'waiting_order', 'multiplying_XB'],
            ['done_multiplying_XB', 'multiplying_XB', 'waiting_order'],
            ['go_decrypting_model', 'waiting_order', 'decrypting_model'],
            ['done_decrypting_model', 'decrypting_model', 'waiting_order'],
            ['go_compute_exp', 'waiting_order', 'compute_exp'],
            ['done_compute_exp', 'compute_exp', 'waiting_order'],
            ['go_compute_div', 'waiting_order', 'compute_div'],
            ['done_compute_div', 'compute_div', 'waiting_order'],
            ['go_compute_argmin', 'waiting_order', 'compute_argmin'],
            ['done_compute_argmin', 'compute_argmin', 'waiting_order']
        ]

        self.FSMcrypto = FSM_crypto()
        self.grafmachine_crypto = GraphMachine(model=self.FSMcrypto,
            states=states_crypto,
            transitions=transitions_crypto,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of Common Cryptonode",
            show_conditions=False)
        return

    def ProcessReceivedPacket_Crypto(self, packet, sender):
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
            self.display(self.name + ' %s: terminated by Master' % (str(self.cryptonode_address)))
            self.terminate = True

        if packet['action'] == 'store_prep':
            self.FSMcrypto.go_storing_prep_object(self, packet)
            self.FSMcrypto.done_storing_prep_object(self)

        if packet['action'] == 'do_local_prep':
            self.FSMcrypto.go_local_preprocessing(self)
            self.FSMcrypto.done_local_preprocessing(self)

        if packet['action'] == 'ask_encrypter':
            self.FSMcrypto.go_sending_encrypter(self, packet)
            self.FSMcrypto.go_waiting_order(self)

        if packet['action'] == 'send_Xy_bl':
            self.FSMcrypto.go_storing_Xy_bl(self, packet)
            self.FSMcrypto.go_waiting_order(self)

        if packet['action'] == 'send_mult_XB':
            self.FSMcrypto.go_multiplying_XB(self, packet)
            self.FSMcrypto.done_multiplying_XB(self)

        if packet['action'] == 'send_model_encr_bl':
            self.FSMcrypto.go_decrypting_model(self, packet)
            self.FSMcrypto.done_decrypting_model(self)

        if packet['action'] == 'ask_exp_bl':
            self.FSMcrypto.go_compute_exp(self, packet)
            self.FSMcrypto.done_compute_exp(self)

        if packet['action'] == 'ask_div_bl':
            self.FSMcrypto.go_compute_div(self, packet)
            self.FSMcrypto.done_compute_div(self)

        if packet['action'] == 'ask_argmin_bl':
            self.FSMcrypto.go_compute_argmin(self, packet)
            self.FSMcrypto.done_compute_argmin(self)

        return self.terminate

