# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM6 

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "May 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
#import pickle
from transitions import State
from transitions.extensions import GraphMachine
import time

class POM6_CommonML_Master(Common_to_all_POMs):
    """
    This class implements ML operations common to POM6 algorithms, run at Master node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, comms, logger, verbose=False, **kwargs):
        """
        Create a :class:`POM6_CommonML_Master` instance.

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
        self.name = 'POM6_CommonML_Master'               # Name
        self.master_address = master_address

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
        self.Nworkers = len(workers_addresses)                    # Nworkers
        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.state_dict = None                      # State of the main script
        self.verbose = verbose                      # print on screen when true
        self.classes = None
        self.process_kwargs(kwargs)

        self.state_dict = {}                        # dictionary storing the execution state
        #self.NC = NC                                # No. Centroids
        self.NI = None
        #self.classes = classes
        self.Npc_dict = {}
        self.sumX_dict = {}
        self.sumy_dict = {}
        self.NP_dict = {}
        for addr in self.workers_addresses:
            self.state_dict.update({addr: ''})
        self.create_FSM_master()
        self.message_counter = 0    # used to number the messages
        
        self.worker_names = {} # dictionary with the mappings worker_id -> pseudo_id
        self.features_info = {}

    def create_FSM_master(self):
        """
        Creates a Finite State Machine to be run at the Master Node

        Parameters
        ----------
        None
        """
        self.display(self.name + ': creating Common ML FSM, POM6')

        '''
        with open('../MMLL/models/POM6/CommonML/POM6_CommonML_FSM_master.pkl', 'rb') as f:
            [states_master, transitions_master] = pickle.load(f)
        '''

        states_master = [
            'waiting_order',
            State(name='computing_R', on_enter=['while_computing_R']),
            State(name='getting_R_DT', on_enter=['while_getting_R_DT']),
            State(name='getting_R_SSP', on_enter=['while_getting_R_SSP']),
            State(name='sending_prep_object', on_enter=['while_sending_prep_object']),
            State(name='asking_local_prep', on_enter=['while_asking_local_prep']),
            State(name='sending_roundrobin', on_enter=['while_sending_roundrobin']),
            State(name='terminating_workers', on_enter=['while_terminating_workers']),
            State(name='getting_Npc', on_enter=['while_getting_Npc']),
            State(name='getting_sumXy', on_enter=['while_getting_sumXy']),
            State(name='Exit', on_enter=['while_Exit'])
        ]

        transitions_master = [
            
            ['go_computing_R', 'waiting_order', 'computing_R'],
            ['go_getting_R_DT', 'computing_R', 'getting_R_DT'],
            ['go_getting_R_SSP', 'computing_R', 'getting_R_SSP'],
            ['go_waiting_order', 'getting_R_DT', 'waiting_order'],
            ['go_waiting_order', 'getting_R_SSP', 'waiting_order'],

            ['go_sending_prep_object', 'waiting_order', 'sending_prep_object'],
            ['go_asking_local_prep', 'sending_prep_object', 'asking_local_prep'],
            ['go_waiting_order', 'asking_local_prep', 'waiting_order'],

            ['go_sending_roundrobin', 'waiting_order', 'sending_roundrobin'],
            ['go_waiting_order', 'sending_roundrobin', 'waiting_order'],

            ['go_terminating_workers', 'waiting_order', 'terminating_workers'],
            ['go_waiting_order', 'terminating_workers', 'waiting_order'],

            ['go_getting_Npc', 'waiting_order', 'getting_Npc'],
            ['go_waiting_order', 'getting_Npc', 'waiting_order'],

            ['go_getting_sumXy', 'waiting_order', 'getting_sumXy'],
            ['go_waiting_order', 'getting_sumXy', 'waiting_order'],

            ['go_Exit', 'waiting_order', 'Exit'],
            ['go_waiting_order', 'Exit', 'waiting_order']
        ]


        class FSM_master(object):

            def while_Exit(self, MLmodel):
                #print(MLmodel.name + 'while_Exit')
                return
          
            def while_computing_R(self, MLmodel):
                packet = {'action': 'compute_Rr', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' sent compute_Rr to all Workers')
                return

            def while_getting_R_DT(self, MLmodel):              
                MLmodel.R_central_decr = np.zeros((MLmodel.NI + 1, MLmodel.NI + 1))
                MLmodel.r_central_decr = np.zeros((MLmodel.NI + 1, 1))
                packet = {'action': 'get_Rr_DT', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' sent get_Rr_DT to all Workers')
                return

            def while_sending_prep_object(self, MLmodel):
                action = 'store_prep'
                data = {'prep_object': MLmodel.prep, 'to': 'CommonML', 'sender': MLmodel.master_address}
                packet = {'action': action, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' sent preprocessing object to all Workers')
                return

            def while_asking_local_prep(self, MLmodel):
                packet = {'action': 'do_local_prep', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' sent do_local_prep to all Workers')
                return

            def while_getting_Npc(self, MLmodel):
                packet = {'action': 'get_Npc', 'to': 'CommonML', 'classes': MLmodel.classes, 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' broadcasted get_Npc to all Workers')
                return

            def while_getting_sumXy(self, MLmodel):
                packet = {'action': 'get_sumXy', 'to': 'CommonML', 'classes': MLmodel.classes, 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' broadcasted get_sumXy to all Workers')
                return

            def while_sending_roundrobin(self, MLmodel, roundrobin_addresses, action, NI=None, xmean=None):
                if action == 'count_patterns':
                    self.N_random_ini = np.random.randint(1000000)
                    destiny_adress = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': action, 'N': self.N_random_ini, 'remain_addresses': remain_addresses, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(destiny_adress, packet)

                if action == 'sum_patterns':
                    self.X_sum_ini = np.random.randint(0, 10000, (NI, 1)).ravel()
                    destiny_adress = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': action, 'X_sum': self.X_sum_ini, 'remain_addresses': remain_addresses, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(destiny_adress, packet)

                if action == 'squared_sum_patterns':
                    self.Xsquared_sum_ini = np.random.randint(0, 10000, (NI, 1)).ravel()
                    destiny_adress = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': action, 'to': 'CommonML', 'Xsquared_sum': self.Xsquared_sum_ini, 'remain_addresses': remain_addresses, 'xmean': xmean, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(destiny_adress, packet)
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
        workers_addresses_terminate: list of strings
            addresses of the workers to be terminated

        """

        message_id = self.master_address + str(self.message_counter)
        self.message_counter += 1            
        packet = {'action': 'STOP', 'to': 'CommonML', 'sender': self.master_address, 'message_id': message_id}

        if workers_addresses_terminate is None:  # We terminate all of them
            workers_addresses_terminate = self.workers_addresses
            self.display(self.name + ' sent STOP to all Workers')
        else:
            self.display(self.name + ' sent STOP to %d Workers' % len(workers_addresses_terminate))

        self.comms.broadcast(packet, workers_addresses_terminate)

        # Updating the list of active users
        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)

        self.FSMmaster.go_Exit(self)
        self.FSMmaster.go_waiting_order(self)

    def get_R_r_Master(self):
        """
        Obtaining R and r from workers

        Parameters
        ----------
            None
        """
        self.FSMmaster.go_computing_R(self)
        self.display(self.name + ': Asking workers to compute R, r')
        self.run_Master()
        self.display(self.name + ': compute R, r is done')
        return self.R_dict, self.r_dict

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
        self.display(self.name + ': Resetting local data')

    def get_Npc(self):
        """
        Obtain the number of patterns per class, to balance uneven pattern distribution among classes

        Parameters
        ----------
        None
        """
        self.display(self.name + ': Asking workers their Npc')
        self.FSMmaster.go_getting_Npc(self)
        self.run_Master()

        self.aggregated_Npc_dict = {} # Number of aggregated patterns per class
        for cla in self.classes:
            count = 0
            for wa in self.workers_addresses:
                count += self.Npc_dict[wa][cla]
            self.aggregated_Npc_dict.update({cla: count})   

        self.display(self.name + ': getting Npc is done')

    def get_sumXy(self):
        """
        Gets from workers their sum of input data multiplied by the targets

        Parameters
        ----------
        None
        """

        self.display(self.name + ': Asking workers their sumX sumy')
        self.FSMmaster.go_getting_sumXy(self)
        self.run_Master()

        self.total_sumX = np.zeros((1, self.NI))
        self.total_sumy = 0
        self.total_NP = 0
        for waddr in self.workers_addresses:
            self.total_sumX += self.sumX_dict[waddr]
            self.total_sumy += self.sumy_dict[waddr]
            self.total_NP += self.NP_dict[waddr]

        self.display(self.name + ': getting sumXy is done')
        
    def local_prep_Master(self, prep_object):
        """
        This is the local preprocessing loop, it runs the following actions:
            - It sends the preprocessing object to the workers 
            - It sends instruction to the workers to preprocess the data

        Parameters
        ----------
            None
        """
        self.prep = prep_object
        self.FSMmaster.go_sending_prep_object(self)
        self.display(self.name + ' : Sending Preprocessing object')
        self.run_Master()
        self.display(self.name + ' : Local Preprocessing is done')

    def start_roundrobin(self, roundrobin_addresses, action, xmean=None):
        """
        Start the roundrobin (ring) protocol

        Parameters
        ----------
            roundrobin_addresses: list of addresses
                Addresses to be used in the ring protocol

            action: string
                Type of action to be executed

            xmean: ndarray
                1-D numpy array containing the mean values

        """

        if action == 'count_patterns':
            self.FSMmaster.go_sending_roundrobin(self, roundrobin_addresses, action)

        if action == 'sum_patterns':
            self.FSMmaster.go_sending_roundrobin(self, roundrobin_addresses, action, self.NI + 1)

        if action == 'squared_sum_patterns':
            self.FSMmaster.go_sending_roundrobin(self, roundrobin_addresses, action, self.NI + 1, xmean)
        self.run_Master()
        pass

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """

        # Checking ACK from all nodes
        if self.chekAllStates('ACK_store_World_PHE'):
            self.FSMmaster.go_sending_w_encr(self)

        if self.chekAllStates('ACK_compute_Rr'):
            self.FSMmaster.go_getting_R_DT(self)

        if self.chekAllStates('ACK_get_Rr_DT'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_stored_w_encr'):
            self.FSMmaster.go_getting_predictions(self)

        if self.chekAllStates('ACK_encr_errors'):
            self.FSMmaster.go_computing_MSE(self)

        if self.chekAllStates('ACK_stored_prep'):
            self.FSMmaster.go_asking_local_prep(self)
        """
        if self.chekAllStates('ACK_encr_preds'):
            self.FSMmaster.go_sending_sorting(self)
        """
        if self.chekAllStates('ACK_sending_qysv'):
            self.FSMmaster.go_sending_sorting(self)

        if self.chekAllStates('ACK_local_prep'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_computing_AUCs'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_Npc'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_sumXy'):
            self.FSMmaster.go_waiting_order(self)


    def ProcessReceivedPacket_Master(self, packet, sender):
        """
        Process the received packet at Master and take some actions, possibly changing the state

        Parameters
        ----------
            packet: packet object 
                packet received (usually a dict with various content)

            sender: string
                id of the sender 0-N
        """
        try:
            if packet['action'][0:3] == 'ACK':
                self.display(self.name + ' received ACK from %s: %s' % (str(sender), packet['action']))
                self.state_dict[sender] = packet['action']

            if packet['action'] == 'ACK_get_Rr_DT':
                self.R_central_decr += packet['data']['R'].reshape((self.NI + 1, self.NI + 1))
                self.r_central_decr += packet['data']['r'].reshape((self.NI + 1, 1))
                self.R_dict.update({sender: packet['data']['R'].reshape((self.NI + 1, self.NI + 1))})
                self.r_dict.update({sender: packet['data']['r'].reshape((self.NI + 1, 1))})

            if packet['action'] == 'ACK_encr_preds':
                self.preds_dict.update({sender: packet['data']['preds_encr']})

            if packet['action'] == 'count_patterns':
                self.Ncount = packet['N']
                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'sum_patterns':
                self.X_sum = packet['X_sum']
                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'squared_sum_patterns':
                self.Xsquared_sum = packet['Xsquared_sum']
                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'ACK_computing_AUCs':
                self.AUCs_dict.update({sender: packet['data']['aucs'][0]})

            if packet['action'] == 'ACK_send_Npc':
                self.Npc_dict.update({sender: packet['npc_dict']})

            if packet['action'] == 'ACK_send_sumXy':
                self.sumX_dict.update({sender: packet['data']['X_sum']})
                self.sumy_dict.update({sender: packet['data']['y_sum']})
                self.NP_dict.update({sender: packet['data']['NP']})

        except:
            print('ERROR AT ProcessReceivedPacket_Master')
            import code
            code.interact(local=locals())
            pass

        return


#===============================================================
#                 Worker   
#===============================================================

class POM6_CommonML_Worker(Common_to_all_POMs):
    '''
    Class implementing the Common operations under POM6, run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`CommonML_Worker` instance.

        Parameters
        ----------
        master_address: string
            address of the master node

        worker_address: string
            id of this worker

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

        verbose: boolean
            indicates if messages are print or not on screen
        """
        self.master_address = master_address
        self.worker_address = worker_address                    # The id of this Worker
        self.model_type = model_type
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.name = 'POM6_CommonML_Worker'           # Name
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
                MLmodel.display(MLmodel.name + ' %s is waiting...' % (str(MLmodel.worker_address)))
                return

            def while_computing_R(self, MLmodel):
                print('STOP AT while_computing_R worker')
                import code
                code.interact(local=locals())
                MLmodel.R = np.dot(MLmodel.Xtr_b.T, MLmodel.Xtr_b)
                MLmodel.r = np.dot(MLmodel.Xtr_b.T, MLmodel.ytr)
                MLmodel.display(self.name + ' %s: computed R and r' % (str(MLmodel.worker_address)))
                action = 'ACK_compute_Rr'
                packet = {'action': action, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_compute_Rr' % (str(MLmodel.worker_address)))

            def while_sending_R_DT(self, MLmodel):
                print('STOP AT while_sending_R_DT worker')
                import code
                code.interact(local=locals())
                MLmodel.display(MLmodel.name + ' %s: sending R to Master, direct transmission' % (str(MLmodel.worker_address)))
                action = 'ACK_get_Rr_DT'
                data = {'R': MLmodel.R, 'r': MLmodel.r}
                packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display('Worker %s: sent ACK_get_Rr_DT' % (str(MLmodel.worker_address)))

            def while_storing_prep_object(self, MLmodel, packet):
                print('STOP AT while_storing_prep_object worker')
                import code
                code.interact(local=locals())
                MLmodel.prep = packet['data']['prep_object']
                MLmodel.display(MLmodel.name + ' %s: stored preprocessing object' % (str(MLmodel.worker_address)))
                action = 'ACK_stored_prep'
                packet = {'action': action, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(self.name + ' %s: sent ACK_stored_prep' % (str(MLmodel.worker_address)))
                return

            def while_local_preprocessing(self, MLmodel):
                print('STOP AT while_local_preprocessing worker')
                import code
                code.interact(local=locals())
                X = np.copy(MLmodel.Xtr_b)
                new_Xtr_b = MLmodel.prep.transform(X)
                MLmodel.Xtr_b = np.copy(new_Xtr_b)
                MLmodel.display(MLmodel.name + ' %s: locally preprocessing data...' % (str(MLmodel.worker_address)))
                action = 'ACK_local_prep'
                packet = {'action': action, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep' % (str(MLmodel.worker_address)))
                return

            def while_computing_roundrobin(self, MLmodel, packet):
                print('STOP AT while_computing_roundrobin worker')
                import code
                code.interact(local=locals())
                if packet['action'] == 'count_patterns':
                    NPtr = MLmodel.Xtr_b.shape[0]
                    N = packet['N'] + NPtr
                    roundrobin_addresses = packet['remain_addresses']
                    destiny_address = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': packet['action'], 'N': N, 'remain_addresses': remain_addresses, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, destiny_address)

                if packet['action'] == 'sum_patterns':
                    x_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()
                    X_sum = packet['X_sum'] + x_sum

                    roundrobin_addresses = packet['remain_addresses']
                    destiny_address = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': packet['action'], 'X_sum': X_sum, 'remain_addresses': remain_addresses, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, destiny_address)

                if packet['action'] == 'squared_sum_patterns':
                    xmean = packet['xmean']
                    X_mean = MLmodel.Xtr_b - xmean
                    X_squared = np.power(X_mean, 2)
                    X_squared_sum = np.sum(X_squared, axis=0).ravel()
                    Xsquared_sum = packet['Xsquared_sum'] + X_squared_sum
                    roundrobin_addresses = packet['remain_addresses']
                    destiny_address = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': packet['action'], 'Xsquared_sum': Xsquared_sum, 'remain_addresses': remain_addresses, 'xmean': xmean, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, destiny_address)

            def while_computing_Npc(self, MLmodel, packet):
                npc_dict = {}
                ytr_str = MLmodel.ytr
                ytr_str = [str(int(y)) for y in ytr_str]
                for cla in packet['classes']:
                    npc_dict.update({cla: ytr_str.count(cla)})
                packet = {'action': 'ACK_send_Npc', 'npc_dict': npc_dict, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_Npc' % (str(MLmodel.worker_address)))
                return

            def while_computing_sumXy(self, MLmodel, packet):
                X_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()
                y_sum = np.sum(MLmodel.ytr, axis=0).ravel()
                NP = MLmodel.Xtr_b.shape[0]  
                data = {'X_sum':X_sum , 'y_sum':y_sum, 'NP':NP}
                packet = {'action': 'ACK_send_sumXy', 'sender': MLmodel.worker_address, 'data':data}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_sumXy' % (str(MLmodel.worker_address)))
                return
        '''
        with open('../MMLL/models/POM6/CommonML/POM6_CommonML_FSM_worker.pkl', 'rb') as f:
            [states_worker, transitions_worker] = pickle.load(f)
        '''

        states_worker = [  
            'Exit',
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='computing_R', on_enter=['while_computing_R']),
            State(name='sending_R_DT', on_enter=['while_sending_R_DT']),
            State(name='storing_prep_object', on_enter=['while_storing_prep_object']),
            State(name='local_preprocessing', on_enter=['while_local_preprocessing']),
            State(name='computing_roundrobin', on_enter=['while_computing_roundrobin']),
            State(name='computing_Npc', on_enter=['while_computing_Npc']),
            State(name='computing_sumXy', on_enter=['while_computing_sumXy'])
            ]
            
        transitions_worker = [
            ['go_Exit', 'waiting_order', 'Exit'],

            ['go_computing_R', 'waiting_order', 'computing_R'],
            ['done_computing_R', 'computing_R', 'waiting_order'],

            ['go_sending_R_DT', 'waiting_order', 'sending_R_DT'],
            ['done_sending_R_DT', 'sending_R_DT', 'waiting_order'],

            ['go_storing_prep_object', 'waiting_order', 'storing_prep_object'],
            ['done_storing_prep_object', 'storing_prep_object', 'waiting_order'],

            ['go_local_preprocessing', 'waiting_order', 'local_preprocessing'],
            ['done_local_preprocessing', 'local_preprocessing', 'waiting_order'],

            ['go_computing_roundrobin', 'waiting_order', 'computing_roundrobin'],
            ['done_computing_roundrobin', 'computing_roundrobin', 'waiting_order'],

            ['go_computing_Npc', 'waiting_order', 'computing_Npc'],
            ['done_computing_Npc', 'computing_Npc', 'waiting_order'],

            ['go_computing_sumXy', 'waiting_order', 'computing_sumXy'],
            ['done_computing_sumXy', 'computing_sumXy', 'waiting_order']

            ]

        
        self.FSMworker = FSM_worker()
        self.grafmachine_worker = GraphMachine(model=self.FSMworker,
            states=states_worker,
            transitions=transitions_worker,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of worker No. %s, POM6" % str(self.worker_address),
            show_conditions=False)
        return

    # This is specific to CommonML
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

        if packet['action'] == 'sending_px':
            print('OK')
            self.FSMworker.go_sending_qysv(self, packet)
            self.FSMworker.done_sending_qysv(self)

        # Compute local R and send back ACK to Master
        if packet['action'] == 'compute_Rr':
            self.FSMworker.go_computing_R(self)
            self.FSMworker.done_computing_R(self)

        # Send R and to Master (DT)
        if packet['action'] == 'get_Rr_DT':
            self.FSMworker.go_sending_R_DT(self)
            self.FSMworker.done_sending_R_DT(self)

        if packet['action'] == 'store_w_encr':
            self.FSMworker.go_storing_w_encr(self, packet)
            self.FSMworker.done_storing_w_encr(self)

        if packet['action'] == 'get_predictions':
            self.FSMworker.go_computing_predictions(self)
            self.FSMworker.done_computing_predictions(self)

        if packet['action'] == 'send_ordering':
            self.FSMworker.go_computing_AUCs(self, packet)
            self.FSMworker.done_computing_AUCs(self)

        if packet['action'] == 'store_prep':
            self.FSMworker.go_storing_prep_object(self, packet)
            self.FSMworker.done_storing_prep_object(self)

        if packet['action'] == 'do_local_prep':
            self.FSMworker.go_local_preprocessing(self)
            self.FSMworker.done_local_preprocessing(self)

        if packet['action'] == 'count_patterns':
            self.FSMworker.go_computing_roundrobin(self, packet)
            self.FSMworker.done_computing_roundrobin(self)

        if packet['action'] == 'sum_patterns':
            self.FSMworker.go_computing_roundrobin(self, packet)
            self.FSMworker.done_computing_roundrobin(self)

        if packet['action'] == 'squared_sum_patterns':
            self.FSMworker.go_computing_roundrobin(self, packet)
            self.FSMworker.done_computing_roundrobin(self)

        if packet['action'] == 'get_Npc':
            self.FSMworker.go_computing_Npc(self, packet)
            self.FSMworker.done_computing_Npc(self)

        if packet['action'] == 'get_sumXy':           
            self.FSMworker.go_computing_sumXy(self, packet)
            self.FSMworker.done_computing_sumXy(self)

        return self.terminate
