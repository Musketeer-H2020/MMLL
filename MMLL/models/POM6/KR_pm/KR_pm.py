# -*- coding: utf-8 -*-
'''
Kernel Regression (public model) under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Apr. 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
#from pympler import asizeof #asizeof.asizeof(my_object)
import pickle

class model():
    def __init__(self):
        self.C = None
        self.w = None

    def predict(self, X_b):
        """
        Predicts outputs given the inputs

        Parameters
        ----------
        X_b: ndarray
            Matrix with the input values

        Returns
        -------
        prediction_values: ndarray

        """


        X = X_b
        NP = X.shape[0]
        NC = self.C.shape[0]
        XC2 = -2 * np.dot(X, self.C.T)
        XC2 += np.sum(np.multiply(X, X), axis=1).reshape((NP, 1))
        XC2 += np.sum(np.multiply(self.C, self.C), axis=1).reshape((1, NC))
        # Gauss
        
        KXC = np.exp(-XC2 / 2.0 /  (self.sigma ** 2))
        #1 ./ ( 1 + ((x).^2 / (2 * sigma ^2 )));
        #KXC = 1 / (1 + (XC2 / 2.0 /  (self.sigma ** 2)  ) ) 
        KXC = np.hstack( (np.ones((NP, 1)), KXC))
        prediction_values = np.dot(KXC, self.w)
        return prediction_values


class KR_pm_Master(Common_to_all_POMs):
    """
    This class implements the Kernel Regression (public model), run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`KR_pm_Master` instance.

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
        super().__init__()
        self.pom = 6
        self.model_type = model_type
        self.name = self.model_type + '_Master'                 # Name
        self.master_address = master_address
        self.workers_addresses = workers_addresses
        
        try:
            kwargs.update(kwargs['model_parameters'])
            del kwargs['model_parameters']
        except Exception as err:
            pass
        self.process_kwargs(kwargs)

        # Convert workers_addresses -> '0', '1', + send_to dict
        self.broadcast_addresses = workers_addresses
        self.Nworkers = len(workers_addresses)                    # Nworkers
        self.workers_addresses = list(range(self.Nworkers))
        self.workers_addresses = [str(x) for x in self.workers_addresses]

        self.state_dict = {}                        # dictionary storing the execution state
        for k in range(0, self.Nworkers):
            self.state_dict.update({self.workers_addresses[k]: ''})

        self.send_to = {}
        self.receive_from = {}
        for k in range(self.Nworkers):
            self.send_to.update({str(k): workers_addresses[k]})
            self.receive_from.update({workers_addresses[k]: str(k)})

        self.logger = logger                        # logger
        self.comms = comms                          # comms lib
        self.verbose = verbose                      # print on screen when true
        self.NI = None
        self.model = model()

        self.create_FSM_master()
        self.FSMmaster.master_address = master_address
        self.message_counter = 0    # used to number the messages
        self.KTK_dict = {}
        self.KTy_dict = {}
        self.NC = self.C.shape[0]
        self.NI = self.C.shape[1]
        self.model.C = self.C
        self.model.sigma = np.sqrt(self.NI) * self.fsigma 
        self.newNI_dict = {}

        '''
        print('#########################################################')
        print(self.broadcast_addresses)
        print(self.workers_addresses)
        print(self.send_to)
        print(self.receive_from)
        print('#########################################################')
        '''

    def create_FSM_master(self):
        """
        Creates a Finite State Machine to be run at the Master Node

        Parameters
        ----------
        None
        """

        self.display(self.name + ': creating FSM')

        '''
        path = '../MMLL/models/POM' + str(self.pom) + '/' + self.model_type + '/' 
        filename = path + 'POM' + str(self.pom) + '_' + self.model_type + '_FSM_master.pkl'
        with open(filename, 'rb') as f:
            [states_master, transitions_master] = pickle.load(f)
        '''

        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='update_tr_data', on_enter=['while_update_tr_data']),
            State(name='getting_KTK', on_enter=['while_getting_KTK']),
            State(name='sending_C', on_enter=['while_sending_C']),
            State(name='updating_w', on_enter=['while_updating_w'])
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_sending_C', 'waiting_order', 'sending_C'],
            ['go_waiting_order', 'sending_C', 'waiting_order'],

            ['go_getting_KTK', 'waiting_order', 'getting_KTK'],
            ['go_updating_w', 'getting_KTK', 'updating_w'],
            ['go_waiting_order', 'updating_w', 'waiting_order']
        ]


        class FSM_master(object):

            self.name = 'FSM_master'

            def while_waiting_order(self, MLmodel):
                MLmodel.display(MLmodel.name + ': WAITING for instructions...')
                return

            def while_update_tr_data(self, MLmodel):
                try:
                    action = 'update_tr_data'
                    data = {}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
                    MLmodel.display(MLmodel.name + ': broadcasted update_tr_data to all Workers')
                except Exception as err:
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    MLmodel.display('ERROR AT while_update_tr_data')
                    import code
                    code.interact(local=locals())
                return

            def while_sending_C(self, MLmodel):
                try:
                    action = 'sending_C'
                    data = {'C': MLmodel.model.C, 'sigma': MLmodel.model.sigma}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
                    MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')

                except Exception as err:
                    print('ERROR AT while_sending_C')
                    import code
                    code.interact(local=locals())         
                return

            def while_getting_KTK(self, MLmodel):
                try:
                    action = 'compute_KTK'
                    data = None
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
                    MLmodel.display(MLmodel.name + ': broadcasted compute_KTK to all Workers')
                except Exception as err:
                    print('ERROR AT while_getting_KTK')
                    import code
                    code.interact(local=locals())         

                return

            def while_updating_w(self, MLmodel):
                try:
                    NC = MLmodel.model.C.shape[0]    
                    MLmodel.KTK_accum = np.zeros((NC + 1, NC + 1))
                    MLmodel.KTy_accum = np.zeros((NC + 1, 1))
                    for waddr in MLmodel.workers_addresses:
                        MLmodel.KTK_accum += MLmodel.KTK_dict[waddr]
                        MLmodel.KTy_accum += MLmodel.KTy_dict[waddr].reshape((NC + 1, 1))

                    MLmodel.model.w = np.dot(np.linalg.inv(MLmodel.KTK_accum + MLmodel.regularization * np.eye(NC + 1)), MLmodel.KTy_accum)        
                except Exception as err:
                    print('ERROR AT while_updating_w')
                    import code
                    code.interact(local=locals())         
                return

            def while_Exit(self, MLmodel):
                #print('while_Exit')
                return

        self.FSMmaster = FSM_master()
        self.grafmachine_master = GraphMachine(model=self.FSMmaster,
            states=states_master,
            transitions=transitions_master,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of the master",
            show_conditions=False)
        return

    def reset(self, NI):
        """
        Create some empty variables needed by the Master Node

        Parameters
        ----------
        NI: integer
            Number of input features
        """
        self.NI = NI
        self.model.w = np.random.normal(0, 0.001, (self.NI + 1, 1))      # weights in plaintext, first value is bias
        self.w_old = np.random.normal(0, 1.0, (self.NI + 1, 1))
        self.XTDaX_accum = np.zeros((self.NI + 1, self.NI + 1))    # Cov. matrix in plaintext
        self.XTDast_accum = np.zeros((self.NI + 1, 1))              # Cov. matrix in plaintext
        self.preds_dict = {}                                           # dictionary storing the prediction errors
        self.XTX_dict = {}
        self.XTy_dict = {}
        self.display(self.name + ': Resetting local data')

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
        self.display(self.name + ': Starting training')

        self.FSMmaster.go_update_tr_data(self)
        self.run_Master()
        # Checking the new NI values
        newNIs = list(set(list(self.newNI_dict.values())))
        if len(newNIs) > 1:
            message = 'ERROR: the training data has different number of features...'
            self.display(message)
            self.display(list(self.newNI_dict.values()))
            raise Exception(message)
        else:
            self.reset(newNIs[0])
            ## Adding bias to validation data, if any
            if self.Xval_b is not None: 
                self.Xval_b = self.add_bias(self.Xval_b).astype(float)
                self.yval = self.yval.astype(float)

        self.FSMmaster.go_sending_C(self)
        self.run_Master()

        self.FSMmaster.go_getting_KTK(self)
        self.run_Master()

        self.display(self.name + ': Training is done')
        self.model.niter = 1


    def predict_Master(self, X_b):
        """
        Predicts outputs given the model and inputs

        Parameters
        ----------
        X_b: ndarray
            Matrix with the input values

        Returns
        -------
        prediction_values: ndarray

        """
        prediction_values = self.model.predict(KXC)
        return prediction_values

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_storing_C'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sending_KTK'):
            self.FSMmaster.go_updating_w(self)
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_update_tr_data'):
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
        try:
            #sender = packet['sender']
            sender = self.receive_from[packet['sender']]

            if packet['action'][0:3] == 'ACK':
                self.display(self.name + ': received ACK from %s: %s' % (str(sender), packet['action']))
                self.state_dict[sender] = packet['action']

            if packet['action'] == 'ACK_sending_KTK':
                self.KTK_dict.update({sender: packet['data']['KTK']})
                self.KTy_dict.update({sender: packet['data']['KTy']})

            if packet['action'] == 'ACK_update_tr_data':
                #print('ProcessReceivedPacket_Master ACK_update_tr_data')
                self.newNI_dict.update({sender: packet['data']['newNI']})

        except Exception as err:
            print('ERROR AT ProcessReceivedPacket_Master')
            import code
            code.interact(local=locals())         

        return


#===============================================================
#                 Worker
#===============================================================
class KR_pm_Worker(Common_to_all_POMs):
    '''
    Class implementing Kernel Regression (public model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`KR_pm_Worker` instance.

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
        """
        self.pom = 6
        self.master_address = master_address
        self.worker_address = worker_address                    # The id of this Worker
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.model_type = model_type
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.name = model_type + '_Worker'    # Name
        self.verbose = verbose                  # print on screen when true
        self.Xtr_b = Xtr_b
        self.ytr = ytr
        self.NPtr = len(ytr)
        self.w = None
        self.create_FSM_worker()
        self.message_id = 0    # used to number the messages

    def create_FSM_worker(self):
        """
        Creates a Finite State Machine to be run at the Worker Node

        Parameters
        ----------
        None
        """
        self.name = 'FSM_worker'

        self.display(self.name + ' %s: creating FSM' % (str(self.worker_address)))

        class FSM_worker(object):

            name = 'FSM_worker'

            def while_waiting_order(self, MLmodel):
                MLmodel.display(MLmodel.name + ' %s: WAITING for instructions...' % (str(MLmodel.worker_address)))
                return

            def while_setting_tr_data(self, MLmodel, packet):
                try:
                    NPtr, newNI = MLmodel.Xtr_b.shape
                    #MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
                    MLmodel.ytr = MLmodel.ytr.astype(float)
                    action = 'ACK_update_tr_data'
                    data = {'newNI': newNI}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_update_tr_data' % (str(MLmodel.worker_address)))
                except Exception as err:
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    #raise
                    import code
                    code.interact(local=locals())
                    #MLmodel.display('ERROR AT while_computing_XTDaX')

            def while_computing_KTK(self, MLmodel):
                try:
                    KTK = np.dot(MLmodel.KXC.T, MLmodel.KXC)
                    KTy = np.dot(MLmodel.KXC.T, MLmodel.ytr)
                    NC = KTK.shape[0] - 1
                    w = np.dot(np.linalg.inv(KTK + 0.1 * np.eye(NC + 1)), KTy)        
                    out = np.dot(MLmodel.KXC, w)

                    action = 'ACK_sending_KTK'
                    data = {'KTK': KTK, 'KTy': KTy}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    #MLmodel.comms.send(MLmodel.master_address, packet)
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_KTK' % (str(MLmodel.worker_address)))
                except Exception as err:
                    print('ERROR AT while_computing_KTK')
                    import code
                    code.interact(local=locals())         
                    pass

            def while_storing_C(self, MLmodel, packet):
                # We store C and compute KXC
                try:
                    MLmodel.C = packet['data']['C']
                    NC = MLmodel.C.shape[0]
                    MLmodel.sigma = packet['data']['sigma']
                    NI = MLmodel.Xtr_b.shape[1]
                    NP = MLmodel.Xtr_b.shape[0]

                    # Check if training data has bias...

                    #MLmodel.sigma = np.sqrt(NI) * MLmodel.fsigma
                    X = MLmodel.Xtr_b
                    XC2 = -2 * np.dot(X, MLmodel.C.T)
                    XC2 += np.sum(np.multiply(X, X), axis=1).reshape((NP, 1))
                    XC2 += np.sum(np.multiply(MLmodel.C, MLmodel.C), axis=1).reshape((1, NC))
                    # Gauss
                    KXC = np.exp(-XC2 / 2.0 /  (MLmodel.sigma ** 2))
                    # Poly
                    #KXC = 1 / (1 + (XC2 / 2.0 /  (MLmodel.sigma ** 2)  ) ) 
                    MLmodel.KXC = np.hstack( (np.ones((NP, 1)), KXC)) # NP x NC + 1

                    action = 'ACK_storing_C'
                    data = {}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_storing_C' % (str(MLmodel.worker_address)))
                    
                except Exception as err:
                    print('ERROR AT while_storing_C')
                    import code
                    code.interact(local=locals())         

                return

        '''
        path = '../MMLL/models/POM' + str(self.pom) + '/' + self.model_type + '/' 
        filename = path + 'POM' + str(self.pom) + '_' + self.model_type + '_FSM_worker.pkl'
        with open(filename, 'rb') as f:
            [states_worker, transitions_worker] = pickle.load(f)
        '''

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='storing_C', on_enter=['while_storing_C']),
            State(name='computing_KTK', on_enter=['while_computing_KTK']),
            State(name='computing_KXC', on_enter=['while_computing_KXC']),
            State(name='Exit', on_enter=['while_Exit'])
        ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_storing_C', 'waiting_order', 'storing_C'],
            ['done_storing_C', 'storing_C', 'waiting_order'],

            ['go_computing_KXC', 'waiting_order', 'computing_KXC'],
            ['done_computing_KXC', 'computing_KXC', 'waiting_order'],

            ['go_computing_KTK', 'waiting_order', 'computing_KTK'],
            ['done_computing_KTK', 'computing_KTK', 'waiting_order'],

            ['go_exit', 'waiting_order', 'Exit']
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
        try:
            # Exit the process
            if packet['action'] == 'STOP':
                self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
                self.terminate = True

            if packet['action'] == 'update_tr_data':
                # We update the training data
                self.FSMworker.go_setting_tr_data(self, packet)
                self.FSMworker.done_setting_tr_data(self)

            if packet['action'] == 'compute_KTK':
                self.FSMworker.go_computing_KTK(self)          
                self.FSMworker.done_computing_KTK(self)

            if packet['action'] == 'sending_C':
                #self.C = packet['data']['C']
                self.FSMworker.go_storing_C(self, packet)
                self.FSMworker.done_storing_C(self)

        except Exception as err:
            print('ERROR AT CheckNewPacket_worker')
            import code
            code.interact(local=locals())

        return self.terminate
