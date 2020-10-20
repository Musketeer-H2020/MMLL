# -*- coding: utf-8 -*-
'''
Multiclass Logistic Classifier (public model) under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "May 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
#from pympler import asizeof #asizeof.asizeof(my_object)
import pickle

class model():
    def __init__(self):
        self.w = None
        self.classes = None

    def sigm(self, x):
        """
        Computes the sigmoid function

        Parameters
        ----------
        x: float
            input value

        Returns
        -------
        sigm(x): float

        """
        return 1 / (1 + np.exp(-x))

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


        preds_dict = {}
        NCLA = len(self.classes)
        NP = X_b.shape[0]
        X = []
        for cla in self.classes:
            s = np.dot(X_b, self.w_dict[cla]).ravel()
            o = self.sigm(s)
            preds_dict.update({cla: o})
            X.append(o)

        X = np.array(X)
        winners = list(np.argmax(X, axis=0))
        o = [self.classes[pos] for pos in winners] 
        return preds_dict, o

class MLC_pm_Master(Common_to_all_POMs):
    """
    This class implements the Multiclass Logistic Classifier (public model), run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`MLC_pm_Master` instance.

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
        #self.NC = NC                                # No. Centroids
        #self.Nmaxiter = Nmaxiter
        self.master_address = master_address
        self.balance_classes = False
        
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
        #self.cryptonode_address = None

        self.logger = logger                        # logger
        self.comms = comms                          # comms lib
        self.state_dict = None                      # State of the main script
        self.verbose = verbose                      # print on screen when true
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.model = model()
        self.epsilon = 0.00000001  # to avoid log(0)
        for k in range(0, self.Nworkers):
            self.state_dict.update({self.workers_addresses[k]: ''})
        #default values
        self.alpha_fixed = 0.1
        self.alpha_min = -1.0
        self.alpha_max = 1.0
        self.alpha_step = (self.alpha_max - self.alpha_min) / 100
        # we extract the model_parameters as extra kwargs, to be all jointly processed
        try:
            kwargs.update(kwargs['model_parameters'])
            del kwargs['model_parameters']
        except:
            pass
        self.process_kwargs(kwargs)
        self.create_FSM_master()
        self.FSMmaster.master_address = master_address
        self.message_counter = 0    # used to number the messages
        self.newNI_dict = {}

        try:
            if self.target_data_description['NT'] == 1:
                if self.target_data_description['output_type'][0]['type'] == 'cat':
                    self.classes = self.target_data_description['output_type'][0]['values']
                else:
                    self.display('Target values must be categorical (string)')
                    sys.exit()
            else:
                self.display('The case with more than one target is not covered yet.')
                sys.exit()
        except Exception as err:
            self.display('The target_data_description is not well defined, please check.', str(err))
            raise

    def create_FSM_master(self):
        """
        Creates a Finite State Machine to be run at the Master Node

        Parameters
        ----------
        None
        """

        self.display(self.name + ': creating FSM')

        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='update_tr_data', on_enter=['while_update_tr_data']),
            State(name='sending_w', on_enter=['while_sending_w']),
            State(name='updating_w', on_enter=['while_updating_w']),
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_sending_w', 'waiting_order', 'sending_w'],
            ['go_waiting_order', 'sending_w', 'waiting_order'],

            ['go_updating_w', 'waiting_order', 'updating_w'],
            ['go_waiting_order', 'updating_w', 'waiting_order'],
            ]

        class FSM_master(object):

            self.name = 'FSM_master'

            def while_waiting_order(self, MLmodel):
                MLmodel.display(MLmodel.name + ': WAITING for instructions...')
                return

            def while_update_tr_data(self, MLmodel):
                try:
                    action = 'update_tr_data'
                    data = {'classes': MLmodel.classes}
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

            def while_sending_w(self, MLmodel):

                action = 'sending_w'
                data = {'w': MLmodel.model.w_dict}

                # Sending classes to workers at iteration 0
                if MLmodel.kiter == 0:
                    data.update({'classes': MLmodel.classes})

                # In case of balancing data, we send the proportions
                if MLmodel.balance_classes:
                    data.update({'npc_dict': MLmodel.aggregated_Npc_dict})

                packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                '''
                for waddr in MLmodel.workers_addresses:
                    MLmodel.comms.send(waddr, packet)
                '''
                MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
                MLmodel.display(MLmodel.name + ': broadcasted w_dict to all Workers')
                return

            def while_updating_w(self, MLmodel):

                # We reset XTDaX_accum_dict and XTDast_accum_dict
                for cla in MLmodel.classes:
                    MLmodel.XTDaX_accum_dict[cla] = np.zeros((MLmodel.NI + 1, MLmodel.NI + 1))
                    MLmodel.XTDast_accum_dict[cla] = np.zeros((MLmodel.NI + 1, 1))

                    # We accumulate data stored at self.XTDaX_dict
                    for waddr in MLmodel.workers_addresses:
                        MLmodel.XTDaX_accum_dict[cla] += MLmodel.XTDaX_dict[waddr][cla]
                        MLmodel.XTDast_accum_dict[cla] += MLmodel.XTDast_dict[waddr][cla]

                    MLmodel.w_old_dict[cla] = np.copy(MLmodel.model.w_dict[cla])

                    w_new = np.dot(np.linalg.inv(MLmodel.XTDaX_accum_dict[cla] + MLmodel.regularization * np.eye(MLmodel.NI + 1)), MLmodel.XTDast_accum_dict[cla])
                    # --------------------------------------------------
                    if MLmodel.Xval_b is not None:
                    #if False:  # Deactivated by now... pending analysis
                        #print('####################################')
                        #w_new = np.dot(np.linalg.inv(MLmodel.XTDaX_accum + MLmodel.regularization * np.eye(MLmodel.NI + 1)), MLmodel.XTDast_accum)
                        # We explore alfa values to find a minimum in the validation error
                        CE_val = []
                        alphas = np.arange(-2,2,0.01)
                        for alpha in alphas:
                            w_tmp = alpha * w_new + (1 - alpha) * MLmodel.w_old_dict[cla]
                            s_val = np.dot(MLmodel.Xval_b, w_tmp).ravel()
                            o_val = MLmodel.sigm(s_val)
                            yval = (MLmodel.yval == cla).astype(float)
                            ce_val = np.mean(MLmodel.cross_entropy(o_val, yval, MLmodel.epsilon))
                            CE_val.append(ce_val)

                        min_pos = np.argmin(CE_val)
                        alpha_opt = alphas[min_pos]
                        MLmodel.display(MLmodel.name + ': optimal alpha = %s' % str(alpha_opt)[0:7])
                        MLmodel.model.w_dict[cla] = alpha_opt * w_new + (1 - alpha_opt) * MLmodel.w_old_dict[cla]
                    else:
                        alpha = 0.2
                        # --------------------------------------------------
                        #alpha = 0.2
                        MLmodel.model.w_dict[cla] = alpha * w_new + (1 - alpha) * MLmodel.w_old_dict[cla]
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
        self.w_old = np.random.normal(0, 1.0, (self.NI + 1, 1))
        self.XTDaX_accum = np.zeros((self.NI + 1, self.NI + 1))    # Cov. matrix in plaintext
        self.XTDast_accum = np.zeros((self.NI + 1, 1))              # Cov. matrix in plaintext
        self.preds_dict = {}                                           # dictionary storing the prediction errors
        self.AUCs_dict = {}                                           # dictionary storing the prediction errors
        self.XTDaX_dict = {}
        self.XTDast_dict = {}
        self.model.w_dict = {}
        self.w_old_dict = {}
        self.XTDaX_accum_dict = {}
        self.XTDast_accum_dict = {}

        for cla in self.classes:
            self.model.w_dict.update({cla: np.random.normal(0, 0.001, (self.NI + 1, 1))})
            self.w_old_dict.update({cla: np.random.normal(0, 1.0, (self.NI + 1, 1))})
            self.XTDaX_accum_dict.update({cla: np.zeros((self.NI + 1, self.NI + 1))})
            self.XTDast_accum_dict.update({cla: np.zeros((self.NI + 1, 1))})

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
        '''        
        #self.aggregated_Npc_dict
        if self.balance_classes:
            self.display(self.name + ': Balancing classes')
            print(self.aggregated_Npc_dict)
        '''
        self.stop_training = False
        self.kiter = 0

        while not self.stop_training:

            # We send the w and get XTDaX
            self.FSMmaster.go_sending_w(self)
            self.run_Master()

            # This updates self.w_dict and self.w_old_dict
            self.FSMmaster.go_updating_w(self)
            self.FSMmaster.go_waiting_order(self)

            self.kiter += 1
            # Stop if Maxiter is reached
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            inc_w = 0
            for cla in self.classes:
                inc_w += np.linalg.norm(self.model.w_dict[cla] - self.w_old_dict[cla]) / np.linalg.norm(self.model.w_dict[cla])
            inc_w = inc_w / len(self.classes)

            # Stop if convergence is reached
            if inc_w < 0.01:
                self.stop_training = True

            #self.display('==================> ' + str(self.Nmaxiter) + ' ' + str(self.kiter) + ' ' + str(inc_w))
            #print('==================> ' + str(self.Nmaxiter) + ' ' + str(self.kiter) + ' ' + str(inc_w))
            #print(self.regularization)
            message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
            #self.display(message)
            print(message)

        self.display(self.name + ': Training is done')
        self.model.niter = self.kiter
        self.model.classes = self.classes

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
        preds_dict, o = self.model.predict(self, X_b)
        return preds_dict, o

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_sending_XTDaX'):
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
        sender = self.receive_from[packet['sender']]
        
        if packet['action'][0:3] == 'ACK':
            self.display(self.name + ' received ACK from %s: %s' % (str(sender), packet['action']))
            self.state_dict[sender] = packet['action']

        if packet['action'] == 'ACK_sending_XTDaX':
            self.XTDaX_dict.update({sender: packet['data']['XTDaX_dict']})
            self.XTDast_dict.update({sender: packet['data']['XTDast_dict']})

        if packet['action'] == 'ACK_update_tr_data':
            #print('ProcessReceivedPacket_Master ACK_update_tr_data')
            self.newNI_dict.update({sender: packet['data']['newNI']})

        return


#===============================================================
#                 Worker
#===============================================================
class MLC_pm_Worker(Common_to_all_POMs):
    '''
    Class implementing Multiclass Logistic Classifier (public model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`MLC_pm_Worker` instance.

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
        #self.Xtr_b = self.add_bias(Xtr_b.astype(float))
        #self.ytr = ytr  # We do not convert to float 
        self.NPtr = len(ytr)
        self.w = None
        self.epsilon = 0.00000001  # to avoid log(0)
        self.create_FSM_worker()
        self.message_id = 0    # used to number the messages
        self.cryptonode_address = None

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
                    MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
                    
                    MLmodel.classes = packet['data']['classes']

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

            def while_computing_XTDaX(self, MLmodel, packet):
                try:
                    MLmodel.display(MLmodel.name + ' %s: computing XTDaX...' % (str(MLmodel.worker_address)))
                    XTDaX_dict = {}
                    XTDast_dict = {}
                    for cla in MLmodel.classes:
                        s = np.dot(MLmodel.Xtr_b, MLmodel.w_dict[cla]).ravel()
                        o = MLmodel.sigm(s)
                        ytr = MLmodel.ytr_dict[cla]
                        st = MLmodel.st_dict[cla]
                        ce = MLmodel.cross_entropy(o, ytr, MLmodel.epsilon)
                        e2 = (st.ravel() - s) ** 2 + 0.000001
                        a = np.sqrt(np.abs(np.divide(ce, e2)))

                        try:
                            wpos = packet['data']['npc_dict'][cla]
                            wneg = 0
                            for ccla in MLmodel.classes:
                                if ccla != cla:
                                    wneg += packet['data']['npc_dict'][ccla]

                            wbalance_pos = (MLmodel.ytr == cla).astype(float) / wpos * (wpos + wneg)
                            wbalance_neg = (MLmodel.ytr != cla).astype(float) / wneg * (wpos + wneg)
                            wbalance = wbalance_pos + wbalance_neg

                            a = np.multiply(a, wbalance)
                        except:
                            pass
                            
                        #XTa = np.dot(MLmodel.Xtr_b.T, np.diag(a))  # this uses a lot of memory
                        XTa = np.multiply(MLmodel.Xtr_b.T, a)                       
                        XTDaX = np.dot(XTa, MLmodel.Xtr_b)
                        XTDast = np.dot(XTa, st)
                        XTDaX_dict.update({cla: XTDaX})
                        XTDast_dict.update({cla: XTDast})

                    action = 'ACK_sending_XTDaX'
                    data = {'XTDaX_dict': XTDaX_dict, 'XTDast_dict': XTDast_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #MLmodel.comms.send(MLmodel.master_address, packet)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_XTDaX' % (str(MLmodel.worker_address)))
                except Exception as err:
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    raise
                    print('STOP AT while_computing_XTDaX')
                    import code
                    code.interact(local=locals())    
                    pass
                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),            
            State(name='computing_XTDaX', on_enter=['while_computing_XTDaX']),
            State(name='Exit', on_enter=['while_Exit'])
           ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_computing_XTDaX', 'waiting_order', 'computing_XTDaX'],
            ['done_computing_XTDaX', 'computing_XTDaX', 'waiting_order'],

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

        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
            self.terminate = True

        if packet['action'] == 'update_tr_data':
            # We update the training data
            self.FSMworker.go_setting_tr_data(self, packet)
            self.FSMworker.done_setting_tr_data(self)

        if packet['action'] == 'sending_w':
            self.w_dict = packet['data']['w']

            try:
                #### processing classes if communicated by the master...
                # This is only executed at the first iteration 
                classes = packet['data']['classes']
                self.classes = [str(cla) for cla in classes]
                self.ytr_dict = {}
                self.st_dict = {}
                for cla in self.classes:
                    y = (self.ytr == cla).astype(float)
                    self.ytr_dict[cla] = np.copy(y)
                    st = -np.log(1.0 / (y * (1.0 - self.epsilon) + self.epsilon * (1.0 - y)) - 1.0)
                    st = st.reshape((-1, 1))
                    self.st_dict.update({cla: st})
            except:
                pass

            self.FSMworker.go_computing_XTDaX(self, packet)
            self.FSMworker.done_computing_XTDaX(self)

        return self.terminate
