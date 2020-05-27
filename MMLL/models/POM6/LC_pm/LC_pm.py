# -*- coding: utf-8 -*-
'''
Logistic Classifier model (public model) under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "May 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine

class model():
    def __init__(self):
        self.w = None

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
        return self.sigm(np.dot(X_b, self.w.ravel()))


class LC_pm_Master(Common_to_all_POMs):
    """
    This class implements the Logistic Classifier model, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=False, **kwargs):
        """
        Create a :class:`LC_pm_Master` instance.

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
        # we extract the model_parameters as extra kwargs, to be all jointly processed
        try:
            kwargs.update(kwargs['model_parameters'])
            del kwargs['model_parameters']
        except Exception as err:
            pass
        self.process_kwargs(kwargs)
        self.create_FSM_master()
        self.FSMmaster.master_address = master_address
        self.message_counter = 0    # used to number the messages
        self.cryptonode_address = None

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
            State(name='sending_w', on_enter=['while_sending_w']),
            State(name='updating_w', on_enter=['while_updating_w']),
        ]

        transitions_master = [
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

            def while_sending_w(self, MLmodel):
                try:
                    action = 'sending_w'
                    data = {'w': MLmodel.model.w}
                    # In case of balancing data, we send the proportions
                    if MLmodel.balance_classes:
                        data.update({'npc_dict': MLmodel.aggregated_Npc_dict})

                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}

                    MLmodel.comms.broadcast(packet, receivers_list=MLmodel.workers_addresses)
                    MLmodel.display(MLmodel.name + ': broadcasted w to all Workers')
                except Exception as err:
                    self.display('ERROR: %s' % err)
                    self.display('ERROR AT while_sending_w')
                    import code
                    code.interact(local=locals())
                return

            def while_updating_w(self, MLmodel):
                try:
                    MLmodel.XTDaX_accum = np.zeros((MLmodel.NI + 1, MLmodel.NI + 1))
                    MLmodel.XTDast_accum = np.zeros((MLmodel.NI + 1, 1))
                    for waddr in MLmodel.workers_addresses:
                        MLmodel.XTDaX_accum += MLmodel.XTDaX_dict[waddr]['XTDaX']
                        MLmodel.XTDast_accum += MLmodel.XTDaX_dict[waddr]['XTDast']

                    # Trying to use the validation set to estimate the optima update
                    MLmodel.w_old = np.copy(MLmodel.model.w)
                    w_new = np.dot(np.linalg.inv(MLmodel.XTDaX_accum + MLmodel.regularization * np.eye(MLmodel.NI + 1)), MLmodel.XTDast_accum)

                    if MLmodel.Xval_b is not None:
                        # We explore alfa values to find a minimum in the validation error
                        CE_val = []
                        alphas = np.arange(-2, 2, 0.01)
                        for alpha in alphas:
                            w_tmp = alpha * w_new + (1 - alpha) * MLmodel.w_old
                            s_val = np.dot(MLmodel.Xval_b, w_tmp).ravel()
                            o_val = MLmodel.sigm(s_val)
                            ce_val = np.mean(MLmodel.cross_entropy(o_val, MLmodel.yval, MLmodel.epsilon))
                            CE_val.append(ce_val)

                        min_pos = np.argmin(CE_val)
                        alpha_opt = alphas[min_pos]
                        MLmodel.display(MLmodel.name + ': optimal alpha = %s' % str(alpha_opt)[0:7])
                        MLmodel.model.w = alpha_opt * w_new + (1 - alpha_opt) * MLmodel.w_old
                    else:
                        alpha = 0.1
                        MLmodel.model.w = alpha * w_new + (1 - alpha) * MLmodel.w_old
                except Exception as err:
                    self.display('ERROR: %s' % err)
                    self.display('ERROR AT while_updating_w')
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
        self.AUCs_dict = {}                                           # dictionary storing the prediction errors
        self.R_dict = {}
        self.r_dict = {}
        self.XTDaX_dict = {}
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

        ## Adding bias to data

        self.stop_training = False
        self.kiter = 0

        self.Xval_b = self.add_bias(self.Xval_b)

        #self.aggregated_Npc_dict may be available
        if self.balance_classes:
            self.display(self.name + ': Balancing classes')
            print(self.aggregated_Npc_dict)

        # Getting number of patterns per class (npc) using common
        while not self.stop_training:

            # We send the w and get XTDaX
            self.FSMmaster.go_sending_w(self)
            self.run_Master()

            # This updates self.w and self.w_old
            self.FSMmaster.go_updating_w(self)
            self.FSMmaster.go_waiting_order(self)

            self.kiter += 1
            # Stop if Maxiter is reached
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            inc_w = np.linalg.norm(self.model.w - self.w_old) / np.linalg.norm(self.w_old)
            # Stop if convergence is reached
            if inc_w < 0.01:
                self.stop_training = True

            #message = '==================> ' + str(self.regularization) + ', ' + str(self.Nmaxiter) + ', ' + str(self.kiter) + ', ' + str(inc_w)
            message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
            #self.display(message)
            print(message)

        self.display(self.name + ': Training is done')

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
        #prediction_values = self.sigm(np.dot(X_b, self.w.ravel()))
        prediction_values = self.model.predict(X_b)
        return prediction_values

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_sending_XTDaX'):
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
            self.display('COMMS_MASTER_RECEIVED %s' % packet['action'], verbose=False)

        if packet['action'] == 'ACK_sending_XTDaX':
            self.XTDaX_dict.update({sender: {'XTDaX': packet['data']['XTDaX'], 'XTDast': packet['data']['XTDast']}})

        return


#===============================================================
#                 Worker
#===============================================================
class LC_pm_Worker(Common_to_all_POMs):
    '''
    Class implementing Logistic Classifier (public model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`LC_pm_Worker` instance.

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
        self.Xtr_b = self.add_bias(Xtr_b)
        self.ytr = ytr
        self.NPtr = len(ytr)
        self.w = None
        self.epsilon = 0.00000001  # to avoid log(0)
        self.st = -np.log(1.0 / (self.ytr * (1.0 - self.epsilon) + self.epsilon * (1.0 - self.ytr)) - 1.0)
        self.st = self.st.reshape((self.NPtr, 1))
        self.create_FSM_worker()

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

            # Enter/exit callbacks are defined here
            def while_waiting_order(self, MLmodel):
                MLmodel.display(self.name + ' %s: WAITING for instructions...' % (str(MLmodel.worker_address)))
                return

            def while_computing_XTDaX(self, MLmodel, packet):
                try:
                    NPtr = MLmodel.Xtr_b.shape[0]
                    s = np.dot(MLmodel.Xtr_b, MLmodel.w).ravel()
                    o = MLmodel.sigm(s)
                    ce = MLmodel.cross_entropy(o, MLmodel.ytr, MLmodel.epsilon)
                    e2 = (MLmodel.st.ravel() - s) ** 2 + 0.000001
                    #a = np.sqrt(np.abs(np.divide(ce, e2)))
                    a = np.abs(np.divide(ce, e2)).reshape((NPtr, 1))

                    try:
                        wpos = packet['data']['npc_dict']['1']
                        wneg = packet['data']['npc_dict']['0']
                        wbalance_pos = (MLmodel.ytr == 1).astype(float) / wpos * (wpos + wneg)
                        wbalance_neg = (MLmodel.ytr == 0).astype(float) / wneg * (wpos + wneg)
                        wbalance = wbalance_pos + wbalance_neg
                        a = np.multiply(a, wbalance)
                    except:
                        pass

                    Xa = MLmodel.Xtr_b * a
                    XaTXa = np.dot(Xa.T, MLmodel.Xtr_b)
                    XaTst = np.dot(Xa.T, MLmodel.st)

                    action = 'ACK_sending_XTDaX'
                    data = {'XTDaX': XaTXa, 'XTDast': XaTst}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_XTDaX' % (str(MLmodel.worker_address)))
                except Exception as err:
                    self.display('ERROR: %s' % err)
                    self.display('ERROR AT while_computing_XTDaX')
                    import code
                    code.interact(local=locals())
                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),

            State(name='computing_XTDaX', on_enter=['while_computing_XTDaX']),

            State(name='Exit', on_enter=['while_Exit'])
           ]

        transitions_worker = [
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

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
            self.terminate = True

        if packet['action'] == 'sending_w':
            # We update the model weights
            self.w = packet['data']['w']
            self.FSMworker.go_computing_XTDaX(self, packet)
            self.FSMworker.done_computing_XTDaX(self)

        return self.terminate
