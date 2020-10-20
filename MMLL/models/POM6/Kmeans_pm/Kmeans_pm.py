# -*- coding: utf-8 -*-
'''
Kmeans (public model) under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Apr. 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine

class model():
    def __init__(self):
        self.c = None

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
        XTC = np.dot(X_b, self.c.T)
        x2 = np.sum(X_b * X_b, axis=1).reshape((-1, 1))
        c2 = np.sum(self.c * self.c, axis=1).reshape((1, -1))
        D = x2 - 2 * XTC + c2
        predictions = np.argmin(D, axis=1)
        return predictions


class Kmeans_pm_Master(Common_to_all_POMs):
    """
    This class implements the Kmeans (public model), run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
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
        self.newNI_dict = {}

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
            State(name='sending_C', on_enter=['while_sending_C']),
            State(name='updating_C', on_enter=['while_updating_C']),
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_sending_C', 'waiting_order', 'sending_C'],
            ['go_waiting_order', 'sending_C', 'waiting_order'],

            ['go_updating_C', 'waiting_order', 'updating_C'],
            ['go_waiting_order', 'updating_C', 'waiting_order'],
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
                    data = {'C': MLmodel.model.c}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    '''
                    for waddr in MLmodel.workers_addresses:
                        MLmodel.comms.send(waddr, packet)
                    '''
                    MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
                    MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')

                except Exception as err:
                    MLmodel.display('ERROR: %s %s' % (str(err), str(type(err))))
                    MLmodel.display('ERROR AT while_sending_C')
                    import code
                    code.interact(local=locals())         
                return
            
            def while_updating_C(self, MLmodel):
                try:
                    #print(MLmodel.c)
                    users = MLmodel.Cinc_dict.keys()
                    NC = MLmodel.NC
                    NI = MLmodel.NI
                    newC = np.zeros((NC, NI))
                    TotalP = np.zeros((NC, 1))
                    Dacum = np.zeros((NC, 1))

                    for user in users:
                        cinc = MLmodel.Cinc_dict[user]
                        for kc in range(0, NC):
                            if cinc['Ninc'][kc] > 0:
                                newC[kc, :] += cinc['C_inc'][kc]
                                TotalP[kc] += cinc['Ninc'][kc]
                                Dacum[kc] += cinc['Dist_acum'][kc]

                    for kc in range(0, NC):
                        if TotalP[kc] > 0:
                            newC[kc, :] = newC[kc, :] / TotalP[kc]
                    
                    MLmodel.model.c = newC

                    MLmodel.display('---------------')
                    l = [int(p) for p in list(TotalP.ravel())]
                    MLmodel.display(str(l))
                    '''
                    print('---------------')
                    print(sum(TotalP))
                    print('---------------')
                    '''
                    # Evaluate if centroid realocation is needed
                    Nmean = np.mean(TotalP)
                    Overpop = list(np.where(TotalP > Nmean * 1.5)[0])
                    Maxpop = np.argmax(TotalP)
                    Overpop = list(set(Overpop + [Maxpop]))
                    Underpop = list(np.where(TotalP < 0.3 * Nmean)[0])
                    for which_under in Underpop:
                        which_over = Overpop[np.random.randint(0, len(Overpop))]
                        newc = MLmodel.model.c[which_over, :] + np.random.normal(0, 0.01, (1, NI))
                        MLmodel.model.c[which_under, :] = newc
                    MLmodel.display(str(Overpop) + ' ' + str(Underpop))
                    MLmodel.display(str(len(Overpop)) + ' ' + str(len(Underpop)))
                    MLmodel.display('---------------')
                except Exception as err:
                    MLmodel.display('ERROR: %s %s' % (str(err), str(type(err))))
                    MLmodel.display('ERROR AT while_updating_C')
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
        self.model.c = np.random.normal(0, 0.3, (self.NC, self.NI))      # centroids
        self.normx_dict = {}                                            # dictionary storing the norm2 of patterns
        self.Distc_dict = {}
        self.Cinc_dict = {}
        self.c_old = np.zeros((self.NC, self.NI))
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

        self.c_orig = np.copy(self.model.c)

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

        stop_training = False
        kiter = 0
        while not stop_training:
            self.c_old = np.copy(self.model.c)
            # LOOP ======================
            # We obtain the dot products
            self.FSMmaster.go_sending_C(self)
            self.run_Master()

            self.FSMmaster.go_updating_C(self)
            self.FSMmaster.go_waiting_order(self)

            inc_normc = np.linalg.norm(self.c_old - self.model.c) / np.linalg.norm(self.model.c)
            self.display(self.name + ': INC_C norm = %s' % str(inc_normc)[0:7])
            #print(self.Nmaxiter, kloop)
            message = 'Maxiter = %d, iter = %d, inc = %f' % (self.Nmaxiter, kiter, inc_normc)
            #self.display(message)
            print(message)
            if kiter == self.Nmaxiter or inc_normc < self.conv_stop:
                stop_training = True
            else:
                kiter += 1

        self.display(self.name + ': Training is done')
        self.model.niter = kiter

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
        #########3  PENDING
        prediction_values = None
        return prediction_values

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_sending_C_inc'):
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
            self.display(self.name + ': received ACK from %s: %s' % (str(sender), packet['action']))
            self.state_dict[sender] = packet['action']

        if packet['action'] == 'ACK_sending_C_inc':
            self.Cinc_dict.update({sender : {'C_inc': packet['data']['C_inc'], 'Ninc': packet['data']['Ninc'], 'Dist_acum': packet['data']['Dist_acum']}})

        if packet['action'] == 'ACK_update_tr_data':
            #print('ProcessReceivedPacket_Master ACK_update_tr_data from %s' % str(sender))
            self.newNI_dict.update({sender: packet['data']['newNI']})

        return


#===============================================================
#                 Worker
#===============================================================
class Kmeans_pm_Worker(Common_to_all_POMs):
    '''
    Class implementing Kmeans (public model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`Kmeans_pm_Worker` instance.

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
        self.NPtr = Xtr_b.shape[0]
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
                    #MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
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

            def while_computing_DXC(self, MLmodel):
                try:
                    #print('1')
                    #X = np.random.normal(0, 1, (100000000, 100000000))
                    #print(aaa)
                    #print('2')
                    MLmodel.display(MLmodel.name + ' %s: Computing DXC' % (str(MLmodel.worker_address)))
                    X = MLmodel.Xtr_b
                    C = MLmodel.C
                    # This could be computed only once
                    normx2 = np.linalg.norm(X, 2, axis=1) ** 2
                    NP = normx2.shape[0]
                    normc2 = np.linalg.norm(C, 2, axis = 1) ** 2
                    NC = normc2.shape[0]
                    DP = np.dot(X, C.T) 
                    DXC = np.kron(normx2, np.ones((NC, 1))).T - 2 * DP + np.kron(np.ones((NP, 1)), normc2)
                    pred = np.argmin(DXC, axis = 1)
                    C_inc = []
                    Ninc = []
                    Dist_acum = []
                    for kc in range(0, NC):
                        cuales = pred == kc
                        Nchunk = np.sum(cuales)

                        if Nchunk > 2:
                            c_kc = np.sum(X[cuales,:], axis=0)
                        else:
                            c_kc = None
                            Nchunk = 0

                        C_inc.append(c_kc)
                        Ninc.append(Nchunk)
                        # Computing accumulated error per centroid
                        Dacum = np.sum(DXC[cuales, kc])
                        Dist_acum.append(Dacum)

                    action = 'ACK_sending_C_inc'
                    data = {'C_inc': C_inc, 'Ninc':Ninc, 'Dist_acum':Dist_acum}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_C_inc' % (str(MLmodel.worker_address)))
                except Exception as err:
                    MLmodel.display('ERROR: %s %s' % (str(err), str(type(err))))
                    MLmodel.display('ERROR AT while_computing_DXC')
                    import code
                    code.interact(local=locals())         

                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='computing_DXC', on_enter=['while_computing_DXC']),
            State(name='Exit', on_enter=['while_Exit']),
           ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_computing_DXC', 'waiting_order', 'computing_DXC'],
            ['done_computing_DXC', 'computing_DXC', 'waiting_order'],

            ['go_exit', 'waiting_order', 'Exit']
            ]

        self.FSMworker = FSM_worker()
        self.grafmachine_worker = GraphMachine(model=self.FSMworker,
            states=states_worker,
            transitions=transitions_worker,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of worker No. %s" % str(self.worker_address),
            show_conditions=True)
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

        if packet['action'] == 'sending_C':
            self.C = packet['data']['C']
            self.FSMworker.go_computing_DXC(self)
            self.FSMworker.done_computing_DXC(self)

        if packet['action'] == 'update_tr_data':
            # We update the training data
            self.FSMworker.go_setting_tr_data(self, packet)
            self.FSMworker.done_setting_tr_data(self)

        return self.terminate
