# -*- coding: utf-8 -*-
'''
Linear Regression model under POM4

'''

__author__ = "Angel Navia-Vázquez  & Francisco González-Serrano"
__date__ = "Apr. 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
#from pympler import asizeof #asizeof.asizeof(my_object)
import pickle

class model():
    def __init__(self):
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
        prediction_values = np.dot(X_b, self.w.ravel())
        return prediction_values


class LR_Master(Common_to_all_POMs):
    """
    This class implements the Linear Regression model, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`LR_Master` instance.

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
        self.pom = 4
        self.model_type = model_type
        self.name = 'POM%d_' % self.pom + self.model_type + '_Master'                 # Name
        #self.NC = NC                                # No. Centroids
        #self.Nmaxiter = Nmaxiter
        self.master_address = master_address
        self.workers_addresses = workers_addresses
        self.cryptonode_address = None
        self.Nworkers = len(workers_addresses)                    # Nworkers
        self.logger = logger                        # logger
        self.comms = comms                          # comms lib
        self.state_dict = None                      # State of the main script
        self.verbose = verbose                      # print on screen when true
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.model = model()
        #self.regularization = regularization
        #self.classes = classes
        #self.balance_classes = balance_classes
        #self.Xval_b = Xval_b
        #self.yval = yval
        self.epsilon = 0.00000001  # to avoid log(0)
        for k in range(0, self.Nworkers):
            self.state_dict.update({self.workers_addresses[k]: ''})
        #default values
        # we extract the model_parameters as extra kwargs, to be all jointly processed
        try:
            kwargs.update(kwargs['model_parameters'])
            del kwargs['model_parameters']
        except:
            pass
        self.process_kwargs(kwargs)
        self.message_counter = 0    # used to number the messages
        self.XTX_dict = {}
        self.XTy_dict = {}
        #self.encrypter = self.cr.get_encrypter()  # to be shared        # self.encrypter.encrypt(np.random.normal(0, 1, (2,3)))
        #self.decrypter = self.cr.get_decrypter()  # to be kept as secret  self.encrypter.decrypt()
        self.create_FSM_master()
        self.FSMmaster.master_address = master_address


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
            State(name='store_Xyblinded', on_enter=['while_store_Xyblinded']),
            State(name='mult_XB', on_enter=['while_mult_XB']),
            State(name='decrypt_model', on_enter=['while_decrypt_model'])
        ]

        transitions_master = [
            ['go_store_Xyblinded', 'waiting_order', 'store_Xyblinded'],
            ['done_store_Xyblinded', 'store_Xyblinded', 'waiting_order'],

            ['go_mult_XB', 'waiting_order', 'mult_XB'],
            ['done_mult_XB', 'mult_XB', 'waiting_order'],

            ['go_decrypt_model', 'waiting_order', 'decrypt_model'],
            ['done_decrypt_model', 'decrypt_model', 'waiting_order']
        ]


        class FSM_master(object):

            self.name = 'FSM_master'

            def while_waiting_order(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' is waiting...')
                except:
                    print('ERROR AT while_waiting_order')
                    import code
                    code.interact(local=locals())

                return
            '''
            def while_send_w_encr(self, MLmodel):
                try:
                    data = {}
                    data.update({'w_encr': MLmodel.w_encr})
                    #wdill = MLmodel.dill_it(MLmodel.wq_encr)
                    #data.update({'wq_encr': wdill})

                    packet = {'action': 'send_w_encr', 'to': 'MLmodel', 'data': data}
                    MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    MLmodel.display(MLmodel.name + ' send_w_encr to workers')
                except:
                    print('ERROR AT while_send_w_encr')
                    import code
                    code.interact(local=locals())
                return
            '''
            def while_mult_XB(self, MLmodel, B_bl):
                try:
                    data = {'B_bl': B_bl}
                    packet = {'action': 'send_mult_XB', 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    MLmodel.display(MLmodel.name + ' send_mult_XB to cryptonode')
                except:
                    print('ERROR AT LR while_mult_XB')
                    import code
                    code.interact(local=locals())
                    pass
                return

            def while_decrypt_model(self, MLmodel, model_encr):
                try:
                    # Adding blinding to model
                    MLmodel.bl = {}
                    model_encr_bl = {}

                    for key in list(model_encr.keys()):
                        x = model_encr[key]
                        #print('while_decrypt_model LR bl=:')
                        M, N = x.shape
                        bl = np.random.normal(0, 1, (M, N))
                        #print(bl)
                        MLmodel.bl.update({key: bl})
                        model_encr_bl.update({key: x + bl})

                    data = {'model_bl': model_encr_bl}
                    packet = {'action': 'send_model_encr_bl', 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    MLmodel.display(MLmodel.name + ' send_model_encr_bl to cryptonode')
                except:
                    print('ERROR AT LR while_decrypt_model')
                    import code
                    code.interact(local=locals())
                    pass
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
        self.w_decr = np.random.normal(0, 0.001, (self.NI + 1, 1))      # weights in plaintext, first value is bias
        self.R_central_decr = np.zeros((self.NI + 1, self.NI + 1))    # Cov. matrix in plaintext
        self.r_central_decr = np.zeros((self.NI + 1, 1))              # Cov. matrix in plaintext
        self.preds_dict = {}                                           # dictionary storing the prediction errors
        self.AUCs_dict = {}                                           # dictionary storing the prediction errors
        self.R_dict = {}
        self.r_dict = {}
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
        print(self.name + ': Starting training')
        #self.X_encr_dict

        #self.MasterMLmodel.BX_dict
        #self.MasterMLmodel.By_dict

        self.NI = self.input_data_description['NI']

        '''
        # Adding encrypted bias
        for waddr in self.X_encr_dict.keys():
            X = self.X_encr_dict[waddr]
            NP = X.shape[0]
            ones = np.ones((NP, 1))
            ones_encr = self.encrypter.encrypt(ones)
            self.X_encr_dict[waddr] = np.hstack((ones_encr, X))
        '''
        self.w = np.random.normal(0, 0.1, (self.NI + 1, 1))
        self.w_encr = self.encrypter.encrypt(self.w)
        self.w_old = np.random.normal(0, 10, (self.NI + 1, 1)) # large to avoid stop at first iteration

        # Data at self.X_encr_dict, self.y_encr_dict

        check = False
        which = '4'

        self.stop_training = False
        kiter = 0
        while not self.stop_training:

            # Computing wTX
            #self.wTX_encr_dict = self.crypto_mult_X(self.w_encr.T)
            self.Xw_encr_dict = {}
            for key in self.X_encr_dict:
                self.Xw_encr_dict.update({key: np.dot(self.X_encr_dict[key], self.w)})
            

            if check:
                X0 = self.decrypter.decrypt(self.X_encr_dict[which])
                #w = self.decrypter.decrypt(self.w_encr)
                o = np.dot(X0, self.w)
                #o_ = np.sum(self.Xw_encr_dict[which], axis=1).reshape((-1, 1))
                o_decr = self.decrypter.decrypt(self.Xw_encr_dict[which])
                print(np.linalg.norm(o - o_decr))  # OK


            # Computing errors
            self.e_encr_dict = {}
            for waddr in self.workers_addresses:
                #X = self.X_encr_dict[waddr]
                y = self.y_encr_dict[waddr].reshape(-1, 1)
                
                # We neeed the mult prototocol to compute this...
                # o_encr = np.dot(X, self.w)
                #o_encr = np.sum(self.wTX_encr_dict[waddr], axis=1).reshape((-1, 1))
                o_encr = self.Xw_encr_dict[waddr]
                e_encr = y - o_encr
                self.e_encr_dict.update({waddr: e_encr})

            if check:
                y0_encr = self.y_encr_dict[which].reshape(-1, 1)
                y0 = self.decrypter.decrypt(y0_encr)
                e_ = y0 - o_decr
                e0 = self.decrypter.decrypt(self.e_encr_dict[which])
                print(np.linalg.norm(e0 - e_))  # OK

            # Computing eX
            self.eX_encr_dict = self.crypto_mult_X(self.e_encr_dict)

            # Computing gradients
            #self.grad_encr_dict = {}

            grad_encr = self.encrypter.encrypt(np.zeros((self.NI + 1, 1)))
            Ntotal = 0
            for waddr in self.workers_addresses:
                eX_encr = self.eX_encr_dict[waddr]
                Ntotal += eX_encr.shape[0]
                grad_encr += np.sum(eX_encr, axis=0).reshape((-1, 1))

            if check:
                grad0 = np.mean(e0 * X0, axis=0).reshape((-1, 1))
                grad0_decr = self.decrypter.decrypt(np.mean(self.eX_encr_dict[which], axis=0).reshape((-1, 1)))
                print(np.linalg.norm(grad0 - grad0_decr))  # OK

            self.w_encr += self.mu * grad_encr / Ntotal

            # Decrypting the model
            self.model_decr = self.decrypt_model({'w': self.w_encr})
            #self.w = self.decrypter.decrypt(self.w_encr)
            self.w = np.copy(self.model_decr['w'])

            if check:
                w_ok = self.decrypter.decrypt(self.w_encr)  ### this is not allowed
                w_mal = self.model_decr['w']
                print(np.linalg.norm(w_ok - w_mal))  # Not OK

            #self.w_encr = self.encrypter.encrypt(self.w)           
            
            # stopping
            inc_w = np.linalg.norm(self.w - self.w_old) / np.linalg.norm(self.w_old)
            # Stop if convergence is reached
            if inc_w < 0.005:
                self.stop_training = True
            if kiter == self.Nmaxiter:
                self.stop_training = True
           
            message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, kiter, inc_w)
            self.display(message, True)
            kiter += 1
            self.w_old = self.w.copy()

        self.model.w = self.w
        self.display(self.name + ': Training is done', True)
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
        if self.chekAllStates('ACK_grads'):
            self.FSMmaster.done_send_w_encr(self)

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
            if packet['action'][0:3] == 'ACK':
                self.state_dict[sender] = packet['action']

            if packet['action'] == 'ACK_grads':
                self.grads_dict.update({sender: packet['data']['grad_encr']})

            if packet['action'] == 'ACK_sent_XB_bl_encr_dict':
                self.XB_bl_encr_dict = packet['data']['XB_bl_encr_dict']
                self.FSMmaster.done_mult_XB(self)

            if packet['action'] == 'ACK_sent_decr_bl_model':
                self.model_decr_bl = packet['data']['model_decr_bl']
                self.FSMmaster.done_decrypt_model(self)

        except Exception as err:
            print('ERROR AT ProcessReceivedPacket_Master')
            raise
            import code
            code.interact(local=locals())         

        return


#===============================================================
#                 Worker
#===============================================================
class LR_Worker(Common_to_all_POMs):
    '''
    Class implementing Linear Regression, run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None, cryptonode_address=None):
        """
        Create a :class:`LR_Worker` instance.

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
        self.pom = 4
        self.master_address = master_address
        self.worker_address = worker_address                    # The id of this Worker
        self.cryptonode_address = cryptonode_address
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.model_type = model_type
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.name = model_type + '_Worker'    # Name
        self.verbose = verbose                  # print on screen when true
        #self.Xtr_b = Xtr_b
        self.Xtr_b = self.add_bias(Xtr_b)
        self.ytr = ytr
        self.NPtr = len(ytr)
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
                MLmodel.display(self.name + ' %s: WAITING for instructions...' % (str(MLmodel.worker_address)))
                return

            def while_compute_gradients(self, MLmodel, packet):
                try:
                    MLmodel.display(MLmodel.name + ' %s: computing gradients...' % (str(MLmodel.worker_address)))
                    w_encr = packet['data']['w_encr']
                   
                    #NW = wq_encr.shape[0]
                    #for kw in range(NW):
                    #    wq_encr[kw, 0].encrypter = MLmodel.cr.encrypter
                    
                    #wq_encr = MLmodel.undill_it(packet['data']['wq_encr'])

                    X = MLmodel.Xtr_b
                    NP = X.shape[0]
                    NI = X.shape[1]
                    y = MLmodel.ytr.reshape(NP, 1)

                    o_encr = np.dot(X, w_encr)
                    e_encr = y - o_encr
                    #Xeq_encr = MLmodel.cr.vmult(X / NP, eq_encr)  # Normalized error by NP
                    Xe_encr = X * e_encr
                    grad_encr = np.sum(Xe_encr, axis=0).reshape((NI, 1)) / NP

                    action = 'ACK_grads'
                    data = {'grad_encr': grad_encr}
                    packet = {'action': action, 'sender': MLmodel.worker_address, 'data': data, 'to': 'MLmodel'}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_grads' % (str(MLmodel.worker_address)))
                except:
                    print('ERROR AT while_compute_gradients')
                    import code
                    code.interact(local=locals())
                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order'])
            ]

        transitions_worker = [
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

            if packet['action'] == 'ACK_sent_encrypter':
                print('STOP AT  ProcessReceivedPacket_Worker')
                import code
                code.interact(local=locals())

                #self.FSMworker.go_storing_Pk(self, packet)
                #self.FSMworker.done_storing_Pk(self)

            if packet['action'] == 'send_w_encr':
                self.FSMworker.go_compute_gradients(self, packet)
                self.FSMworker.done_compute_gradients(self)
        except:
            print('ERROR AT ProcessReceivedPacket_Worker')
            import code
            code.interact(local=locals())

        return self.terminate



#===============================================================
#                 Crypto
#===============================================================
class LR_Crypto(Common_to_all_POMs):
    '''
    Class implementing Linear Regression, run at Crypto

    '''

    def __init__(self, cryptonode_address, master_address, model_type, comms, logger, verbose=True):
        """
        Create a :class:`LR_Crypto` instance.

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

        """
        self.pom = 4
        self.master_address = master_address
        self.cryptonode_address = cryptonode_address
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.model_type = model_type
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.name = model_type + '_Cryptor'    # Name
        self.verbose = verbose                  # print on screen when true
        self.create_FSM_crypto()
        self.message_id = 0    # used to number the messages

    def create_FSM_crypto(self):
        """
        Creates a Finite State Machine to be run at the Worker Node

        Parameters
        ----------
        None
        """
        self.name = 'FSM_crypto'

        self.display(self.name + ': creating FSM')

        class FSM_crypto(object):

            name = 'FSM_crypto'

            def while_waiting_order(self, MLmodel):
                MLmodel.display(self.name + ' %s: WAITING for instructions...' % (str(MLmodel.worker_address)))
                return

        states_crypto = [
            State(name='waiting_order', on_enter=['while_waiting_order'])
        ]

        transitions_crypto = []

        self.FSMcrypto = FSM_crypto()
        self.grafmachine_crypto = GraphMachine(model=self.FSMcrypto,
            states=states_crypto,
            transitions=transitions_crypto,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of crypto",
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
        try:
            # Exit the process
            if packet['action'] == 'STOP':
                self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
                self.terminate = True

            if packet['action'] == 'send_Pk':
                self.FSMworker.go_storing_Pk(self, packet)
                self.FSMworker.done_storing_Pk(self)

            if packet['action'] == 'send_w_encr':
                self.FSMworker.go_compute_gradients(self, packet)
                self.FSMworker.done_compute_gradients(self)
        except:
            print('ERROR AT ProcessReceivedPacket_Crypto')
            import code
            code.interact(local=locals())

        return self.terminate
