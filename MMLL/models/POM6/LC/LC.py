# -*- coding: utf-8 -*-
'''
Logistic Classifier model under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Jan 2021"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
from pympler import asizeof #asizeof.asizeof(my_object)
import pickle
import dill
import time

class Model():
    """
    Logistic Classifier model.
    """
    def __init__(self):
        self.w = None
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']
        t = time.time()
        seed = int((t - int(t)) * 10000)
        np.random.seed(seed=seed)

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

    def predict(self, X):
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
        X_b = np.hstack((np.ones((X.shape[0], 1)), X))
        return self.sigm(np.dot(X_b, self.w.ravel()))

    def save(self, filename=None):
        """
        Saves the trained model to file. The valid file extensions are:            
            - "pkl": saves the model as a Python3 pickle file       
            - "onnx": saves the model using Open Neural Network Exchange format (ONNX)'            
            - "pmml": saves the model using Predictive Model Markup Language (PMML)'      

        Parameters
        ----------
        filename: string
            path+filename          
        """
        if filename is None:
            print('=' * 80)
            print('Model Save Error: A valid filename must be provided, otherwise nothing is saved. The valid file extensions are:')            
            print('\t - "pkl": saves the model as a Python3 pickle file')            
            print('\t - "onnx": saves the model using Open Neural Network Exchange format (ONNX)')            
            print('\t - "pmml": saves the model using Predictive Model Markup Language (PMML)')            
            print('=' * 80)
        else:
            # Checking filename extension
            extension = filename.split('.')[-1]
            if extension not in self.supported_formats:
                print('=' * 80)
                print('Model Save Error: Unsupported format. The valid file extensions are:')            
                print('\t - "pkl": saves the model as a Python3 pickle file')            
                print('\t - "onnx": saves the model using Open Neural Network Exchange format (ONNX)')            
                print('\t - "pmml": saves the model using Predictive Model Markup Language (PMML)')            
                print('=' * 80)
            else:
                if not self.is_trained:
                    print('=' * 80)
                    print('Model Save Error: model not trained yet, nothing to save.')
                    print('=' * 80)
                else:
                    try:
                        if extension == 'pkl':
                            with open(filename, 'wb') as f:
                                pickle.dump(self, f)
                            print('=' * 80)
                            print('Model saved at %s in pickle format.' %filename)
                            print('=' * 80)
                        elif extension == 'onnx':
                            from sklearn import linear_model
                            from skl2onnx import convert_sklearn # conda install -c conda-forge skl2onnx
                            from skl2onnx.common.data_types import FloatTensorType

                            export_model = linear_model.LogisticRegression()
                            export_model.coef_ = self.w[1:].ravel()
                            NI = export_model.coef_.shape[0]
                            export_model.intercept_ = self.w[0]
                            export_model.classes_ = np.array([0., 1.])

                            # Convert into ONNX format
                            input_type = [('float_input', FloatTensorType([None, NI]))]
                            onnx_model = convert_sklearn(export_model, initial_types=input_type)
                            with open(filename, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            print('=' * 80)
                            print('Model saved at %s in ONNX format.' %filename)
                            print('=' * 80)
                        elif extension == 'pmml':
                            from sklearn import linear_model
                            export_model = linear_model.LogisticRegression()
                            NI = self.w[1:].ravel().shape[0]
                            X = np.random.normal(0, 1, (100, NI))
                            w = np.random.normal(0, 1, (NI, 1))
                            y = ((np.sign(np.dot(X, w)) + 1) / 2.0).ravel()
                            export_model.fit(X, y)
                            export_model.coef_ = self.w[1:].T
                            export_model.intercept_ = self.w[0]
                            from sklearn2pmml import sklearn2pmml # pip install git+https://github.com/jpmml/sklearn2pmml.git
                            from sklearn2pmml.pipeline import PMMLPipeline
                            pipeline = PMMLPipeline([("classifier", export_model)])
                            sklearn2pmml(pipeline, filename, with_repr = True)
                            print('=' * 80)
                            print('Model saved at %s in PMML format.' %filename)
                            print('=' * 80)
                        else:
                            print('=' * 80)
                            print('Model Save Error: model cannot be saved at %s.' %filename)
                            print('=' * 80)
                    except:
                        print('=' * 80)
                        print('Model Save Error: model cannot be saved at %s, please check the provided path/filename.' %filename)
                        print('=' * 80)
                        raise


class LC_Master(Common_to_all_POMs):
    """
    This class implements the Logistic Classifier model, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=False, **kwargs):
        """
        Create a :class:`LC_Master` instance.

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
        
        kwargs: Keyword arguments.

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
        self.all_workers_addresses = [str(x) for x in self.workers_addresses]
        
        self.send_to = {}
        self.receive_from = {}
        for k in range(self.Nworkers):
            self.send_to.update({str(k): workers_addresses[k]})
            self.receive_from.update({workers_addresses[k]: str(k)})

        self.logger = logger                        # logger
        self.comms = comms                          # comms lib
        self.state_dict = None                      # State of the main script
        self.verbose = verbose                      # print on screen when true
        self.NI = None
        self.model = Model()
        self.epsilon = 0.00000001  # to avoid log(0)
        
        self.state_dict = {}                        # dictionary storing the execution state
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
        self.message_counter = 100    # used to number the messages
        self.cryptonode_address = None
        self.newNI_dict = {}
        self.train_data_is_ready = False
        self.grady_dict = {}
        self.s_dict = {}
        self.grads_dict = {}
        self.Ztr_dict = {}
        self.NPtr_dict = {}
        t = time.time()
        seed = int((t - int(t)) * 10000)
        np.random.seed(seed=seed)

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
            State(name='computing_XTw', on_enter=['while_computing_XTw']),
            State(name='computing_oi', on_enter=['while_computing_oi']),

            State(name='updating_w', on_enter=['while_updating_w']),
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_computing_XTw', 'waiting_order', 'computing_XTw'],
            ['go_waiting_order', 'computing_XTw', 'waiting_order'],

            ['go_computing_oi', 'waiting_order', 'computing_oi'],
            ['go_waiting_order', 'computing_oi', 'waiting_order'],


            ['go_updating_w', 'waiting_order', 'updating_w'],
            ['go_waiting_order', 'updating_w', 'waiting_order'],
            ]

        class FSM_master(object):

            self.name = 'FSM_master'

            def while_waiting_order(self, MLmodel):
                MLmodel.display(MLmodel.name + ': WAITING for instructions...')
                return

            def while_update_tr_data(self, MLmodel):
                # Always all, only once
                try:
                    action = 'update_tr_data'
                    data = {}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)
                    MLmodel.comms.broadcast(packet)
                    MLmodel.display(MLmodel.name + ': broadcasted update_tr_data to all Workers')

                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    MLmodel.display('ERROR AT while_update_tr_data')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_computing_XTw(self, MLmodel):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    action = 'computing_XTw'
                    MLmodel.x = MLmodel.model.w.T
                    NItrain = MLmodel.x.shape[1]
                    K = int(NItrain / 2)
                    # Guardar
                    MLmodel.A = np.random.uniform(-10, 10, K).reshape((1, K))
                    MLmodel.C = np.random.uniform(-10, 10, K).reshape((1, K))
                    MLmodel.xa = MLmodel.x[:, 0:K]
                    MLmodel.xb = MLmodel.x[:, K:]
                    # Enviar
                    xa_ = MLmodel.xa + MLmodel.A  
                    xb_ = MLmodel.xb + MLmodel.C
                    P = MLmodel.A + MLmodel.C   # warning, check the sum is nonzero (low prob...)

                    # broadcasts xa_, xb_, P
                    action = 'sending_xaxbP'
                    data = {'xa_': xa_, 'xb_': xb_, 'P': P}
                    del xa_, xb_, P
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    #message_id = MLmodel.master_address + '_' + str(MLmodel.message_counter)
                    #packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address, 'message_id': message_id}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)
                    del data
                    #MLmodel.message_counter += 1
                   
                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': computing_XTw with all Workers')
                    else:
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
                        MLmodel.display(MLmodel.name + ': computing_XTw with Workers: %s' % str(MLmodel.selected_workers))
                
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    MLmodel.display('ERROR AT while_computing_XTw')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_computing_oi(self, MLmodel):

                try:
                    MLmodel.o_dict = {}

                    for addr in MLmodel.workers_addresses:
                        MLmodel.display('PROC_MASTER_START', verbose=False)
                        #MLmodel.display('PROC_MASTER_START', verbose=False)
                        U = MLmodel.s_dict[addr]['ya_'] * (MLmodel.xa + 2 * MLmodel.A) + MLmodel.s_dict[addr]['yb_'] * (MLmodel.xb + 2 * MLmodel.C) + MLmodel.s_dict[addr]['Q'] * (MLmodel.A + 2 * MLmodel.C)
                        u = np.sum(U, axis=1)  
                        del U
                        s = u + MLmodel.s_dict[addr]['v']
                        del u

                        o = MLmodel.sigm(s).reshape((-1, 1))
                          
                        NPtr = o.shape[0]
                        MLmodel.NPtr_dict.update({addr: NPtr})

                        oZtr = np.dot(o.T, MLmodel.Ztr_dict[addr])
                        action = 'sending_oZtr'
                        data = {'oZtr': oZtr}
                        del o
                        MLmodel.display('PROC_MASTER_END', verbose=False)
                         
                        #message_id = MLmodel.master_address + '_' + str(MLmodel.message_counter)
                        #packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address, 'message_id': message_id}
                        packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                        message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                        packet.update({'message_id': message_id})
                        MLmodel.message_counter += 1
                        size_bytes = asizeof.asizeof(dill.dumps(packet))
                        MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, addr, message_id, str(size_bytes)), verbose=False)

                        del data
                        #size_bytes = asizeof.asizeof(dill.dumps(packet))
                        #MLmodel.display('PROC_MASTER_END', verbose=False)
                        #MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, addr, message_id, str(size_bytes)), verbose=False)
                        #MLmodel.message_counter += 1
                        MLmodel.comms.send(packet, MLmodel.send_to[addr])
                        #del packet, size_bytes, message_id
                        del packet
                        MLmodel.display(MLmodel.name + ' %s: sent sending_oZtr to %s' % (str(MLmodel.master_address), str(addr)))
                    
                    del MLmodel.xa, MLmodel.xb, MLmodel.A, MLmodel.C

                except:
                    raise
                    '''
                    print('ERROR AT while_computing_oi')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_updating_w(self, MLmodel):

                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    #self.grads_dict.update({sender: {'ya_': packet['data']['ya_'], 'yb_': packet['data']['Q'], 'Q': packet['data']['Q'], 'v': packet['data']['v']}})
                    #MLmodel.display('PROC_MASTER_START', verbose=False)
                    grad_acum = np.zeros((MLmodel.NItrain, 1))

                    NPtr_train = 0
                    for waddr in MLmodel.workers_addresses:
                        grad_acum += MLmodel.grads_dict[waddr]
                        NPtr_train += MLmodel.NPtr_dict[waddr]

                    grad_acum = grad_acum / NPtr_train                  
                    MLmodel.w_old = np.copy(MLmodel.model.w)

                    if MLmodel.Xval is None:  # A validation set is not provided
                        MLmodel.model.w = MLmodel.model.w - MLmodel.mu * grad_acum
                        del grad_acum
                    else:  # We obtain the optimal update
                        NIval = MLmodel.Xval.shape[1]
                        w_ = MLmodel.model.w[0: NIval + 1]
                        grad_acum_ = grad_acum[0: NIval + 1]

                        CE_val = []
                        mus = np.arange(0.01, 10.0, 0.01)
                        Xw = np.dot(MLmodel.add_bias(MLmodel.Xval), w_)
                        Xgrad = np.dot(MLmodel.add_bias(MLmodel.Xval), grad_acum_)

                        for mu in mus:
                            s_val = Xw - mu * Xgrad
                            o_val = MLmodel.sigm(s_val).ravel()
                            ce_val = np.mean(MLmodel.cross_entropy(o_val, MLmodel.yval, MLmodel.epsilon))
                            CE_val.append(ce_val)

                        del Xw, Xgrad, s_val, o_val

                        min_pos = np.argmin(CE_val)
                        mu_opt = mus[min_pos]
                        del mus
                        MLmodel.model.w = MLmodel.model.w - mu_opt * grad_acum
                        del grad_acum
                        MLmodel.ceval_old = MLmodel.ceval

                        MLmodel.ceval = CE_val[min_pos]
                        del CE_val
                        print('Optimal mu = %f, CE val=%f' % (mu_opt, MLmodel.ceval))

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                except:
                    raise
                    '''
                    print('ERROR AT while_updating_w')
                    import code
                    code.interact(local=locals())
                    '''
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
        self.display('MASTER_INIT', verbose=False)
        self.stop_training = False
        self.kiter = 0

        if not self.train_data_is_ready: 
            self.FSMmaster.go_update_tr_data(self)
            self.run_Master()
            # Checking the new NI values
            print(list(self.newNI_dict.values()))
            # Number of features in training data, without bias
            newNIs = list(set(list(self.newNI_dict.values())))
            if len(newNIs) > 1:
                message = 'ERROR: the training data has different number of features...'
                self.display(message)
                self.display(list(self.newNI_dict.values()))
                raise Exception(message)
            else:
                self.reset(newNIs[0])
                ## Adding bias to validation data, if any
                self.NI = newNIs[0] + 1
                if self.Xval is not None: 
                    self.Xval_b = self.add_bias(self.Xval).astype(float)
                    self.yval = self.yval.astype(float)
            self.train_data_is_ready = True

        # self.broadcast_addresses  direcciones pycloud
        # self.workers_addresses  0->N, active
        # self.all_workers_addresses  0->N all that joined the task
        # self.selected_workers
        self.receivers_list = None
        if self.selected_workers is not None:
            self.workers_addresses = self.selected_workers
        else:
            self.workers_addresses = self.all_workers_addresses[:]

        self.Nworkers  = len(self.workers_addresses) 
        self.state_dict = {}                        # dictionary storing the execution state
        for k in range(0, self.Nworkers):
            self.state_dict.update({self.workers_addresses[k]: ''})
        self.receivers_list=[]
        for worker in self.workers_addresses:
            self.receivers_list.append(self.send_to[worker])

        self.Bob_data_s = False
        self.Bob_data_grad = False

       # Checking dimensions
        if int(self.NI / 2) != self.NI / 2: # add one value 
            self.w_orig_size = self.NI
            self.NItrain = self.NI + 1
        else:
            self.w_orig_size = self.NI
            self.NItrain = self.NI

        self.model.w = np.random.normal(0, 0.0001, (self.NItrain, 1))      # weights in plaintext, first value is bias
        self.w_old = np.random.normal(0, 1.0, (self.NItrain, 1))
        self.ceval = 10
        
        while not self.stop_training:

            self.display('MASTER_ITER_START', verbose=False)

            # We XTw
            self.FSMmaster.go_computing_XTw(self)
            self.run_Master()
            # We receive  self.s_dict, self.Ztr_dict (once)

            # processing outputs from every worker
            self.FSMmaster.go_computing_oi(self)
            self.run_Master()

            # This updates self.w and self.w_old
            self.FSMmaster.go_updating_w(self)
            self.FSMmaster.go_waiting_order(self)

            self.kiter += 1
            # Stop if Maxiter is reached
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            if self.Xval is None:  # A validation set is not provided
                inc_w = np.linalg.norm(self.model.w - self.w_old) / np.linalg.norm(self.w_old)
                # Stop if convergence is reached
                if inc_w < self.conv_stop:
                    self.stop_training = True

                #message = '==================> ' + str(self.regularization) + ', ' + str(self.Nmaxiter) + ', ' + str(self.kiter) + ', ' + str(inc_w)
                message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
                #self.display(message)
                print(message)
            else:
                # stopping
                inc_ceval = np.linalg.norm(self.ceval - self.ceval_old) / np.linalg.norm(self.ceval_old)

                # Stop if convergence is reached
                if inc_ceval < self.conv_stop:
                    self.stop_training = True

                message = 'Maxiter = %d, iter = %d, inc_CE_val = %f' % (self.Nmaxiter, self.kiter, inc_ceval)
                #self.display(message, verbose=True)
                print(message)
            self.display('MASTER_ITER_END', verbose=False)


        self.display(self.name + ': Training is done')
        self.model.niter = self.kiter
        self.model.is_trained = True

        # reduciendo a dimensión original
        self.model.w = self.model.w[0:self.w_orig_size, :]
        self.display('MASTER_FINISH', verbose=False)

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

        if self.chekAllStates('ACK_sending_s'):
            if not self.Bob_data_s:
                self.Bob_data_s = True
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sending_grad'):
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
            try:
                self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass

        if packet['action'] == 'ACK_sending_XTDaX':
            self.XTDaX_dict.update({sender: {'XTDaX': packet['data']['XTDaX'], 'XTDast': packet['data']['XTDast']}})

        if packet['action'] == 'ACK_update_tr_data':
            #print('ProcessReceivedPacket_Master ACK_update_tr_data')
            self.newNI_dict.update({sender: packet['data']['newNI']})

        if packet['action'] == 'ACK_sending_s':
            if not self.Bob_data_s:
                self.s_dict.update({sender: {'ya_': packet['data']['ya_'], 'yb_': packet['data']['yb_'], 'Q': packet['data']['Q'], 'v': packet['data']['v']}})
                self.Ztr_dict.update({sender: packet['data']['Ztr']})
            else:
                self.s_dict[sender]['v'] = packet['data']['v']

        if packet['action'] == 'ACK_sending_grad':
            try:
                self.grads_dict.update({sender: packet['data']['grad']})
            except:
                pass

        return


#===============================================================
#                 Worker
#===============================================================
class LC_Worker(Common_to_all_POMs):
    '''
    Class implementing Logistic Classifier (private model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None):
        """
        Create a :class:`LC_Worker` instance.

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

        self.w = None
        self.epsilon = 0.00000001  # to avoid log(0)
        self.create_FSM_worker()
        self.added_bias=False

        self.Bob_data_s = False
        self.Bob_data_grad = False
        #self.grady = np.mean(self.Xtr_b * self.ytr.reshape((-1, 1)), axis=0).reshape((self.NI, 1))
        #self.grady = np.dot(self.Xtr_b.T, self.ytr.reshape((-1, 1)))

        #self.Cmat = np.random.normal(0, 1, (self.NI, self.NI))
        #self.Dmat = np.linalg.inv(self.Cmat)
        #self.Ztr = np.dot(self.Xtr_b, self.Cmat)
        self.message_counter = 100
        t = time.time()
        seed = int((t - int(t)) * 10000)
        np.random.seed(seed=seed)

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

            def while_setting_tr_data(self, MLmodel, packet):
                try:
                    NPtr = MLmodel.Xtr_b.shape[0]
                    if not MLmodel.added_bias:  # Only ad bias once
                        MLmodel.newNI = MLmodel.Xtr_b.shape[1]
                        MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
                        MLmodel.added_bias = True
                    
                    MLmodel.ytr = MLmodel.ytr.astype(float)
                    MLmodel.st = -np.log(1.0 / (MLmodel.ytr * (1.0 - MLmodel.epsilon) + MLmodel.epsilon * (1.0 - MLmodel.ytr)) - 1.0)
                    action = 'ACK_update_tr_data'
                    data = {'newNI': MLmodel.newNI}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}

                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_update_tr_data' % (str(MLmodel.worker_address)))
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    #raise
                    import code
                    code.interact(local=locals())
                    #MLmodel.display('ERROR AT while_computing_XTDaX')
                    '''

            def while_computing_s(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    #MLmodel.display('PROC_WORKER_START', verbose=False)
                    xa_ = packet['data']['xa_']
                    xb_ = packet['data']['xb_']
                    P = packet['data']['P']

                    # Only once
                    if not MLmodel.Bob_data_s:
                        # Checking NI
                        NI = MLmodel.Xtr_b.shape[1]
                        NPtr = MLmodel.Xtr_b.shape[0]

                        if NI/2 != int(NI/2):
                            MLmodel.Xtr_b = np.hstack((MLmodel.Xtr_b, np.random.normal(0, 0.01, (NPtr, 1))))

                        NPtr_train = MLmodel.Xtr_b.shape[0]
                        NI_train = MLmodel.Xtr_b.shape[1]
                        # RMD
                        MLmodel.Cmat = np.random.normal(0, 1, (NI_train, NI_train))
                        MLmodel.Dmat = np.linalg.inv(MLmodel.Cmat)
                        MLmodel.Ztr = np.dot(MLmodel.Xtr_b, MLmodel.Cmat)

                        K = int(NI_train / 2)

                        y = MLmodel.Xtr_b
                        MLmodel.yas = y[:, 0:K]
                        MLmodel.ybs = y[:, K:]
                        del y

                        MLmodel.Bs = np.random.uniform(-10, 10, (NPtr_train, K))
                        MLmodel.Ds = np.random.uniform(-10, 10, (NPtr_train, K))
                        MLmodel.Qs = MLmodel.Bs - MLmodel.Ds    # warning, check the sum is nonzero (low prob...)
                        MLmodel.ya_s = MLmodel.Bs - MLmodel.yas  
                        MLmodel.yb_s = MLmodel.Ds - MLmodel.ybs

                    V = xa_ * (2 * MLmodel.yas - MLmodel.Bs) + xb_ * (2 * MLmodel.ybs - MLmodel.Ds) + P * (MLmodel.Ds - 2 * MLmodel.Bs)
                    v = np.sum(V, axis=1)
                    del xa_, xb_, P, V

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    # send to Master ya_, yb_, Q, v
                    action = 'ACK_sending_s'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    if not MLmodel.Bob_data_s:
                        data = {'ya_': MLmodel.ya_s, 'yb_': MLmodel.yb_s, 'Q': MLmodel.Qs, 'v': v, 'Ztr': MLmodel.Ztr}
                        MLmodel.Bob_data_s = True
                    else:
                        data = {'v': v}

                    del v



                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    del data

                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    del packet#, size_bytes
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_s' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_s')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_computing_grad(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    #MLmodel.display('PROC_WORKER_START', verbose=False)
                    #oZtr = packet['data']['oZtr']                 
                    #oXtr = np.dot(oZtr, MLmodel.Dmat).T
                    #grad = oXtr - np.dot(MLmodel.Xtr_b.T, MLmodel.ytr.reshape((-1, 1)))
                    grad = np.dot(packet['data']['oZtr'], MLmodel.Dmat).T - np.dot(MLmodel.Xtr_b.T, MLmodel.ytr.reshape((-1, 1)))
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    # Worker sends grad to master
                    action = 'ACK_sending_grad'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    # grady is only sent the first time...
                    #if MLmodel.message_counter == 1:
                    #    data.update({'grady': MLmodel.grady})
                    data = {'grad': grad}
                    #packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address, 'message_id': message_id}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    del data
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    del packet
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_grad' % (str(MLmodel.worker_address)))                   
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_grad')
                    import code
                    code.interact(local=locals())
                    '''

                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='computing_s', on_enter=['while_computing_s']),
            State(name='computing_grad', on_enter=['while_computing_grad']),

            State(name='computing_XTDaX', on_enter=['while_computing_XTDaX']),
            State(name='Exit', on_enter=['while_Exit'])
           ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_computing_s', 'waiting_order', 'computing_s'],
            ['done_computing_s', 'computing_s', 'waiting_order'],

            ['go_computing_grad', 'waiting_order', 'computing_grad'],
            ['done_computing_grad', 'computing_grad', 'waiting_order'],

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
        try:
            self.display('COMMS_WORKER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
        except:
            self.display('WORKER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
            pass

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
            self.terminate = True

        if packet['action'] == 'update_tr_data':
            # We update the training data
            self.FSMworker.go_setting_tr_data(self, packet)
            self.FSMworker.done_setting_tr_data(self)

        if packet['action'] == 'sending_xaxbP':
            self.FSMworker.go_computing_s(self, packet)
            self.FSMworker.done_computing_s(self)

        if packet['action'] == 'sending_oZtr':
            self.FSMworker.go_computing_grad(self, packet)
            self.FSMworker.done_computing_grad(self)

        '''
        if packet['action'] == 'sending_w':
            # We update the model weights
            self.w = packet['data']['w']
            self.FSMworker.go_computing_XTDaX(self, packet)
            self.FSMworker.done_computing_XTDaX(self)
        '''


        return self.terminate
