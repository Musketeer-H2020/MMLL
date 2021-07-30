# -*- coding: utf-8 -*-
'''
Logistic Classifier model under POM5

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Febr. 2021"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
import pickle
from pympler import asizeof #asizeof.asizeof(my_object)
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
        X: ndarray
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

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
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
        self.pom = 5
        self.model_type = model_type
        self.name = 'POM%d_' % self.pom + self.model_type + '_Master'                 # Name
        #self.NC = NC                                # No. Centroids
        #self.Nmaxiter = Nmaxiter
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
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.model = Model()
        #self.regularization = regularization
        #self.classes = classes
        #self.balance_classes = balance_classes
        #self.Xval_b = Xval_b
        #self.yval = yval
        self.epsilon = 0.00000001  # to avoid log(0)
        self.momentum = 0

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
        self.message_counter = 100    # used to number the messages
        self.XTX_dict = {}
        self.XTy_dict = {}
        self.encrypter = self.cr.get_encrypter()  # to be shared        # self.encrypter.encrypt(np.random.normal(0, 1, (2,3)))
        self.decrypter = self.cr.get_decrypter()  # to be kept as secret  self.encrypter.decrypt()
        self.create_FSM_master()
        self.FSMmaster.master_address = master_address
        self.newNI_dict = {}
        self.train_data_is_ready = False
        self.encrypter_sent = False
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
            State(name='send_w_encr', on_enter=['while_send_w_encr']),
            State(name='crypto_loop', on_enter=['while_crypto_loop']),
            State(name='comp_exp_bl', on_enter=['while_comp_exp_bl']),
            State(name='comp_div_bl', on_enter=['while_comp_div_bl']),
         ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_send_w_encr', 'waiting_order', 'send_w_encr'],
            ['done_send_w_encr', 'send_w_encr', 'waiting_order'],

            ['go_crypto_loop', 'waiting_order', 'crypto_loop'],
            ['done_crypto_loop', 'crypto_loop', 'waiting_order'],

            ['go_comp_exp_bl', 'crypto_loop', 'comp_exp_bl'],
            ['done_comp_exp_bl', 'comp_exp_bl', 'crypto_loop'],

            ['go_comp_div_bl', 'crypto_loop', 'comp_div_bl'],
            ['done_comp_div_bl', 'comp_div_bl', 'crypto_loop']

        ]

        class FSM_master(object):

            self.name = 'FSM_master'

            def while_waiting_order(self, MLmodel):
                '''
                try:
                    MLmodel.display(MLmodel.name + ' is waiting...')
                except:
                    print('ERROR AT while_waiting_order')
                    import code
                    code.interact(local=locals())
                '''
                return

            def while_update_tr_data(self, MLmodel):
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

            def while_send_w_encr(self, MLmodel):
                try:
                    data = {}
                    data.update({'w_encr': MLmodel.w_encr})
                    #wdill = MLmodel.dill_it(MLmodel.wq_encr)
                    #data.update({'wq_encr': wdill})

                    action = 'send_w_encr'
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': broadcasted w to all Workers')
                    else:
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
                        MLmodel.display(MLmodel.name + ': broadcasted w to Workers: %s' % str(MLmodel.selected_workers))
                except:
                    raise
                    '''
                    print('ERROR AT while_send_w_encr')
                    import code
                    code.interact(local=locals())
                    '''

                return

            def while_Exit(self, MLmodel):
                #print('while_Exit')
                return

            def while_crypto_loop(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' ------ Master at crypto_loop  ------------ ')
                except:
                    raise
                    '''
                    print('ERROR AT while_crypto_loop')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_comp_exp_bl(self, MLmodel, packet):
                try:

                    MLmodel.display(MLmodel.name + ' is computing exp(-s)...')
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    s_encr_bl = packet['data']['s_encr_bl']
                    sender = packet['sender']
                    del packet

                    s_bl = MLmodel.decrypter.decrypt(s_encr_bl)
                    exp_s_bl = np.exp(-s_bl)
                    exp_s_bl_encr = MLmodel.encrypter.encrypt(exp_s_bl)
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ACK_sent_exp_s_bl_encr'
                    data = {'exp_s_bl_encr': exp_s_bl_encr}
                    del exp_s_bl_encr
                    message_id = MLmodel.master_address + '_' + str(MLmodel.message_counter)
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.master_address}
                    del data

                    destination = sender
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    #MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)
                    worker_name = MLmodel.receive_from[sender]
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, worker_name, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, destination)
                    del packet
                    MLmodel.display(MLmodel.name + ': sent %s' % action)
                except:
                    raise
                    '''
                    print('ERROR AT while_comp_exp_bl')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_comp_div_bl(self, MLmodel, packet):
                try:

                    MLmodel.display(MLmodel.name + ' is computing blinded division...')
                    num_bl = packet['data']['num_bl']
                    den_bl_encr = packet['data']['den_bl_encr']
                    sender = packet['sender']
                    del packet
                                        
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    
                    den_bl = MLmodel.decrypter.decrypt(den_bl_encr)
                    sigm_bl = num_bl / den_bl
                    sigm_encr_bl = MLmodel.encrypter.encrypt(sigm_bl)
                    
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ACK_sent_sigm_encr_bl'
                    data = {'sigm_encr_bl': sigm_encr_bl}
                    del sigm_encr_bl
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.master_address}
                    del data

                    destination = sender
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    #MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)
                    worker_name = MLmodel.receive_from[sender]
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, worker_name, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, destination)
                    del packet
                    MLmodel.display(MLmodel.name + ': sent %s' % action)
                except:
                    raise
                    '''
                    print('ERROR AT while_comp_div_bl')
                    import code
                    code.interact(local=locals())
                    '''
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
        self.display(self.name + ': Starting training', verbose=True)
        self.display('MASTER_INIT', verbose=False)

        self.NI = self.input_data_description['NI']
        self.w = np.random.normal(0, 0.1, (self.NI + 1, 1))
        self.w_encr = self.encrypter.encrypt(self.w)
        self.grad_old = np.zeros((self.NI + 1, 1))

        if not self.train_data_is_ready: 
            self.FSMmaster.go_update_tr_data(self)
            self.run_Master()
            # Checking the new NI values
            print(list(self.newNI_dict.values()))
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

        self.grads_dict = {}
        self.NP_dict = {}

        self.stop_training = False
        kiter = 0
        ceval = 1000
        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)
            self.FSMmaster.go_send_w_encr(self)
            self.FSMmaster.done_send_w_encr(self)

            # FSMaster en estado "crypto_loop", desde aquí puede responder a los workers
            # Los workers comienzan cómputo y pueden pedir operaciones al Master
            self.FSMmaster.go_crypto_loop(self)
            # El Master espera a servir de cryptoprocessor. Cuando tenga todos los ACK_grads, sale y sigue
            self.run_Master()

            self.display('PROC_MASTER_START', verbose=False)

            grad = np.zeros((self.NI + 1, 1))
            NPtotal = 0
            for key in self.grads_dict:
                grad_encr = self.grads_dict[key]
                #gradq = self.cr.vmasterDec_BCP(gradq_encr, self.cr.PK)
                #grad_decr = self.cr.vQinv_m(gradq, gradq_encr[0, 0].N)
                grad_decr = self.decrypter.decrypt(grad_encr)
                grad += grad_decr
                #NPtotal += self.NP_dict[key]

            #grad = grad / self.Nworkers
            grad = self.mu * grad / len(self.workers_addresses)

            self.w_old = self.w.copy()
            ceval_old = ceval

            if self.Xval is not None:
                self.Xval = None
                self.display('Warning: Validation set is not used during training', verbose=True)
            
            if self.Xval is None:  # A validation set is not provided
                self.w_old = self.w.copy()

                # Moment update
                momentum_term = self.momentum * self.grad_old
                v = momentum_term +  grad
                self.w = self.w - v 
                #self.w += self.mu * grad
                self.grad_old = np.copy(grad)

                #self.w -= self.mu * grad
                #del grad
                # stopping
                inc_w = np.linalg.norm(self.w - self.w_old) / np.linalg.norm(self.w_old)

                # stopping
                inc_w = np.linalg.norm(self.w - self.w_old) / np.linalg.norm(self.w_old)
                #print(0.001, inc_w)
                # Stop if convergence is reached
                if inc_w < self.conv_stop:
                    self.stop_training = True

                message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, kiter, inc_w)
                #self.display(message, verbose=True)
                print(message)

            else:
                NIval = self.Xval.shape[1]
                w_ = self.w[0: NIval + 1]
                grad_acum_ = grad[0: NIval + 1]

                CE_val = []
                mus = np.arange(0.01, 10.0, 0.01)
                Xw = np.dot(self.add_bias(self.Xval), w_)
                Xgrad = np.dot(self.add_bias(self.Xval), grad_acum_)

                for mu in mus:
                    s_val = Xw - mu * Xgrad
                    o_val = self.sigm(s_val).ravel()
                    ce_val = np.mean(self.cross_entropy(o_val, self.yval, self.epsilon))
                    CE_val.append(ce_val)

                del Xw, Xgrad, s_val, o_val

                min_pos = np.argmin(CE_val)
                mu_opt = mus[min_pos]
                del mus
                self.w = self.w - mu_opt * grad
                del grad
                ceval = CE_val[min_pos]
                del CE_val
                print('Optimal mu = %f, CE val=%f' % (mu_opt, ceval))

                # stopping
                inc_ceval = np.linalg.norm(ceval - ceval_old) / np.linalg.norm(ceval_old)

                # Stop if convergence is reached
                if inc_ceval < self.conv_stop:
                    self.stop_training = True

                message = 'Maxiter = %d, iter = %d, inc_CE_val = %f' % (self.Nmaxiter, kiter, inc_ceval)
                #self.display(message, verbose=True)
                print(message)

            if kiter == self.Nmaxiter:
                self.stop_training = True

            kiter += 1
            #print(self.w)

            #self.wq = self.cr.vQ(self.w)
            #self.wq_encr = self.cr.vEncrypt(self.wq)
            self.w_encr = self.encrypter.encrypt(self.w)
            self.display('PROC_MASTER_END', verbose=False)
            self.display('MASTER_ITER_END', verbose=False)

        self.model.w = self.w
        self.display(self.name + ': Training is done', verbose=True)
        self.model.niter = kiter
        self.model.is_trained = True
        self.display('MASTER_FINISH', verbose=False)

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_grads'):
            self.FSMmaster.done_crypto_loop(self)

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
            #sender = self.receive_from[sender]
            self.display(self.name + ': received %s from worker %s' % (packet['action'], sender), verbose=True)
            try:
                self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass
                
            if packet['action'][0:3] == 'ACK':
                self.state_dict[sender] = packet['action']

            if packet['action'] == 'ask_exp_bl':
                self.FSMmaster.go_comp_exp_bl(self, packet)
                self.FSMmaster.done_comp_exp_bl(self)

            if packet['action'] == 'ask_div_bl':
                self.FSMmaster.go_comp_div_bl(self, packet)
                self.FSMmaster.done_comp_div_bl(self)

            if packet['action'] == 'ACK_grads':
                self.grads_dict.update({sender: packet['data']['grad_encr']})
                self.NP_dict.update({sender: packet['data']['NP']})

            if packet['action'] == 'ACK_update_tr_data':
                #print('ProcessReceivedPacket_Master ACK_update_tr_data')
                self.newNI_dict.update({sender: packet['data']['newNI']})
        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Master')
            import code
            code.interact(local=locals())
            '''        
        return


#===============================================================
#                 Worker
#===============================================================
class LC_Worker(Common_to_all_POMs):
    '''
    Class implementing Logistic Classifier (private model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
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
        self.pom = 5
        self.master_address = master_address
        self.worker_address = worker_address                    # The id of this Worker
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.model_type = model_type
        self.comms = comms                      # The comms library
        self.logger = logger                    # logger
        self.name = model_type + '_Worker'    # Name
        self.verbose = verbose                  # print on screen when true
        #self.Xtr_b = self.add_bias(Xtr_b)
        #self.ytr = ytr
        #self.NPtr = len(ytr)
        self.create_FSM_worker()
        self.message_id = 0    # used to number the messages
        self.message_counter = 100 # used to number the messages
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

            def while_waiting_order(self, MLmodel):
                MLmodel.display(self.name + ' %s: WAITING for instructions...' % (str(MLmodel.worker_address)))
                return

            def while_setting_tr_data(self, MLmodel, packet):
                try:
                    NPtr, newNI = MLmodel.Xtr_b.shape
                    MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
                    MLmodel.ytr = MLmodel.ytr.astype(float)
                    action = 'ACK_update_tr_data'
                    data = {'newNI': newNI}
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
                    import code
                    code.interact(local=locals())
                    #MLmodel.display('ERROR AT while_computing_XTDaX')
                    '''

            # Split this in two, for clarity???
            def while_compute_states(self, MLmodel, packet):
                try:
                    check = False
                    MLmodel.display(MLmodel.name + ' %s: computing states...' % (str(MLmodel.worker_address)))
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    w_encr = packet['data']['w_encr']
                    del packet
                    X = MLmodel.Xtr_b

                    y = MLmodel.ytr.reshape(-1, 1)
                    MLmodel.s_encr = np.dot(X, w_encr)
                    del X

                    if check:
                        MLmodel.w = MLmodel.decrypter.decrypt(w_encr)
                        MLmodel.s = MLmodel.decrypter.decrypt(MLmodel.s_encr)
                        MLmodel.s_ok = np.dot(MLmodel.Xtr_b, MLmodel.w)
                        e = np.linalg.norm(MLmodel.s_ok - MLmodel.s)
                        print('Error in s', e)

                    # Añadiendo blinding y enviado a Master para calcular exp(-s)
                    NP = MLmodel.s_encr.shape[0]
                    MLmodel.gamma = np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))       
                    #MLmodel.gamma = np.zeros((NP, 1)).reshape((-1, 1))       

                    MLmodel.s_encr_bl = (MLmodel.s_encr + MLmodel.gamma)
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_exp_bl'
                    data = {'s_encr_bl': MLmodel.s_encr_bl}
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}

                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)
                    
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.worker_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT while_compute_states')
                    import code
                    code.interact(local=locals())
                    '''
                return


            def while_compute_exp(self, MLmodel, packet):
                try:
                    check = False
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    exp_s_bl_encr = packet['data']['exp_s_bl_encr']
                    NP = exp_s_bl_encr.shape[0]
                    del packet

                    # We deblind exp_s_bl_encr
                    exp_s_encr = exp_s_bl_encr * np.exp(MLmodel.gamma)

                    if check:
                        exp_s_ok = np.exp(-MLmodel.s)
                        exp_s_encr_decr = MLmodel.decrypter.decrypt(exp_s_encr)
                        e1 = np.linalg.norm(exp_s_encr_decr - exp_s_ok)
                        print('Error in exp_s', e1)

                    # creating num, den
                    MLmodel.r1 = np.random.uniform(2, 6, (NP, 1)).reshape((-1, 1))
                    MLmodel.r2 = np.random.uniform(2, 6, (NP, 1)).reshape((-1, 1))
                    #MLmodel.r1 = np.ones((NP, 1)).reshape((-1, 1))
                    #MLmodel.r2 = np.ones((NP, 1)).reshape((-1, 1))

                    MLmodel.num_bl = MLmodel.r1
                    MLmodel.den_bl_encr = (1 + exp_s_encr) * MLmodel.r2
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_div_bl'
                    data = {'num_bl': MLmodel.num_bl, 'den_bl_encr': MLmodel.den_bl_encr } 
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}
                    del data
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s' % (str(MLmodel.worker_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT while_compute_exp')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_compute_gradients(self, MLmodel, packet):
                try:
                    check = False
                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    sigm_encr_bl = packet['data']['sigm_encr_bl']
                    del packet
                    #MLmodel.display('PROC_WORKER_START', verbose=False)
                    # recibidos sigm_encr_bl y calculamos sigmoide y gradientes
                    sigm_encr = sigm_encr_bl / MLmodel.r1 * MLmodel.r2

                    if check:
                        sigm_ok = 1 / (1 + np.exp(-MLmodel.s))
                        sigm_encr_decr = MLmodel.decrypter.decrypt(sigm_encr)
                        e1 = np.linalg.norm(sigm_encr_decr - sigm_ok)
                        print('Error in sigma', e1)
               
                    MLmodel.display(MLmodel.name + ' %s: computing gradients...' % (str(MLmodel.worker_address)))
                    X = MLmodel.Xtr_b
                    NP = X.shape[0]
                    NI = X.shape[1]                    
                    y = MLmodel.ytr.reshape(NP, 1)
                    e_encr = sigm_encr - y 

                    Xe_encr = X * e_encr
                    del X, y
                    grad_encr = np.mean(Xe_encr, axis=0).reshape((NI, 1))
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_grads'
                    data = {'grad_encr': grad_encr, 'NP': NP}
                    del grad_encr
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}
                    del data

                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    del packet
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_grads' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_compute_gradients')
                    import code
                    code.interact(local=locals())
                    '''
                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='compute_states', on_enter=['while_compute_states']),
            State(name='compute_exp', on_enter=['while_compute_exp']),
            State(name='compute_gradients', on_enter=['while_compute_gradients']),
            State(name='storing_Pk', on_enter=['while_storing_Pk'])
            ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_compute_states', 'waiting_order', 'compute_states'],
            ['go_compute_exp', 'compute_states', 'compute_exp'],
            ['go_compute_gradients', 'compute_exp', 'compute_gradients'],
            ['done_compute_gradients', 'compute_gradients', 'waiting_order'],

            ['go_storing_Pk', 'waiting_order', 'storing_Pk'],
            ['done_storing_Pk', 'storing_Pk', 'waiting_order']
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

        try:
            # Exit the process
            if packet['action'] == 'STOP':
                self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
                self.terminate = True
            
            if packet['action'] == 'update_tr_data':
                # We update the training data
                self.FSMworker.go_setting_tr_data(self, packet)
                self.FSMworker.done_setting_tr_data(self)

            if packet['action'] == 'send_Pk':
                self.FSMworker.go_storing_Pk(self, packet)
                self.FSMworker.done_storing_Pk(self)

            if packet['action'] == 'send_w_encr':
                self.FSMworker.go_compute_states(self, packet)

            if packet['action'] == 'ACK_sent_exp_s_bl_encr':
                self.FSMworker.go_compute_exp(self, packet)
                #self.FSMworker.done_compute_s3(self)

            if packet['action'] == 'ACK_sent_sigm_encr_bl':
                self.FSMworker.go_compute_gradients(self, packet)
                self.FSMworker.done_compute_gradients(self)

        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Worker')
            import code
            code.interact(local=locals())
            '''
        return self.terminate
