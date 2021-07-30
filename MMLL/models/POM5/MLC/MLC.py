# -*- coding: utf-8 -*-
'''
Multiclass Logistic Classifier model under POM5

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
    Multiclass Logistic Classifier model.
    """
    def __init__(self):
        self.w_dict = None
        self.classes = None
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
        X_b = np.hstack((np.ones((X.shape[0], 1)), X)).astype(float)

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
                            export_model = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
                            NI = self.w_dict[self.classes[0]][1:].ravel().shape[0]
                            X = np.random.normal(0, 1, (100, NI))
                            y = np.array((self.classes * 100)[0: 100]).ravel()
                            #y_ = np.copy(y)
                            #for i, cla in enumerate(self.classes):
                            #    y_[y_ == cla] = i
                            #y_ = y_.astype(int)       
                            export_model.fit(X, y)
                            W = []
                            for cla in self.classes:
                                W.append(self.w_dict[cla].ravel())

                            W = np.array(W)

                            export_model.coef_ = W[:, 1:]
                            export_model.intercept_ = W[:, 0].ravel()

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
                            export_model = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
                            NI = self.w_dict[self.classes[0]][1:].ravel().shape[0]
                            X = np.random.normal(0, 1, (100, NI))
                            y = np.array((self.classes * 100)[0: 100]).ravel()
                            #y_ = np.copy(y)
                            #for i, cla in enumerate(self.classes):
                            #    y_[y_ == cla] = i
                            #y_ = y_.astype(int)       
                            export_model.fit(X, y)
                            W = []
                            for cla in self.classes:
                                W.append(self.w_dict[cla].ravel())

                            W = np.array(W)

                            export_model.coef_ = W[:, 1:]
                            export_model.intercept_ = W[:, 0].ravel()

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


class MLC_Master(Common_to_all_POMs):
    """
    This class implements the Multiclass Logistic Classifier model, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`MLC_Master` instance.

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
            State(name='send_w_encr', on_enter=['while_send_w_encr']),
            State(name='crypto_loop', on_enter=['while_crypto_loop']),
            State(name='comp_exp_bl', on_enter=['while_comp_exp_bl']),
            State(name='comp_div_bl', on_enter=['while_comp_div_bl']),

            State(name='updating_w', on_enter=['while_updating_w']),

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
            ['done_comp_div_bl', 'comp_div_bl', 'crypto_loop'],

            ['go_updating_w', 'waiting_order', 'updating_w'],
            ['go_waiting_order', 'updating_w', 'waiting_order']

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
                    data.update({'w_encr': MLmodel.w_encr_dict, 'classes': MLmodel.classes})
                    # WARNING
                    #data.update({'decrypter': MLmodel.decrypter})

                    action = 'send_w_encr'
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': broadcasted encrypted weights to all Workers')
                    else:
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
                        MLmodel.display(MLmodel.name + ': broadcasted encrypted weights to Workers: %s' % str(MLmodel.selected_workers))
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
                    s_encr_bl_dict = packet['data']['s_encr_bl']
                    sender = packet['sender']
                    del packet

                    #MLmodel.display('PROC_MASTER_START', verbose=False)
                    exp_s_bl_encr_dict = {}

                    for cla in MLmodel.classes:
                        s_bl = MLmodel.decrypter.decrypt(s_encr_bl_dict[cla])
                        exp_s_bl = np.exp(-s_bl)
                        '''
                        print('-----------------')
                        print(np.max(exp_s_bl))
                        print(np.min(exp_s_bl))                        
                        print('-----------------')
                        '''
                        exp_s_bl_encr = MLmodel.encrypter.encrypt(exp_s_bl)
                        exp_s_bl_encr_dict.update({cla: exp_s_bl_encr})

                    del s_encr_bl_dict, s_bl, exp_s_bl_encr

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ACK_sent_exp_s_bl_encr'
                    data = {'exp_s_bl_encr': exp_s_bl_encr_dict}
                    del exp_s_bl_encr_dict
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

                    # send back
                    #MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)
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

                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    MLmodel.display(MLmodel.name + ' is computing blinded division...')
                    num_bl_dict = packet['data']['num_bl']
                    den_bl_encr_dict = packet['data']['den_bl_encr']
                    sender = packet['sender']
                    del packet
                    
                    sigm_encr_bl_dict = {}
                    for cla in MLmodel.classes:
                        den_bl = MLmodel.decrypter.decrypt(den_bl_encr_dict[cla])
                        sigm_bl = num_bl_dict[cla] / den_bl
                        sigm_encr_bl_dict.update({cla: MLmodel.encrypter.encrypt(sigm_bl)})

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ACK_sent_sigm_encr_bl'
                    data = {'sigm_encr_bl': sigm_encr_bl_dict}
                    del sigm_encr_bl_dict
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

            def while_updating_w(self, MLmodel):

                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    #MLmodel.w_old_dict = dict(MLmodel.model.w_dict)
                    MLmodel.ce_val_old = MLmodel.ce_val
                    MLmodel.ce_val = 0

                    for cla in MLmodel.classes:
                        grad_acum = np.zeros((MLmodel.NI + 1, 1))
                        #NPtr_train = 0
                        for waddr in MLmodel.workers_addresses:
                            grad_encr = MLmodel.grads_dict[waddr][cla]
                            grad_decr = MLmodel.decrypter.decrypt(grad_encr)
                            grad_acum += grad_decr
                            #NPtr_train += MLmodel.NP_dict[waddr][cla]

                        #grad_acum = grad_acum / NPtr_train                  
                        grad_acum = MLmodel.mu * grad_acum / len(MLmodel.workers_addresses)                  

                        MLmodel.Xval = None # Only this option
                        if MLmodel.Xval is None:  # A validation set is not provided
                            #MLmodel.model.w_dict[cla] = MLmodel.model.w_dict[cla] - MLmodel.mu * grad_acum
                            # Momentum
                            v_1 = np.copy(MLmodel.grad_old_dict[cla]) # old gradient
                            momentum = MLmodel.momentum * v_1
                            v = momentum + grad_acum
                            MLmodel.model.w_dict[cla] = MLmodel.model.w_dict[cla] - v
                            MLmodel.grad_old_dict[cla] = grad_acum
                            

                            # We update the encrypted version
                            MLmodel.w_encr_dict[cla] =  MLmodel.encrypter.encrypt(MLmodel.model.w_dict[cla])
                            del grad_acum, grad_encr, grad_decr

                        else:  # We obtain the optimal update for every class
                            NIval = MLmodel.Xval.shape[1]
                            w_ = MLmodel.model.w_dict[cla][0: NIval + 1]
                            grad_acum_ = grad_acum[0: NIval + 1]

                            CE_val = []
                            mus = np.arange(-0.52, 10.0, 0.1)
                            Xw = np.dot(MLmodel.add_bias(MLmodel.Xval), w_)
                            Xgrad = np.dot(MLmodel.add_bias(MLmodel.Xval), grad_acum_)
                            yval = np.array(MLmodel.yval == cla).astype(float).reshape((-1, 1))

                            for mu in mus:
                                s_val = Xw - mu * Xgrad
                                o_val = MLmodel.sigm(s_val).ravel()
                                ce_val = np.mean(MLmodel.cross_entropy(o_val, yval, MLmodel.epsilon))
                                CE_val.append(ce_val)

                            del Xw, Xgrad, s_val, o_val

                            min_pos = np.argmin(CE_val)
                            mu_opt = mus[min_pos]
                            del mus
                            MLmodel.model.w_dict[cla] = MLmodel.model.w_dict[cla] - mu_opt * grad_acum
                            del grad_acum
                            ceval = CE_val[min_pos]
                            MLmodel.ce_val += ceval
                            del CE_val
                            print('Optimal mu = %f for class=%s, CE val=%f' % (mu_opt, cla, ceval))

                            MLmodel.ce_val = MLmodel.ce_val / len(MLmodel.classes)

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                except:
                    raise
                    '''
                    print('ERROR AT while_updating_w')
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
        #self.w_decr = np.random.normal(0, 0.001, (self.NI + 1, 1))      # weights in plaintext, first value is bias
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
                    #self.yval = self.yval.astype(float)
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

        self.display('=============================================== ', True)
        self.display('===========  List of participants  ============ ', True)
        for worker in self.workers_addresses:
            message = '%s - %s' % (worker, self.send_to[worker])
            self.display(message, True)
        self.display('=============================================== ', True)


        self.grads_dict = {}
        self.NP_dict = {}

        self.model.w_dict = {}
        self.w_encr_dict = {}
        self.w_old_dict = {}
        self.grad_old_dict = {}
        self.model.classes = self.classes

        self.Xval = None # Do not use Xval
        
        for cla in self.classes:
            self.model.w_dict.update({cla: np.random.normal(0, 0.001, (self.NI + 1, 1))})
            self.w_encr_dict.update({cla: self.encrypter.encrypt(self.model.w_dict[cla])})
            self.w_old_dict.update({cla: np.random.normal(0, 1.0, (self.NI + 1, 1))})
            self.grad_old_dict.update({cla: np.random.normal(0, 0.001, (self.NI + 1, 1))})

        self.ce_val = 10

        self.stop_training = False
        self.kiter = 0
        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)

            self.FSMmaster.go_send_w_encr(self)
            self.FSMmaster.done_send_w_encr(self)

            # FSMaster en estado "crypto_loop", desde aquí puede responder a los workers
            # Los workers comienzan cómputo y pueden pedir operaciones al Master
            self.FSMmaster.go_crypto_loop(self)
            # El Master espera a servir de cryptoprocessor. Cuando tenga todos los ACK_grads, sale y sigue
            self.run_Master()

            # This updates self.w and self.w_old
            self.FSMmaster.go_updating_w(self)
            self.FSMmaster.go_waiting_order(self)

            self.kiter += 1
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            self.display('PROC_MASTER_START', verbose=False)

            inc_w = 0
            for cla in self.classes:
                inc_w += np.linalg.norm(self.model.w_dict[cla] - self.w_old_dict[cla]) / np.linalg.norm(self.w_old_dict[cla])
                self.w_old_dict[cla] = np.copy(self.model.w_dict[cla])

            if self.Xval is None:  # A validation set is not provided
                # Stop if convergence is reached
                if inc_w < self.conv_stop:
                    self.stop_training = True               

                message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
                self.display(message, verbose=False)
                print(message)
            else:
                print(self.ce_val, self.ce_val_old)
                
                if self.ce_val < self.ce_val_old:  
                    message = 'Maxiter = %d, iter = %d, CE val = %f' % (self.Nmaxiter, self.kiter, self.ce_val)
                    print(message)
                    self.w_old_dict = dict(self.model.w_dict)
                else:
                    self.stop_training = True
                    # We retain the last weight values
                    self.model.w_dict = dict(self.w_old_dict)

            self.display('PROC_MASTER_END', verbose=False)
            self.display('MASTER_ITER_END', verbose=False)

        self.display(self.name + ': Training is done', verbose=True)
        self.model.niter = self.kiter
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
            print('=================   ALL GRADS   =================')
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
            self.display(self.name + ': received %s from worker %s' % (packet['action'], sender), verbose=True)
            try:
                self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass

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

            #sender = self.receive_from[sender]
            if packet['action'][0:3] == 'ACK':
                self.state_dict[sender] = packet['action']

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
class MLC_Worker(Common_to_all_POMs):
    '''
    Class implementing Logistic Classifier (private model), run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`MLC_Worker` instance.

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
                    #MLmodel.ytr = MLmodel.ytr.astype(float)
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

            def while_compute_states(self, MLmodel, packet):
                try:
                    check = False
                    if check: 
                        MLmodel.decrypter = packet['data']['decrypter']

                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    MLmodel.classes = packet['data']['classes']
                    MLmodel.w_encr_dict = packet['data']['w_encr']
                    del packet

                    #y = MLmodel.ytr.reshape(-1, 1)                   
                    MLmodel.display(MLmodel.name + ' %s: computing states...' % (str(MLmodel.worker_address)))
                    
                    #print('MODEL --------------------------------')
                    MLmodel.s_encr_dict = {}
                    for cla in MLmodel.classes:
                        MLmodel.s_encr_dict.update({cla: np.dot(MLmodel.Xtr_b, MLmodel.w_encr_dict[cla])})
                        #print(list(MLmodel.decrypter.decrypt(MLmodel.w_encr_dict[cla]).ravel()))

                    if check:
                        for cla in MLmodel.classes: 
                            w = MLmodel.decrypter.decrypt(MLmodel.w_encr_dict[cla])
                            s = MLmodel.decrypter.decrypt(MLmodel.s_encr_dict[cla])
                            s_ok = np.dot(MLmodel.Xtr_b, w)
                            e = np.linalg.norm(s_ok - s)
                            print('Error in s', e)

                    # Añadiendo blinding y enviado a Master para calcular exp(-s)
                    NP = MLmodel.Xtr_b.shape[0]
                    MLmodel.gamma_dict = {}
                    MLmodel.s_encr_bl_dict = {}
                    for cla in MLmodel.classes:
                        # WARNING 0*                    
                        MLmodel.gamma_dict.update({cla: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1)) })                 
                        MLmodel.s_encr_bl_dict.update({cla: MLmodel.s_encr_dict[cla] + MLmodel.gamma_dict[cla]})
                    
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_exp_bl'
                    data = {'s_encr_bl': MLmodel.s_encr_bl_dict}
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

                    exp_s_bl_encr_dict = packet['data']['exp_s_bl_encr']
                    del packet

                    # Keep theese two
                    MLmodel.r1_dict = {} # equal to num_bl_dict = {}
                    MLmodel.r2_dict = {}

                    den_bl_encr_dict = {}
                    for cla in MLmodel.classes:                    
                        # We deblind exp_s_bl_encr
                        exp_s_encr = exp_s_bl_encr_dict[cla] * np.exp(MLmodel.gamma_dict[cla])

                        if check:
                                w = MLmodel.decrypter.decrypt(MLmodel.w_encr_dict[cla])
                                s_ok = np.dot(MLmodel.Xtr_b, w)
                                exp_s_ok = np.exp(-s_ok)
                                exp_s = MLmodel.decrypter.decrypt(exp_s_encr)
                                e = np.linalg.norm(exp_s_ok - exp_s)
                                print('Error in exp(-s)', e)

                        # creating num, den
                        NP = exp_s_bl_encr_dict[cla].shape[0]
                        # WARNING 1   
                        MLmodel.r1_dict.update({cla: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})
                        MLmodel.r2_dict.update({cla: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})
                        #MLmodel.r1_dict.update({cla: np.ones((NP, 1)).reshape((-1, 1))})
                        #MLmodel.r2_dict.update({cla: np.ones((NP, 1)).reshape((-1, 1))})
                        try:
                            den_bl_encr_dict.update({cla: (1.0 + exp_s_encr) * MLmodel.r2_dict[cla]})
                        except:
                            print('ERROR: number too small')
                            import code
                            code.interact(local=locals())


                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_div_bl'
                    data = {'num_bl': MLmodel.r1_dict, 'den_bl_encr': den_bl_encr_dict}
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}
                    del data, den_bl_encr_dict
                    
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
                    sigm_encr_bl_dict = packet['data']['sigm_encr_bl']
                    del packet
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    # recibidos sigm_encr_bl y calculamos sigmoide y gradientes
                    
                    MLmodel.display(MLmodel.name + ' %s: computing gradients...' % (str(MLmodel.worker_address)))
                    
                    grad_encr_dict = {}
                    NP_dict = {}
                    for cla in MLmodel.classes:                    
                        sigm_encr = sigm_encr_bl_dict[cla] / MLmodel.r1_dict[cla] * MLmodel.r2_dict[cla]
                        ytr = ((np.array(MLmodel.ytr == cla)).astype(float)).reshape((-1, 1))

                        if check:
                            w = MLmodel.decrypter.decrypt(MLmodel.w_encr_dict[cla])
                            s_ok = np.dot(MLmodel.Xtr_b, w)
                            exp_s_ok = np.exp(-s_ok)
                            sigm_ok = 1.0 / (1.0 + exp_s_ok)
                            sigm = MLmodel.decrypter.decrypt(sigm_encr)
                            e = np.linalg.norm(sigm_ok - sigm)
                            print('Error in sigm(s)', e)
               
                        NP_dict.update({cla: MLmodel.Xtr_b.shape[0]})
                        e_encr = sigm_encr - ytr
                        grad = np.dot(MLmodel.Xtr_b.T, e_encr).reshape((-1, 1))

                        #Xe_encr = MLmodel.Xtr_b * e_encr
                        #grad = np.sum(Xe_encr, axis=0).reshape((-1, 1))
                        
                        grad_encr_dict.update({cla: grad / MLmodel.Xtr_b.shape[0]}) 

                        if check:
                            e_ok =  sigm_ok - ytr
                            er = MLmodel.decrypter.decrypt(e_encr)
                            e = np.linalg.norm(e_ok - er)
                            print('Error in e', e)
                            grad_ok = np.mean(MLmodel.Xtr_b * e_ok, axis=0).reshape((-1, 1))
                            grad_decr = MLmodel.decrypter.decrypt(grad_encr_dict[cla])
                            e = np.linalg.norm(grad_ok - grad_decr)
                            print(list(grad_decr.ravel()))
                            print('Error in grad', e)

                    del sigm_encr, e_encr#, Xe_encr
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_grads'
                    data = {'grad_encr': grad_encr_dict, 'NP': NP_dict}
                    del grad_encr_dict, NP_dict
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
