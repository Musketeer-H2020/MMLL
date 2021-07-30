# -*- coding: utf-8 -*-
'''
Kernel Regression under POM5

'''

__author__ = "Angel Navia-Vázquez "
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
    Kernel Regression model.
    """
    def __init__(self):
        self.C = None
        self.w = None
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']
        t = time.time()
        seed = int((t - int(t)) * 10000)
        np.random.seed(seed=seed)

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
                            from skl2onnx import convert_sklearn # conda install -c conda-forge skl2onnx
                            from skl2onnx.common.data_types import FloatTensorType
                            from sklearn.svm import SVR                            

                            NC = self.C.shape[0]
                            NI = self.C.shape[1]
                            gamma = 1 / 2 / self.sigma**2                           
                            export_model = SVR(C=1.0, gamma=gamma)
                            X = np.random.normal(0, 1, (100, NI))
                            w = np.random.normal(0, 1, (NI, 1))
                            y = np.sign(np.dot(X, w)).ravel()
                            export_model.fit(X, y)
                            export_model.support_vectors_ = self.C
                            export_model._dual_coef_ = self.w[1:, :].T
                            export_model.dual_coef_ = self.w[1:, :].T
                            export_model._intercept_ = self.w[0, :]
                            export_model.intercept_ = self.w[0, :]
                            export_model.n_support_[0] = NC
                            export_model.support_ = np.array(range(NC))

                            # Convert into ONNX format
                            input_type = [('float_input', FloatTensorType([None, NI]))]
                            onnx_model = convert_sklearn(export_model, initial_types=input_type)
                            with open(filename, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            print('=' * 80)
                            print('Model saved at %s in ONNX format.' %filename)
                            print('=' * 80)

                        elif extension == 'pmml':
                            from sklearn2pmml import sklearn2pmml # pip install git+https://github.com/jpmml/sklearn2pmml.git
                            from sklearn2pmml.pipeline import PMMLPipeline
                            from sklearn.svm import SVR                            

                            NC = self.C.shape[0]
                            NI = self.C.shape[1]
                            gamma = 1 / 2 / self.sigma**2                           
                            export_model = SVR(C=1.0, gamma=gamma)
                            X = np.random.normal(0, 1, (100, NI))
                            w = np.random.normal(0, 1, (NI, 1))
                            y = np.sign(np.dot(X, w)).ravel()
                            export_model.fit(X, y)
                            export_model.support_vectors_ = self.C
                            export_model._dual_coef_ = self.w[1:, :].T
                            export_model.dual_coef_ = self.w[1:, :].T
                            export_model._intercept_ = self.w[0, :]
                            export_model.intercept_ = self.w[0, :]
                            export_model.n_support_[0] = NC
                            export_model.support_ = np.array(range(NC))
                            
                            pipeline = PMMLPipeline([("estimator", export_model)])
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


class KR_Master(Common_to_all_POMs):
    """
    This class implements the Kernel Regression, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`KR_Master` instance.

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
        #self.regularization = regularization
        #self.classes = classes
        #self.balance_classes = balance_classes
        #self.Xval_b = Xval_b
        #self.yval = yval
        self.epsilon = 0.00000001  # to avoid log(0)
        for k in range(0, self.Nworkers):
            self.state_dict.update({self.workers_addresses[k]: ''})
        #default values
        #self.landa = 0.5
        self.mu = 0.1
        self.momentum = 0
        self.regularization = 0.001

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
        self.model = Model()
        self.model.C = self.C
        self.model.sigma = np.sqrt(self.input_data_description['NI']) * self.fsigma 
        self.Kacum_dict = {}
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

            State(name='getting_KTK', on_enter=['while_getting_KTK']),
            State(name='selecting_C', on_enter=['while_selecting_C']),
            State(name='sending_C', on_enter=['while_sending_C']),
            State(name='updating_w', on_enter=['while_updating_w'])
         ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_send_w_encr', 'waiting_order', 'send_w_encr'],
            ['done_send_w_encr', 'send_w_encr', 'waiting_order'],

            ['go_selecting_C', 'waiting_order', 'selecting_C'],
            ['go_waiting_order', 'selecting_C', 'waiting_order'],

            ['go_sending_C', 'waiting_order', 'sending_C'],
            ['go_waiting_order', 'sending_C', 'waiting_order'],

            ['go_getting_KTK', 'waiting_order', 'getting_KTK'],
            ['go_updating_w', 'getting_KTK', 'updating_w'],
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
                    data.update({'w_encr': MLmodel.w_encr})
                    # WARNING, only for debug purposes...
                    #data.update({'decrypter': MLmodel.decrypter})
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

            def while_selecting_C(self, MLmodel):
                try:
                    action = 'selecting_C'
                    data = {'C': MLmodel.model.C, 'sigma': MLmodel.model.sigma}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')
                    else:
                        MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                        MLmodel.display(MLmodel.name + ': broadcasted C to Workers: %s' % str([MLmodel.receive_from[w] for w in MLmodel.selected_workers]))

                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_selecting_C')
                    import code
                    code.interact(local=locals())
                    '''         
                return

            def while_sending_C(self, MLmodel):
                try:
                    action = 'sending_C'
                    data = {'C': MLmodel.model.C, 'sigma': MLmodel.model.sigma}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')
                    else:
                        MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                        MLmodel.display(MLmodel.name + ': broadcasted C to Workers: %s' % str([MLmodel.receive_from[w] for w in MLmodel.selected_workers]))

                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_sending_C')
                    import code
                    code.interact(local=locals())
                    '''         
                return

            def while_getting_KTK(self, MLmodel):
                try:
                    action = 'compute_KTK'
                    data = None
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': broadcasted compute_KTK to all workers')
                    else:
                        MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                        MLmodel.display(MLmodel.name + ': broadcasted compute_KTK to Workers: %s' % str([MLmodel.receive_from[w] for w in MLmodel.selected_workers]))
                    
                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_getting_KTK')
                    import code
                    code.interact(local=locals())
                    '''       

                return

            def while_updating_w(self, MLmodel):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    NC = MLmodel.model.C.shape[0]    
                    MLmodel.KTK_accum = np.zeros((NC + 1, NC + 1))
                    MLmodel.KTy_accum = np.zeros((NC + 1, 1))
                    for waddr in MLmodel.workers_addresses:
                        MLmodel.KTK_accum += MLmodel.KTK_dict[waddr]
                        MLmodel.KTy_accum += MLmodel.KTy_dict[waddr].reshape((NC + 1, 1))

                    MLmodel.model.w = np.dot(np.linalg.inv(MLmodel.KTK_accum + MLmodel.regularization * np.eye(NC + 1)), MLmodel.KTy_accum)        
                    MLmodel.display('PROC_MASTER_END', verbose=False)
                except Exception as err:
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

        self.display('PROC_MASTER_START', verbose=False)

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

        '''
        self.FSMmaster.go_selecting_C(self)
        self.run_Master()

        # Selecting centroids with largest projection
        Ncandidates = self.C.shape[0]
        Kacum_total = np.zeros(Ncandidates)
        for addr in self.workers_addresses:
            Kacum_total += self.Kacum_dict[addr]

        index = np.argsort(-Kacum_total)
        self.C = self.C[index[0: self.NC], :]
        '''

        # Computing and storing KXC_val
        if self.Xval is not None:
            XC2 = -2 * np.dot(self.Xval, self.C.T)
            XC2 += np.sum(np.multiply(self.Xval, self.Xval), axis=1).reshape((-1, 1))
            XC2 += np.sum(np.multiply(self.C, self.C), axis=1).reshape((1, self.NC))
            # Gauss
            KXC_val = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2))
            self.KXC_val = np.hstack( (np.ones((self.Xval.shape[0], 1)), KXC_val)) # NP_val x NC + 1
            self.yval.astype(float).reshape((-1, 1))

        self.model.C = self.C
        self.NC = self.C.shape[0]

        self.FSMmaster.go_sending_C(self)
        self.run_Master()

        #self.NI = self.input_data_description['NI']
        if self.Xval is None:  # A validation set is not provided
            self.w = np.random.normal(0, 0.1, (self.NC + 1, 1))
        else:
            # Initialization with val solution
            KTK_accum = np.dot(self.KXC_val.T, self.KXC_val)
            KTy_accum = np.dot(self.KXC_val.T, self.yval.reshape((-1, 1)))
            NP = self.KXC_val.shape[0]
            wini = np.dot(np.linalg.inv(KTK_accum/NP + self.regularization * np.eye(self.NC + 1)), KTy_accum / NP)        
            self.w = np.copy(wini)

        self.w_old = np.random.normal(0, 1, (self.NC + 1, 1))
        self.grad_old = np.random.normal(0, 0.001, (self.NC + 1, 1))
        # Encrypting w
        self.w_encr = self.encrypter.encrypt(self.w)
        self.grads_dict = {}
        self.NP_dict = {}

        self.stop_training = False
        self.kiter = 0
        self.mseval = 1000

        self.kiter = 1
        self.display('PROC_MASTER_END', verbose=False)

        #self.display('MASTER_ITER_START', verbose=False)
        #self.display('PROC_MASTER_START', verbose=False)

        #self.FSMmaster.go_send_w_encr(self)
        #self.run_Master()

        #self.FSMmaster.go_getting_KTK(self)
        #self.run_Master()

        #self.FSMmaster.go_updating_w(self)

        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)
            self.FSMmaster.go_send_w_encr(self)
            self.run_Master()

            self.display('PROC_MASTER_START', verbose=False)

            NPtotal = 0
            grad = np.zeros((self.NC + 1, 1))
            for key in self.grads_dict:
                grad_encr = self.grads_dict[key]
                NPtotal += self.NP_dict[key]
                #gradq = self.cr.vmasterDec_BCP(gradq_encr, self.cr.PK)
                #grad_decr = self.cr.vQinv_m(gradq, gradq_encr[0, 0].N)
                grad_decr = self.decrypter.decrypt(grad_encr)
                grad += grad_decr

            #grad = grad / NPtotal

            grad += np.random.normal(0, self.regularization, grad.shape)
            
            # Momentum
            v_1 = np.copy(self.grad_old) # old gradient
            #v_1 = self.landa * v_1 + (1 - self.landa) * np.copy(grad)
            momentum = self.momentum * v_1

            grad = self.mu * grad / len(self.workers_addresses)                  

            v = momentum + grad

            self.w_old = self.w.copy()
            self.w = self.w.reshape((-1, 1)) - v
            self.grad_old = np.copy(grad)

            inc_w = np.linalg.norm(self.w - self.w_old) / np.linalg.norm(self.w_old)
            message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
            self.display(message, verbose=True)
            #print(message)

            self.kiter += 1
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            # Stop if convergence is reached
            if inc_w < self.conv_stop:
                self.stop_training = True

            '''    
            else:
                MSE_val = []
                mus = np.arange(0, 10.0, 0.0001)
                Xw = np.dot(self.KXC_val, self.w)
                Xgrad = np.dot(self.KXC_val, grad)

                for mu in mus:
                    s_val = Xw + mu * Xgrad
                    o_val = s_val.ravel()
                    mse_val = np.mean((o_val - self.yval.ravel())**2)
                    MSE_val.append(mse_val)

                del Xw, Xgrad, s_val, o_val

                min_pos = np.argmin(MSE_val)
                mu_opt = mus[min_pos]
                del mus
                self.w_old = self.w.copy()
                self.w = self.w + mu_opt * grad
                del grad
                # stopping
                inc_w = np.linalg.norm(self.w - self.w_old) / np.linalg.norm(self.w_old)

                self.mseval_old = self.mseval

                self.mseval = MSE_val[min_pos]
                del MSE_val
                print('Optimal mu = %f, MSE val=%f' % (mu_opt, self.mseval))

                inc_mseval = (self.mseval_old - self.mseval)/ self.mseval
                # Stop if convergence is reached
                if inc_mseval <= 0:
                    self.stop_training = True

                message = 'Maxiter = %d, iter = %d, inc_MSE_val = %f, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_mseval, inc_w)
                #self.display(message, verbose=True)
                print(message)
            self.w_encr = self.encrypter.encrypt(self.w)
            '''
            
        self.display('PROC_MASTER_END', verbose=False)

        self.display('MASTER_ITER_END', verbose=False)

        self.model.w = self.w
        self.model.niter = self.kiter
        self.model.is_trained = True
        self.display(self.name + ': Training is done', verbose=True)
        self.display('MASTER_FINISH', verbose=False)


    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_grads'):
            self.FSMmaster.done_send_w_encr(self)

        if self.chekAllStates('ACK_update_tr_data'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_projecting_C'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_storing_C'):
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

            if packet['action'] == 'ACK_grads':
                self.grads_dict.update({sender: packet['data']['grad_encr']})
                self.NP_dict.update({sender: packet['data']['NP']})

            if packet['action'] == 'ACK_update_tr_data':
                #print('ProcessReceivedPacket_Master ACK_update_tr_data')
                self.newNI_dict.update({sender: packet['data']['newNI']})

            if packet['action'] == 'ACK_projecting_C':
                self.Kacum_dict.update({sender: packet['data']['Kacum']})

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
class KR_Worker(Common_to_all_POMs):
    '''
    Class implementing Linear Regression, run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`KR_Worker` instance.

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
                    #MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
                    MLmodel.Xtr_b = MLmodel.Xtr_b.astype(float)
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

            def while_compute_gradients(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    MLmodel.display(MLmodel.name + ' %s: computing gradients...' % (str(MLmodel.worker_address)))
                    w_encr = packet['data']['w_encr']
                    try:
                        # Warning, only for debugging
                        decrypter = packet['data']['decrypter']
                    except:
                        pass
                    #NW = wq_encr.shape[0]
                    #for kw in range(NW):
                    #    wq_encr[kw, 0].encrypter = MLmodel.cr.encrypter
                    
                    #wq_encr = MLmodel.undill_it(packet['data']['wq_encr'])

                    NP = MLmodel.KXC.shape[0]
                    NI = MLmodel.KXC.shape[1]
                    y = MLmodel.ytr.reshape(-1, 1)

                    o_encr = np.dot(MLmodel.KXC, w_encr)
                    e_encr = (o_encr.reshape(-1, 1) - y)

                    '''
                    Xo_encr = MLmodel.KXC * o_encr
                    Xy = MLmodel.KXC * y
                    grad_o_encr = np.sum(Xo_encr, axis=0).reshape((-1, 1))
                    grady = np.dot(y.T, MLmodel.KXC).reshape(-1, 1)
                    grado_encr = np.dot(o_encr.T, MLmodel.KXC).reshape(-1, 1)
                    #Xe_encr = MLmodel.KXC * e_encr
                    #grad_encr = np.sum(Xe_encr, axis=0).reshape((-1, 1))
                    '''
                    grad_encr = np.dot(e_encr.T, MLmodel.KXC).reshape(-1, 1) / NP
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_grads'
                    data = {'grad_encr': grad_encr, 'NP': NP}
                    packet = {'action': action, 'sender': MLmodel.worker_address, 'data': data, 'to': 'MLmodel'}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_grads' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_compute_gradients')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_projecting_C(self, MLmodel, packet):
                # We project X over C and return accumulated
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    MLmodel.C = packet['data']['C']
                    NC = MLmodel.C.shape[0]
                    MLmodel.sigma = packet['data']['sigma']
                    NI = MLmodel.Xtr_b.shape[1]
                    NP = MLmodel.Xtr_b.shape[0]

                    #MLmodel.sigma = np.sqrt(NI) * MLmodel.fsigma
                    X = MLmodel.Xtr_b
                    XC2 = -2 * np.dot(X, MLmodel.C.T)
                    XC2 += np.sum(np.multiply(X, X), axis=1).reshape((NP, 1))
                    XC2 += np.sum(np.multiply(MLmodel.C, MLmodel.C), axis=1).reshape((1, NC))
                    # Gauss
                    KXC = np.exp(-XC2 / 2.0 /  (MLmodel.sigma ** 2))
                    Kacum = np.sum(KXC, axis = 0)
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_projecting_C'
                    data = {'Kacum': Kacum}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_projecting_C' % (str(MLmodel.worker_address)))
                    
                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_projecting_C')
                    import code
                    code.interact(local=locals())
                    '''   
                return

            def while_storing_C(self, MLmodel, packet):
                # We store C and compute KXC
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
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

                    # th minimum 
                    min_th = 1e-10
                    which = (MLmodel.KXC < min_th)
                    MLmodel.KXC[which]  = 0

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_storing_C'
                    data = {}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_storing_C' % (str(MLmodel.worker_address)))
                    
                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_storing_C')
                    import code
                    code.interact(local=locals())
                    '''       
                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='compute_gradients', on_enter=['while_compute_gradients']),
            State(name='storing_Pk', on_enter=['while_storing_Pk']),
            State(name='projecting_C', on_enter=['while_projecting_C']),
            State(name='storing_C', on_enter=['while_storing_C'])
            ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_compute_gradients', 'waiting_order', 'compute_gradients'],
            ['done_compute_gradients', 'compute_gradients', 'waiting_order'],

            ['go_storing_Pk', 'waiting_order', 'storing_Pk'],
            ['done_storing_Pk', 'storing_Pk', 'waiting_order'],

            ['go_projecting_C', 'waiting_order', 'projecting_C'],
            ['done_projecting_C', 'projecting_C', 'waiting_order'],

            ['go_storing_C', 'waiting_order', 'storing_C'],
            ['done_storing_C', 'storing_C', 'waiting_order'],


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
                self.FSMworker.go_compute_gradients(self, packet)
                self.FSMworker.done_compute_gradients(self)

            if packet['action'] == 'selecting_C':
                #self.C = packet['data']['C']
                self.FSMworker.go_projecting_C(self, packet)
                self.FSMworker.done_projecting_C(self)

            if packet['action'] == 'sending_C':
                #self.C = packet['data']['C']
                self.FSMworker.go_storing_C(self, packet)
                self.FSMworker.done_storing_C(self)

        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Worker')
            import code
            code.interact(local=locals())
            '''


        return self.terminate
