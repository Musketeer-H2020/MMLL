# -*- coding: utf-8 -*-
'''
Multiclass Logistic Classifier model under POM4

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
        preds_dict = {}
        NCLA = len(self.classes)
        X_b = np.hstack((np.ones((X.shape[0], 1)), X)).astype(float)
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
    This class implements the Multiclass Logistic Classifier  model, run at Master node. It inherits from Common_to_all_POMs.
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
        #self.regularization = regularization
        #self.classes = classes
        #self.balance_classes = balance_classes
        #self.Xval_b = Xval_b
        #self.yval = yval
        self.epsilon = 0.00000001  # to avoid log(0)
        self.momentum = 0
        self.regularization = 0.001
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
        #self.encrypter = self.cr.get_encrypter()  # to be shared        # self.encrypter.encrypt(np.random.normal(0, 1, (2,3)))
        #self.decrypter = self.cr.get_decrypter()  # to be kept as secret  self.encrypter.decrypt()
        self.create_FSM_master()
        self.FSMmaster.master_address = master_address
        self.added_bias = False
        self.model = Model()
        self.train_data_is_ready = False
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
            State(name='store_Xyblinded', on_enter=['while_store_Xyblinded']),
            State(name='mult_XB', on_enter=['while_mult_XB']),
            State(name='mult_XBM', on_enter=['while_mult_XBM']),
            State(name='decrypt_model', on_enter=['while_decrypt_model']),
            State(name='decrypt_modelM', on_enter=['while_decrypt_modelM']),
            State(name='mult_AB', on_enter=['while_mult_AB']),
            State(name='compute_expM', on_enter=['while_compute_expM']),
            State(name='compute_div', on_enter=['while_compute_divM']),
            State(name='compute_divM', on_enter=['while_compute_divM'])
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_store_Xyblinded', 'waiting_order', 'store_Xyblinded'],
            ['done_store_Xyblinded', 'store_Xyblinded', 'waiting_order'],

            ['go_mult_XB', 'waiting_order', 'mult_XB'],
            ['done_mult_XB', 'mult_XB', 'waiting_order'],

            ['go_mult_XBM', 'waiting_order', 'mult_XBM'],
            ['done_mult_XBM', 'mult_XBM', 'waiting_order'],

            ['go_mult_AB', 'waiting_order', 'mult_AB'],
            ['done_mult_AB', 'mult_AB', 'waiting_order'],

            ['go_compute_expM', 'waiting_order', 'compute_expM'],
            ['done_compute_expM', 'compute_expM', 'waiting_order'],

            ['go_compute_div', 'waiting_order', 'compute_div'],
            ['done_compute_div', 'compute_div', 'waiting_order'],

            ['go_compute_divM', 'waiting_order', 'compute_divM'],
            ['done_compute_divM', 'compute_divM', 'waiting_order'],

            ['go_decrypt_model', 'waiting_order', 'decrypt_model'],
            ['done_decrypt_model', 'decrypt_model', 'waiting_order'],

            ['go_decrypt_modelM', 'waiting_order', 'decrypt_modelM'],
            ['done_decrypt_modelM', 'decrypt_modelM', 'waiting_order']
        ]


        class FSM_master(object):

            self.name = 'FSM_master'

            def while_waiting_order(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' is waiting...')
                except:
                    raise
                    '''
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


            def while_mult_XB(self, MLmodel, B_bl):
                try:
                    data = {'B_bl': B_bl}
                    action = 'send_mult_XB'
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
                    #destination = MLmodel.cryptonode_address
                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    MLmodel.display(MLmodel.name + ' send_mult_XB to cryptonode')
                except:
                    raise
                    '''
                    print('ERROR AT LR while_mult_XB')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_mult_XBM(self, MLmodel, B_bl):
                try:
                    data = {'B_bl': B_bl}
                    action = 'send_mult_XBM'
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    
                    #destination = MLmodel.cryptonode_address
                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    MLmodel.display(MLmodel.name + ' send_mult_XBM to cryptonode')
                except:
                    raise
                    '''
                    print('ERROR AT MLC while_mult_XBM')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_expM(self, MLmodel, s_encr_dict):
                # Multiclass
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    MLmodel.gamma_dict = {}
                    MLmodel.s_encr_bl_dict = {}

                    for waddr in s_encr_dict.keys():
                        gamma_dict = {}
                        s_encr_bl_dict = {}
                        for cla in MLmodel.classes:
                            NP = MLmodel.s_encr_dict[waddr][cla].shape[0]
                            gamma_dict.update({cla: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})                  
                            s_encr_bl_dict.update({cla: MLmodel.s_encr_dict[waddr][cla] + gamma_dict[cla]})
                                        
                        MLmodel.gamma_dict.update({waddr: gamma_dict})                  
                        MLmodel.s_encr_bl_dict.update({waddr: s_encr_bl_dict})
                    
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ask_expM_bl'
                    data = {'s_encr_bl_dict': MLmodel.s_encr_bl_dict}
                    packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}

                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.master_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT LC master while_compute_expM')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_div(self, MLmodel, exp_s_encr_dict):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    MLmodel.r1_dict = {}
                    MLmodel.r2_dict = {}
                    MLmodel.num_bl_dict = {}
                    MLmodel.den_bl_dict = {}

                    for waddr in exp_s_encr_dict.keys():
                        NP = exp_s_encr_dict[waddr].shape[0]
                        MLmodel.r1_dict.update({waddr: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})
                        MLmodel.r2_dict.update({waddr: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})
                        num = MLmodel.encrypter.encrypt(MLmodel.r1_dict[waddr])
                        MLmodel.num_bl_dict.update({waddr: num})
                        den = (1 + exp_s_encr_dict[waddr]) * MLmodel.r2_dict[waddr]
                        MLmodel.den_bl_dict.update({waddr: den})
                    
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ask_div_bl'
                    data = {'num_bl_dict': MLmodel.num_bl_dict, 'den_bl_dict': MLmodel.den_bl_dict}
                    packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}

                    #destination = MLmodel.cryptonode_address
                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.master_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT LR master while_compute_div')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_divM(self, MLmodel, exp_s_encr_dict):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    MLmodel.r1_dict = {}
                    MLmodel.r2_dict = {}
                    MLmodel.num_bl_dict = {}
                    MLmodel.den_bl_dict = {}

                    for waddr in exp_s_encr_dict.keys():
                        r1_dict = {}
                        r2_dict = {}
                        num_bl_dict = {}
                        den_bl_dict = {}
                        for cla in MLmodel.classes:
                            NP = exp_s_encr_dict[waddr][cla].shape[0]
                            r1_dict.update({cla: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})
                            r2_dict.update({cla: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})
                            num = MLmodel.encrypter.encrypt(r1_dict[cla])
                            num_bl_dict.update({cla: num})
                            den = (1 + exp_s_encr_dict[waddr][cla]) * r2_dict[cla]
                            den_bl_dict.update({cla: den})

                        MLmodel.r1_dict.update({waddr: r1_dict})
                        MLmodel.r2_dict.update({waddr: r2_dict})
                        MLmodel.num_bl_dict.update({waddr: num_bl_dict})
                        MLmodel.den_bl_dict.update({waddr: den_bl_dict})

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ask_divM_bl'
                    data = {'num_bl_dict': MLmodel.num_bl_dict, 'den_bl_dict': MLmodel.den_bl_dict}
                    packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}

                    #destination = MLmodel.cryptonode_address
                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.master_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT MLC master while_compute_divM')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_decrypt_modelM(self, MLmodel, model_encr_dict):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    # Adding blinding to model
                    MLmodel.bl_dict = {}
                    MLmodel.model_encr_bl_dict = {}
                    bl_dict = {}
                    model_encr_bl_dict = {}

                    for key in list(model_encr_dict.keys()):
                        if key == 'wM':
                            classes = list(model_encr_dict['wM'].keys())
                            for cla in classes: 
                                x = model_encr_dict[key][cla]
                                M, N = x.shape
                                bl = np.random.normal(0, 1, (M, N))
                                bl_dict.update({cla: bl})
                                try:
                                    model_encr_bl_dict.update({cla: x + bl})
                                except Exception as err:
                                    print('ERROR at  while_decrypt_modelM')
                                    print('***** NUMERICAL OVERFLOW *******')
                                    print(err)
                                    raise
                                    #import code
                                    #code.interact(local=locals())

                        MLmodel.bl_dict.update({key: bl_dict})
                        MLmodel.model_encr_bl_dict.update({key: model_encr_bl_dict})

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    data = {'model_bl': MLmodel.model_encr_bl_dict}
                    action = 'send_modelM_encr_bl'
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    
                    #destination = MLmodel.cryptonode_address
                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    MLmodel.display(MLmodel.name + ' send_model_encr_bl to cryptonode')
                except:
                    raise
                    '''
                    print('ERROR AT MLC while_decrypt_modelM')
                    import code
                    code.interact(local=locals())
                    pass
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
        print(self.name + ': Starting training')
        self.display('MASTER_INIT', verbose=False)
        #self.X_encr_dict

        #self.MasterMLmodel.BX_dict
        #self.MasterMLmodel.By_dict

        self.display('PROC_MASTER_START', verbose=False)

        '''
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

        self.receivers_list = None
        if self.selected_workers is not None:
            self.workers_addresses = self.selected_workers
        else:
            self.workers_addresses = self.all_workers_addresses[:]

        '''
        print(self.workers_addresses)

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

        message = '%s - %s' % (self.cryptonode_address, self.send_to[self.cryptonode_address])
        self.display(message, True)
        self.display('=============================================== ', True)

        # Data at self.X_encr_dict, self.y_encr_dict

        check = False
        which = self.workers_addresses[0]

        self.NI = self.input_data_description['NI']
        self.grads_dict = {}
        self.NP_dict = {}
        self.model.w_dict = {}
        self.w_encr_dict = {}
        self.w_old_dict = {}

        self.classes =  self.target_data_description['output_type'][0]['values']
        self.model.classes = self.classes
        self.grad_old_dict = {}

        for cla in self.classes:
            self.model.w_dict.update({cla: np.random.normal(0, 0.001, (self.NI + 1, 1))})
            self.w_old_dict.update({cla: np.random.normal(0, 1.0, (self.NI + 1, 1))})
            self.w_encr_dict.update({cla: 0})
            self.grad_old_dict.update({cla: np.random.normal(0, 0.001, (self.NI + 1, 1))})

        self.stop_training = False
        self.selected_workers = self.workers_addresses
        kiter = 0

        if self.Xval is not None:
            message = 'WARNING: Validation data is not used during training.'
            self.display(message, True)

        self.display('PROC_MASTER_END', verbose=False)

        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)

            self.display('PROC_MASTER_START', verbose=False)
            # Computing s=wTX
            #self.wTX_encr_dict = self.crypto_mult_X(self.w_encr.T)
            self.s_encr_dict = {}
            for waddr in self.selected_workers:
                s_encr_dict = {}
                for cla in self.classes: 
                    s_encr_dict.update({cla: np.dot(self.X_encr_dict[waddr], self.model.w_dict[cla])})
                self.s_encr_dict.update({waddr: s_encr_dict})


            if check:
                err = 0
                X0 = self.decrypter.decrypt(self.X_encr_dict[which])
                for cla in self.classes:
                    #w = self.decrypter.decrypt(self.w_encr)
                    s = np.dot(X0, self.model.w_dict[cla])
                    #o_ = np.sum(self.Xw_encr_dict[which], axis=1).reshape((-1, 1))
                    s_decr = self.decrypter.decrypt(self.s_encr_dict[which][cla])
                    err += np.linalg.norm(s - s_decr)
                    
                print('Error in s = %f' % err)  # OK

            self.display('PROC_MASTER_END', verbose=False)

            # Añadiendo blinding y enviado a Crypto para calcular exp(-s)
            self.FSMmaster.go_compute_expM(self, self.s_encr_dict)
            self.run_Master()
            # self.exps_bl_encr_dict
            # blinding in self.gamma_dict

            self.display('PROC_MASTER_START', verbose=False)
            # deblinding
            self.exp_s_encr_dict = {}
            for waddr in self.exps_bl_encr_dict.keys():
                exp_s_encr_dict = {} 
                for cla in self.classes:
                    exp_s_encr_dict.update({cla: self.exps_bl_encr_dict[waddr][cla] * np.exp(self.gamma_dict[waddr][cla])})
                self.exp_s_encr_dict.update({waddr: exp_s_encr_dict})

            if check:
                err = 0
                for cla in self.classes:
                    exp_s_decr = self.decrypter.decrypt(self.exp_s_encr_dict[which][cla])
                    s = np.dot(X0, self.model.w_dict[cla])
                    exp_s = np.exp(-s)
                    err += np.linalg.norm(exp_s - exp_s_decr)

                print('Error in exp(-s) = %f' % err)  # OK
            self.display('PROC_MASTER_END', verbose=False)

            # Añadiendo blinding y enviado a Crypto para calcular la division
            self.FSMmaster.go_compute_divM(self, self.exp_s_encr_dict)
            self.run_Master()
            # self.sigm_encr_bl_dict

            self.display('PROC_MASTER_START', verbose=False)
            # deblinding
            self.o_encr_dict = {}
            for waddr in self.workers_addresses:
                o_encr_dict = {} 
                for cla in self.classes:
                    o_encr = self.sigm_encr_bl_dict[waddr][cla] / self.r1_dict[waddr][cla] * self.r2_dict[waddr][cla]
                    o_encr_dict.update({cla: o_encr})
                self.o_encr_dict.update({waddr: o_encr_dict})

            if check:
                err = 0
                for cla in self.classes:
                    o_decr = self.decrypter.decrypt(self.o_encr_dict[which][cla])
                    s = np.dot(X0, self.model.w_dict[cla])
                    exp_s = np.exp(-s)
                    o_orig = 1 / (1 + np.exp(-s))
                    err += np.linalg.norm(exp_s - exp_s_decr)
                print('Error in o = %f' % err)  # OK

            self.e_encr_dict = {}
            NPtotal = 0
            for waddr in self.y_encr_dict.keys():
                NPtotal += self.y_encr_dict[self.workers_addresses[0]][self.classes[0]].reshape(-1, 1).shape[0]
                e_encr_dict = {}
                for cla in self.classes:
                    y = self.y_encr_dict[waddr][cla].reshape(-1, 1)
                    e_encr = self.o_encr_dict[waddr][cla].reshape(-1, 1) - y 
                    e_encr_dict.update({cla: e_encr})
                self.e_encr_dict.update({waddr: e_encr_dict})

            if check:
                err = 0
                for cla in self.classes:
                    o_decr = self.decrypter.decrypt(self.o_encr_dict[which][cla])
                    s = np.dot(X0, self.model.w_dict[cla])
                    exp_s = np.exp(-s)
                    o_orig = 1 / (1 + np.exp(-s))
                    y_orig = self.decrypter.decrypt(self.y_encr_dict[which][cla].reshape(-1, 1))
                    e_orig = o_orig - y_orig
                    e_decr = self.decrypter.decrypt(self.e_encr_dict[which][cla])
                    err += np.linalg.norm(e_orig - e_decr)

                print('Error in e = %f' % err)  # OK
            self.display('PROC_MASTER_END', verbose=False)

            # Computing eX
            self.eX_encr_dict = self.crypto_mult_XM(self.e_encr_dict)

            if check:
                err = 0
                for cla in self.classes:
                    o_decr = self.decrypter.decrypt(self.o_encr_dict[which][cla])
                    s = np.dot(X0, self.model.w_dict[cla])
                    exp_s = np.exp(-s)
                    o_orig = 1 / (1 + np.exp(-s))
                    y_orig = self.decrypter.decrypt(self.y_encr_dict[which][cla].reshape(-1, 1))
                    e_orig = y_orig - o_orig
                    eX_orig = X0 * e_orig
                    eX_decr = self.decrypter.decrypt(self.eX_encr_dict[which][cla])
                    err += np.linalg.norm(eX_orig - eX_decr)
                print('Error in eX = %f' % err)  # OK

            self.display('PROC_MASTER_START', verbose=False)

            grad_encr_dict = {}  # one per class
            for cla in self.classes:
                grad_encr = self.encrypter.encrypt(np.zeros((self.NI + 1, 1)))
                for waddr in self.workers_addresses:
                    eX_encr = self.eX_encr_dict[waddr][cla]
                    #grad_encr += np.sum(eX_encr, axis=0).reshape((-1, 1)) / NPtotal
                    grad_encr += np.mean(eX_encr, axis=0).reshape((-1, 1))
                
                grad_encr_dict.update({cla: grad_encr})  # one per class
          
            
            if check:
                err = 0
                for cla in self.classes:
                    o_decr = self.decrypter.decrypt(self.o_encr_dict[which][cla])
                    s = np.dot(X0, self.model.w_dict[cla])
                    exp_s = np.exp(-s)
                    o_orig = 1 / (1 + np.exp(-s))
                    y_orig = self.decrypter.decrypt(self.y_encr_dict[which][cla].reshape(-1, 1))
                    e_orig = y_orig - o_orig
                    eX_orig = X0 * e_orig
                    grad_orig = np.sum(eX_orig, axis=0).reshape((-1, 1))
                    grad_decr = self.decrypter.decrypt(np.sum(self.eX_encr_dict[which][cla], axis=0).reshape((-1, 1)))
                    err += np.linalg.norm(grad_orig - grad_decr)
                print('Error in grad = %f' % err)  # OK
            
            for cla in self.classes:
                #self.w_encr_dict[cla] = self.w_encr_dict[cla] + self.mu * grad_encr_dict[cla]
                grad_acum = self.mu * grad_encr_dict[cla] / len(self.workers_addresses)                  

                # Momentum
                v_1 = np.copy(self.grad_old_dict[cla]) # old gradient
                momentum = self.momentum * v_1
                v = momentum + grad_acum
                self.w_encr_dict[cla] = self.w_encr_dict[cla] - v
                self.grad_old_dict[cla] = grad_acum

            # Decrypting the model
            self.model_decr_dict = self.decrypt_modelM({'wM': self.w_encr_dict})

            #self.w_old = dict(self.model.w_dict)
            self.model.w_dict = dict(self.model_decr_dict['wM'])

            #print(self.model.w_dict)

            for cla in self.classes:
                self.w_encr_dict[cla] = self.encrypter.encrypt(self.model.w_dict[cla])           
            
            # stopping
            inc_w = 0
            for cla in self.classes:
                inc_w += np.linalg.norm(self.model.w_dict[cla] - self.w_old_dict[cla]) / np.linalg.norm(self.w_old_dict[cla])
                self.w_old_dict[cla] = np.copy(self.model.w_dict[cla])           

            # Stop if convergence is reached
            if inc_w < self.conv_stop:
                self.stop_training = True
            if kiter == self.Nmaxiter:
                self.stop_training = True
           
            message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, kiter, inc_w)
            self.display(message, True)
            kiter += 1
            self.display('PROC_MASTER_END', verbose=False)
            self.display('MASTER_ITER_END', verbose=False)

        self.model.niter = kiter
        self.model.is_trained = True
        self.display(self.name + ': Training is done')
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

            if packet['action'] == 'ACK_grads':
                self.grads_dict.update({sender: packet['data']['grad_encr']})

            if packet['action'] == 'ACK_sent_XB_bl_encr_dict':
                self.XB_bl_encr_dict = packet['data']['XB_bl_encr_dict']
                self.FSMmaster.done_mult_XB(self)

            if packet['action'] == 'ACK_sent_XBM_bl_encr_dict':
                self.XB_bl_encr_dict = packet['data']['XB_bl_encr_dict']
                self.FSMmaster.done_mult_XBM(self)

            if packet['action'] == 'ACK_sent_decr_bl_model':
                self.model_decr_bl = packet['data']['model_decr_bl']
                self.FSMmaster.done_decrypt_model(self)

            if packet['action'] == 'ACK_sent_decr_bl_modelM':
                self.model_decr_bl_dict = packet['data']['model_decr_bl_dict']
                self.FSMmaster.done_decrypt_modelM(self)

            if packet['action'] == 'ACK_expM_bl':
                self.exps_bl_encr_dict = packet['data']['exps_bl_dict']
                self.FSMmaster.done_compute_expM(self)

            if packet['action'] == 'ACK_div_bl':
                self.sigm_encr_bl_dict = packet['data']['sigm_encr_bl_dict']
                self.FSMmaster.done_compute_div(self)

            if packet['action'] == 'ACK_divM_bl':
                self.sigm_encr_bl_dict = packet['data']['sigm_encr_bl_dict']
                self.FSMmaster.done_compute_divM(self)

            #sender = packet['sender']
            if packet['action'][0:3] == 'ACK':
                self.state_dict[sender] = packet['action']
                if sender == self.cryptonode_address:     
                    if packet['action'] not in ['ACK_send_ping']:
                        try:
                            self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], 'ca', str(packet['message_id'])), verbose=False)
                        except:
                            self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], 'ca'), verbose=False)                    
                            pass
                else:
                    if packet['action'] not in ['ACK_send_ping']:       
                            try:
                                self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
                            except:
                                self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                                pass

        except Exception as err:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Master')
            raise
            import code
            code.interact(local=locals())    
            '''    
        return


#===============================================================
#                 Worker
#===============================================================
class MLC_Worker(Common_to_all_POMs):
    '''
    Class implementing Multiclass Logistic Classifier , run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None, cryptonode_address=None):
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
        self.Xtr_b = Xtr_b
        #self.Xtr_b = self.add_bias(Xtr_b)
        self.ytr = ytr
        self.NPtr = len(ytr)
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

            def while_compute_gradients(self, MLmodel, packet):
                try:
                    MLmodel.display(MLmodel.name + ' %s: computing gradients...' % (str(MLmodel.worker_address)))
                    MLmodel.display('PROC_WORKER_START', verbose=False)
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
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_grads'
                    data = {'grad_encr': grad_encr}
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
            self.display('COMMS_WORKER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
        except:
            self.display('WORKER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
            pass
        try:
            # Exit the process
            if packet['action'] == 'STOP':
                self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
                self.terminate = True

            if packet['action'] == 'ACK_sent_encrypter':
                print('ProcessReceivedPacket_Worker ACK_sent_encrypter')
                #self.FSMworker.go_storing_Pk(self, packet)
                #self.FSMworker.done_storing_Pk(self)

            if packet['action'] == 'send_w_encr':
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



#===============================================================
#                 Crypto
#===============================================================
class MLC_Crypto(Common_to_all_POMs):
    '''
    Class implementing Multiclass Logistic Classifier , run at Crypto

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
        self.message_counter = 100 # used to number the messages
        t = time.time()
        seed = int((t - int(t)) * 10000)
        np.random.seed(seed=seed)

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
            self.display('COMMS_CRYPTO_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
        except:
            self.display('CRYPTO MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
            pass
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
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Crypto')
            import code
            code.interact(local=locals())
            '''

        return self.terminate
