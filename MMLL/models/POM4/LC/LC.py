# -*- coding: utf-8 -*-
'''
Logistic Classifier model under POM4

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Dec. 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
#from pympler import asizeof #asizeof.asizeof(my_object)
import pickle
from  sklearn.linear_model import LogisticRegression

class Model():
    """
    Logistic Classifier model.
    """
    def __init__(self):
        self.w = None
        self.is_trained = False
        self.supported_formats = ['pkl', 'onnx', 'pmml']

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
        prediction_values = self.sigm(np.dot(X_b, self.w.ravel()))
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
                            from sklearn import linear_model
                            from skl2onnx import convert_sklearn # conda install -c conda-forge skl2onnx
                            from skl2onnx.common.data_types import FloatTensorType

                            export_model = linear_model.LogisticRegression()

                            NI = self.w.shape[0] - 1
                            X = np.random.normal(0, 1, (100, NI))
                            w = np.random.normal(0, 1, (NI, 1))
                            y = ((np.sign(np.dot(X, w)) + 1) / 2.0).ravel()
                            export_model.fit(X, y)
                            export_model.coef_ = self.w[1:].T
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
        self.model = Model()
        self.added_bias = False
        self.train_data_is_ready = False

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
            State(name='decrypt_model', on_enter=['while_decrypt_model']),
            State(name='mult_AB', on_enter=['while_mult_AB']),
            State(name='compute_exp', on_enter=['while_compute_exp']),
            State(name='compute_div', on_enter=['while_compute_div'])


        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_store_Xyblinded', 'waiting_order', 'store_Xyblinded'],
            ['done_store_Xyblinded', 'store_Xyblinded', 'waiting_order'],

            ['go_mult_XB', 'waiting_order', 'mult_XB'],
            ['done_mult_XB', 'mult_XB', 'waiting_order'],

            ['go_mult_AB', 'waiting_order', 'mult_AB'],
            ['done_mult_AB', 'mult_AB', 'waiting_order'],

            ['go_compute_exp', 'waiting_order', 'compute_exp'],
            ['done_compute_exp', 'compute_exp', 'waiting_order'],

            ['go_compute_div', 'waiting_order', 'compute_div'],
            ['done_compute_div', 'compute_div', 'waiting_order'],

            ['go_decrypt_model', 'waiting_order', 'decrypt_model'],
            ['done_decrypt_model', 'decrypt_model', 'waiting_order']
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
                    packet = {'action': 'send_mult_XB', 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
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

            def while_compute_exp(self, MLmodel, s_encr_dict):
                try:
                    MLmodel.gamma_dict = {}
                    MLmodel.s_encr_bl_dict = {}
                    for waddr in s_encr_dict.keys():
                        NP = MLmodel.s_encr_dict[waddr].shape[0]
                        MLmodel.gamma_dict.update({waddr: np.random.uniform(1, 2, (NP, 1)).reshape((-1, 1))})                  
                        MLmodel.s_encr_bl_dict.update({waddr: MLmodel.s_encr_dict[waddr] + MLmodel.gamma_dict[waddr]})

                    action = 'ask_exp_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'s_encr_bl_dict': MLmodel.s_encr_bl_dict}
                    packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}
                    #size_bytes = asizeof.asizeof(dill.dumps(packet))
                    #MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)
                    #MLmodel.message_counter += 1
                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.master_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT LC master while_compute_exp')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_div(self, MLmodel, exp_s_encr_dict):
                try:
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

                    action = 'ask_div_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'num_bl_dict': MLmodel.num_bl_dict, 'den_bl_dict': MLmodel.den_bl_dict}
                    packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}
                    #size_bytes = asizeof.asizeof(dill.dumps(packet))
                    #MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)
                    #MLmodel.message_counter += 1
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
                    packet = {'action': 'send_model_encr_bl', 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])
                    MLmodel.display(MLmodel.name + ' send_model_encr_bl to cryptonode')
                except:
                    raise
                    '''
                    print('ERROR AT LR while_decrypt_model')
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
        #self.X_encr_dict

        #self.MasterMLmodel.BX_dict
        #self.MasterMLmodel.By_dict


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

        # Data at self.X_encr_dict, self.y_encr_dict

        check = False
        which = '0'

        self.NI = self.input_data_description['NI']
        self.w = np.random.normal(0, 0.001, (self.NI + 1, 1))
        self.w_encr = self.encrypter.encrypt(self.w)
        self.w_old = np.random.normal(0, 10, (self.NI + 1, 1)) # large to avoid stop at first iteration

        self.stop_training = False
        kiter = 0
        self.grads_dict = {}

        self.selected_workers = self.workers_addresses

        if self.Xval is not None:
            message = 'WARNING: Validation data is not used during training.'
            self.display(message, True)

        self.stop_training = False
        kiter = 0
        while not self.stop_training:
            # Computing s=wTX
            #self.wTX_encr_dict = self.crypto_mult_X(self.w_encr.T)
            self.s_encr_dict = {}
            for waddr in self.selected_workers:
                self.s_encr_dict.update({waddr: np.dot(self.X_encr_dict[waddr], self.w)})

            if check:
                X0 = self.decrypter.decrypt(self.X_encr_dict[which])
                #w = self.decrypter.decrypt(self.w_encr)
                s = np.dot(X0, self.w)
                #o_ = np.sum(self.Xw_encr_dict[which], axis=1).reshape((-1, 1))
                s_decr = self.decrypter.decrypt(self.s_encr_dict[which])
                print('Error in s = %f' % np.linalg.norm(s - s_decr))  # OK

            # Añadiendo blinding y enviado a Crypto para calcular exp(-s)
            self.FSMmaster.go_compute_exp(self, self.s_encr_dict)
            self.run_Master()
            # self.exps_bl_encr_dict
            # deblinding
            self.exp_s_encr_dict = {}
            for waddr in self.exps_bl_encr_dict.keys():
                exp_s_encr = self.exps_bl_encr_dict[waddr] * np.exp(self.gamma_dict[waddr])
                self.exp_s_encr_dict.update({waddr: exp_s_encr})

            if check:
                exp_s_decr = self.decrypter.decrypt(self.exp_s_encr_dict[which])
                exp_s = np.exp(-s)
                print('Error in exp(-s) = %f' % np.linalg.norm(exp_s - exp_s_decr))  # OK

            # Añadiendo blinding y enviado a Crypto para calcular la division
            self.FSMmaster.go_compute_div(self, self.exp_s_encr_dict)
            self.run_Master()
            #sigm_encr_bl_dict

            # Deblinding
            self.o_encr_dict = {}
            for waddr in self.workers_addresses:
                o_encr = self.sigm_encr_bl_dict[waddr] / self.r1_dict[waddr] * self.r2_dict[waddr]
                self.o_encr_dict.update({waddr: o_encr})

            if check:
                o_decr = self.decrypter.decrypt(self.o_encr_dict[which])
                o_orig = 1 / (1 + np.exp(-s))
                print('Error in o = %f' % np.linalg.norm(o_orig - o_decr))  # OK


            self.e_encr_dict = {}
            NPtotal = 0
            for waddr in self.y_encr_dict.keys():
                y = self.y_encr_dict[waddr].reshape(-1, 1)
                NPtotal += y.shape[0]
                e_encr = y - self.o_encr_dict[waddr]
                self.e_encr_dict.update({waddr: e_encr})

            if check:
                y_orig = self.decrypter.decrypt(self.y_encr_dict[which].reshape(-1, 1))
                e_orig = y_orig - o_orig
                e_decr = self.decrypter.decrypt(self.e_encr_dict[which])
                print('Error in e = %f' % np.linalg.norm(e_decr - e_orig))  # OK
           
            # Computing eX
            self.eX_encr_dict = self.crypto_mult_X(self.e_encr_dict)

            if check:
                eX_orig = X0 * e_orig
                eX_decr = self.decrypter.decrypt(self.eX_encr_dict[which])
                print('Error in eX = %f' % np.linalg.norm(eX_orig - eX_decr))  # OK

            grad_encr = self.encrypter.encrypt(np.zeros((self.NI + 1, 1)))
            for waddr in self.workers_addresses:
                eX_encr = self.eX_encr_dict[waddr]
                grad_encr += np.sum(eX_encr, axis=0).reshape((-1, 1)) / NPtotal
            
            if check:
                grad_orig = np.sum(eX_orig, axis=0).reshape((-1, 1))
                grad_decr = self.decrypter.decrypt(np.sum(self.eX_encr_dict[which], axis=0).reshape((-1, 1)))
                print('Error in grad = %f' % np.linalg.norm(grad_orig - grad_decr))  # OK

            self.w_encr += self.mu * grad_encr

            # Decrypting the model
            self.model_decr = self.decrypt_model({'w': self.w_encr})

            self.w_old = np.copy(self.w)
            self.w = np.copy(self.model_decr['w'])

            self.w_encr = self.encrypter.encrypt(self.w)           
            
            # stopping
            inc_w = np.linalg.norm(self.w - self.w_old) / np.linalg.norm(self.w_old)
            # Stop if convergence is reached
            if inc_w < self.conv_stop:
                self.stop_training = True
            if kiter == self.Nmaxiter:
                self.stop_training = True
           
            message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, kiter, inc_w)
            self.display(message, True)
            kiter += 1
            self.w_old = self.w.copy()

        self.model.w = self.w
        self.model.niter = kiter
        self.model.is_trained = True
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

            if packet['action'] == 'ACK_exp_bl':
                self.exps_bl_encr_dict = packet['data']['exps_bl_dict']
                self.FSMmaster.done_compute_exp(self)

            if packet['action'] == 'ACK_div_bl':
                self.sigm_encr_bl_dict = packet['data']['sigm_encr_bl_dict']
                self.FSMmaster.done_compute_div(self)

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
class LC_Worker(Common_to_all_POMs):
    '''
    Class implementing Linear Regression, run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None, cryptonode_address=None):
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
class LC_Crypto(Common_to_all_POMs):
    '''
    Class implementing Logistic Classification, run at Crypto

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
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Crypto')
            import code
            code.interact(local=locals())
            '''
        return self.terminate
