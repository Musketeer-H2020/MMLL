# -*- coding: utf-8 -*-
'''
Kernel Regression under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Jan. 2021"

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
        self.pom = 6
        self.model_type = model_type
        self.name = self.model_type + '_Master'                 # Name
        self.master_address = master_address
        self.workers_addresses = workers_addresses
        
        try:
            kwargs.update(kwargs['model_parameters'])
            del kwargs['model_parameters']
        except Exception as err:
            pass
        self.process_kwargs(kwargs)

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
        self.verbose = verbose                      # print on screen when true
        self.NI = None       

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
        
        self.KTK_dict = {}
        self.KTy_dict = {}
        #self.NC = self.C.shape[0]
        self.NI = self.C.shape[1]
        self.newNI_dict = {}
        self.model = Model()
        self.model.C = self.C
        self.model.sigma = np.sqrt(self.NI) * self.fsigma 
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
            State(name='getting_KTK', on_enter=['while_getting_KTK']),
            State(name='selecting_C', on_enter=['while_selecting_C']),
            State(name='sending_C', on_enter=['while_sending_C']),
            State(name='updating_w', on_enter=['while_updating_w'])
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

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
                MLmodel.display(MLmodel.name + ': WAITING for instructions...')
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

                    MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                    if MLmodel.selected_workers is None: 
                        MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')
                    else:
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

                    MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                    if MLmodel.selected_workers is None: 
                        MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')
                    else:
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

                    MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                    if MLmodel.selected_workers is None: 
                        MLmodel.display(MLmodel.name + ': broadcasted compute_KTK to all workers')
                    else:
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
        self.model.w = np.random.normal(0, 0.001, (self.NI + 1, 1))      # weights in plaintext, first value is bias
        self.w_old = np.random.normal(0, 1.0, (self.NI + 1, 1))
        self.XTDaX_accum = np.zeros((self.NI + 1, self.NI + 1))    # Cov. matrix in plaintext
        self.XTDast_accum = np.zeros((self.NI + 1, 1))              # Cov. matrix in plaintext
        self.preds_dict = {}                                           # dictionary storing the prediction errors
        self.XTX_dict = {}
        self.XTy_dict = {}
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
        self.display('MASTER_ITER_START', verbose=False)

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

        if self.Xval is not None:
            message = 'WARNING: Validation data is not used during training.'
            self.display(message, True)

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
        self.model.C = self.C
        self.NC = self.C.shape[0]

        self.FSMmaster.go_sending_C(self)
        self.run_Master()

        self.FSMmaster.go_getting_KTK(self)
        self.run_Master()

        self.display(self.name + ': Training is done')
        self.model.niter = 1
        self.model.is_trained = True
        self.display('MASTER_ITER_END', verbose=False)
        self.display('MASTER_FINISH', verbose=False)

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_update_tr_data'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_projecting_C'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_storing_C'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sending_KTK'):
            self.FSMmaster.go_updating_w(self)
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
        if packet is not None:
            try:
                #sender = packet['sender']
                sender = self.receive_from[packet['sender']]

                if packet['action'][0:3] == 'ACK':
                    self.display(self.name + ': received ACK from %s: %s' % (str(sender), packet['action']))
                    self.state_dict[sender] = packet['action']
                    try:
                        self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
                    except:
                        self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                        pass

                if packet['action'] == 'ACK_update_tr_data':
                    self.newNI_dict.update({sender: packet['data']['newNI']})

                if packet['action'] == 'ACK_projecting_C':
                    self.Kacum_dict.update({sender: packet['data']['Kacum']})

                if packet['action'] == 'ACK_sending_KTK':
                    self.KTK_dict.update({sender: packet['data']['KTK']})
                    self.KTy_dict.update({sender: packet['data']['KTy']})


            except Exception as err:
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
    Class implementing Kernel Regression (private model), run at Worker

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
        self.NPtr = len(ytr)
        self.w = None
        self.create_FSM_worker()
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

            def while_waiting_order(self, MLmodel):
                MLmodel.display(MLmodel.name + ' %s: WAITING for instructions...' % (str(MLmodel.worker_address)))
                return

            def while_setting_tr_data(self, MLmodel, packet):
                try:
                    NPtr, newNI = MLmodel.Xtr_b.shape
                    #MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
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

            def while_computing_KTK(self, MLmodel):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    KTK = np.dot(MLmodel.KXC.T, MLmodel.KXC)
                    KTy = np.dot(MLmodel.KXC.T, MLmodel.ytr)
                    NC = KTK.shape[0] - 1
                    w = np.dot(np.linalg.inv(KTK + 0.1 * np.eye(NC + 1)), KTy)        
                    out = np.dot(MLmodel.KXC, w)
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_sending_KTK'
                    data = {'KTK': KTK, 'KTy': KTy}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_KTK' % (str(MLmodel.worker_address)))
                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_computing_KTK')
                    import code
                    code.interact(local=locals())         
                    pass
                    '''

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
            State(name='projecting_C', on_enter=['while_projecting_C']),
            State(name='storing_C', on_enter=['while_storing_C']),
            State(name='computing_KTK', on_enter=['while_computing_KTK']),
            State(name='computing_KXC', on_enter=['while_computing_KXC']),
            State(name='Exit', on_enter=['while_Exit'])
        ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_projecting_C', 'waiting_order', 'projecting_C'],
            ['done_projecting_C', 'projecting_C', 'waiting_order'],

            ['go_storing_C', 'waiting_order', 'storing_C'],
            ['done_storing_C', 'storing_C', 'waiting_order'],

            ['go_computing_KXC', 'waiting_order', 'computing_KXC'],
            ['done_computing_KXC', 'computing_KXC', 'waiting_order'],

            ['go_computing_KTK', 'waiting_order', 'computing_KTK'],
            ['done_computing_KTK', 'computing_KTK', 'waiting_order'],

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

        if packet is not None:
            try:
                # Exit the process
                if packet['action'] == 'STOP':
                    self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
                    self.terminate = True

                if packet['action'] == 'update_tr_data':
                    # We update the training data
                    self.FSMworker.go_setting_tr_data(self, packet)
                    self.FSMworker.done_setting_tr_data(self)

                if packet['action'] == 'compute_KTK':
                    self.FSMworker.go_computing_KTK(self)          
                    self.FSMworker.done_computing_KTK(self)

                if packet['action'] == 'selecting_C':
                    #self.C = packet['data']['C']
                    self.FSMworker.go_projecting_C(self, packet)
                    self.FSMworker.done_projecting_C(self)

                if packet['action'] == 'sending_C':
                    #self.C = packet['data']['C']
                    self.FSMworker.go_storing_C(self, packet)
                    self.FSMworker.done_storing_C(self)

            except Exception as err:
                raise
                '''
                print('ERROR AT CheckNewPacket_worker')
                import code
                code.interact(local=locals())
                '''

        return self.terminate
