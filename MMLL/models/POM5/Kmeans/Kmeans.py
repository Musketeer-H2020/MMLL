# -*- coding: utf-8 -*-
'''
Kmeans model under POM5

'''

__author__ = "Angel Navia-Vázquez  & Francisco González-Serrano"
__date__ = "Apr. 2020"

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
    Kmeans model.
    """
    def __init__(self):
        self.c = None
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
        X_b: ndarray
            Matrix with the input values

        Returns
        -------
        prediction_values: ndarray

        """
        XTC = np.dot(X, self.c.T)
        x2 = np.sum(X * X, axis=1).reshape((-1, 1))
        c2 = np.sum(self.c * self.c, axis=1).reshape((1, -1))
        D = x2 - 2 * XTC + c2
        predictions = np.argmin(D, axis=1)
        return predictions

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
                            from sklearn.cluster import KMeans
                            NC = self.c.shape[0]
                            NI = self.c.shape[1]
                            
                            export_model = KMeans(n_clusters=NC, random_state=0)
                            X = np.random.normal(0, 1, (100, NI))
                            export_model.fit(X)
                            export_model.cluster_centers_ = self.c

                            # Convert into ONNX format
                            input_type = [('float_input', FloatTensorType([None, NI]))]
                            onnx_model = convert_sklearn(export_model, initial_types=input_type)
                            with open(filename, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            print('=' * 80)
                            print('Model saved at %s in ONNX format.' %filename)
                            print('=' * 80)

                        elif extension == 'pmml':
                            
                            from sklearn.cluster import KMeans
                            NC = self.c.shape[0]
                            NI = self.c.shape[1]

                            export_model = KMeans(n_clusters=NC, random_state=0)
                            X = np.random.normal(0, 1, (100, NI))
                            export_model.fit(X)
                            export_model.cluster_centers_ = self.c

                            from sklearn2pmml import sklearn2pmml # pip install git+https://github.com/jpmml/sklearn2pmml.git
                            from sklearn2pmml.pipeline import PMMLPipeline
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

class Kmeans_Master(Common_to_all_POMs):
    """
    This class implements the Kmeans model, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`Kmeans_Master` instance.

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
        # we extract the model_parameters as extra kwargs, to be all jointly processed
        try:
            kwargs.update(kwargs['model_parameters'])
            del kwargs['model_parameters']
        except:
            pass
        self.process_kwargs(kwargs)
        self.model = Model()
        self.message_counter = 100    # used to number the messages
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

        '''
        path = '../MMLL/models/POM' + str(self.pom) + '/' + self.model_type + '/' 
        filename = path + 'POM' + str(self.pom) + '_' + self.model_type + '_FSM_master.pkl'
        with open(filename, 'rb') as f:
            [states_master, transitions_master] = pickle.load(f)
        '''
        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='update_tr_data', on_enter=['while_update_tr_data']),
            State(name='send_C_encr', on_enter=['while_send_C_encr']),
            State(name='crypto_loop', on_enter=['while_crypto_loop']),
            State(name='min_bl', on_enter=['while_min_bl']),
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_send_C_encr', 'waiting_order', 'send_C_encr'],
            ['done_send_C_encr', 'send_C_encr', 'waiting_order'],

            ['go_crypto_loop', 'waiting_order', 'crypto_loop'],
            ['done_crypto_loop', 'crypto_loop', 'waiting_order'],

            ['go_min_bl', 'crypto_loop', 'min_bl'],
            ['done_min_bl', 'min_bl', 'crypto_loop'],
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

                    #MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
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

            def while_send_C_encr(self, MLmodel):
                try:
                    data = {}
                    data.update({'C_encr': MLmodel.C_encr})
                    data.update({'C2_encr': MLmodel.C2_encr})
                    action = 'send_C_encr'
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    #MLmodel.comms.broadcast(packet, MLmodel.workers_addresses)
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    #MLmodel.comms.broadcast(packet, receivers_list=MLmodel.broadcast_addresses)
                    MLmodel.comms.broadcast(packet)
                    MLmodel.display(MLmodel.name + ' send_C_encr to workers')
                    #code.interact(local=locals())
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    raise
                    print('ERROR AT while_send_C_encr')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_crypto_loop(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' at while_crypto_loop')
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    raise
                    print('ERROR AT while_mult_loop')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_min_bl(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    distXC_encr_bl = packet['data']['distXC_encr_bl']
                    distXC_bl = MLmodel.decrypter.decrypt(distXC_encr_bl)
                    argmin = np.argmin(distXC_bl, axis=1)
                    
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    # sending back the argmin
                    sender = packet['sender']
                    action = 'sent_argmin'
                    data = {'argmin': argmin}
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.master_address}
                    # send back
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    worker_name = MLmodel.receive_from[sender]
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, worker_name, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, sender)
                    MLmodel.display(MLmodel.name + ': sent argmin')
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    raise
                    print('ERROR AT while_min_bl')
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
        self.display(self.name + ': Starting training')
        self.display('MASTER_INIT', verbose=False)

        # We take the first value, but all are available for consistency checking
        #self.NI = list(self.NI_dict.values())[0]
        self.NI = self.input_data_description['NI']
        self.C = np.random.normal(0, 0.1, (self.NC, self.NI))
        self.stop_training = False
        self.comms.send_to = self.send_to

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

        #self.send_to
        #self.worker_names
        #print(self.worker_names)

        if self.Xval is not None:
            message = 'WARNING: Validation data is not used during training.'
            self.display(message, True)

        kiter = 0
        while not self.stop_training:
            #print('Iteration = %d' % kiter)
            self.display('MASTER_ITER_START', verbose=False)

            self.C_encr = self.encrypter.encrypt(self.C)

            self.C2 = np.sum(self.C * self.C, axis=1).reshape((self.NC, 1))
            self.C2_encr = self.encrypter.encrypt(self.C2)

            # Storing the contributions from the workers
            self.C_inc_dict = {}
            self.N_inc_dict = {}

            self.FSMmaster.go_send_C_encr(self)
            self.FSMmaster.done_send_C_encr(self)
            
            self.FSMmaster.go_crypto_loop(self)
            self.run_Master()
            # FSMaster entra en estado "crypto_loop", desde aquí puede responder a los workers
            # Los workers comienzan cómputo y pueden pedir operaciones al Master
            # El Master espera a servir, Cuando tenga todos los ACK_, sale y sigue
            
            self.display('PROC_MASTER_START', verbose=False)
           
            newC = np.zeros((self.NC, self.NI))
            TotalP = np.zeros((self.NC, 1))

            for waddr in self.workers_addresses:
                cinc = self.C_inc_dict[waddr]
                ninc = self.N_inc_dict[waddr]
                for kc in range(0, self.NC):
                    try:
                        newC[kc, :] += cinc[kc]
                        TotalP[kc] += ninc[kc]
                    except:
                        pass
            
            for kc in range(0, self.NC):
                if TotalP[kc] > 0:
                    newC[kc, :] = newC[kc, :] / TotalP[kc]

            self.C_old = self.C.copy()
            self.C = newC

            self.display('---------------')
            l = [int(p) for p in list(TotalP.ravel())]
            self.display(str(l))

            # Evaluate if centroid realocation is needed
            Nmean = np.mean(TotalP)
            Overpop = list(np.where(TotalP > Nmean * 1.5)[0])
            Maxpop = np.argmax(TotalP)
            Overpop = list(set(Overpop + [Maxpop]))
            Underpop = list(np.where(TotalP <  0.3 * Nmean)[0])
            for which_under in Underpop:
                which_over = Overpop[np.random.randint(0, len(Overpop))]
                newc = self.C[which_over, :] + np.random.normal(0, 0.01, (1, self.NI))
                self.C[which_under, :] = newc.copy()
            self.display(str(Overpop) + ' ' + str(Underpop))
            self.display(str(len(Overpop)) + ' ' + str(len(Underpop)))
            self.display('---------------')

            '''
            '''

            inc_normc = np.linalg.norm(self.C_old - self.C)
            self.display(self.name + ': INC_C norm = %s' % str(inc_normc)[0:7])

            # Stop if convergence is reached
            if inc_normc < self.conv_stop:
                self.stop_training = True
            if kiter == self.Nmaxiter:
                self.stop_training = True

            message = 'Maxiter = %d, iter = %d, inc = %f' % (self.Nmaxiter, kiter, inc_normc)
            #self.display(message)
            print(message)

            kiter += 1

            self.display('PROC_MASTER_END', verbose=False)

            self.display('MASTER_ITER_END', verbose=False)

        self.model.c = self.C
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

        if self.chekAllStates('ACK_sending_C_inc'):
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
            try:
                self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass
            self.display(self.name + ': received %s from worker %s' % (packet['action'], sender), verbose=True)
            
            if packet['action'][0:3] == 'ACK':
                self.state_dict[sender] = packet['action']

            if packet['action'] == 'ACK_update_tr_data':
                #print('ProcessReceivedPacket_Master ACK_update_tr_data')
                self.newNI_dict.update({sender: packet['data']['newNI']})

            if packet['action'] == 'ask_min_bl':
                self.FSMmaster.go_min_bl(self, packet)
                self.FSMmaster.done_min_bl(self)

            if packet['action'] == 'ACK_sending_C_inc':
                self.C_inc_dict.update({sender: packet['data']['C_inc']})
                self.N_inc_dict.update({sender: packet['data']['N_inc']})
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
class Kmeans_Worker(Common_to_all_POMs):
    '''
    Class implementing Kmeans, run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`Kmeans_Worker` instance.

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
        self.Xtr_b = Xtr_b
        self.X2 = np.sum(Xtr_b * Xtr_b, axis = 1).reshape(-1, 1)
        #self.ytr = ytr
        self.NPtr = Xtr_b.shape[0]
        self.create_FSM_worker()
        self.message_id = 0    # used to number the messages
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
                    MLmodel.Xtr_b = MLmodel.Xtr_b.astype(float)
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

            def while_computing_distXC(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    MLmodel.display(MLmodel.name + ' %s: computing_distXC...' % (str(MLmodel.worker_address)))
                    MLmodel.C_encr = packet['data']['C_encr']
                    MLmodel.C2_encr = packet['data']['C2_encr']
                    X = MLmodel.Xtr_b
                    MLmodel.NP = X.shape[0]
                    MLmodel.NC = MLmodel.C_encr.shape[0]
                    
                    MLmodel.distXC_encr = -2 * np.dot(X, MLmodel.C_encr.T)
                    MLmodel.distXC_encr += np.ones((MLmodel.NP, MLmodel.NC)) * (MLmodel.C2_encr.T)
                    MLmodel.distXC_encr += np.ones((MLmodel.NP, MLmodel.NC)) * (MLmodel.X2)  # esto sería opcional, para dist min es irrelevante

                    # Adding blinding, different at every row, same for columns
                    b = np.random.normal(0, 5, (MLmodel.NP, 1))
                    MLmodel.bl = np.zeros((MLmodel.NP, MLmodel.NC))
                    MLmodel.bl[:, :] = b
                    #MLmodel.bl_encr = MLmodel.encrypter.encrypt(MLmodel.bl)
                    MLmodel.distXC_encr_bl = MLmodel.distXC_encr + MLmodel.bl

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_min_bl'
                    data = {'distXC_encr_bl': MLmodel.distXC_encr_bl}
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ask_min_bl' % (str(MLmodel.worker_address)))
                except Exception as err:
                    raise
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    print('ERROR AT while_computing_distXC')
                    import code
                    code.interact(local=locals())
                    '''
                    '''
                return

            def while_computing_incC(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    argmin = packet['data']['argmin']
                    C_inc = {}
                    N_inc = {}
                    
                    for kc in range(0, MLmodel.NC):
                        cuales = argmin == kc
                        Nchunk = np.sum(cuales)
                        if Nchunk > 2:
                            c_kc = np.sum(MLmodel.Xtr_b[cuales, :], axis=0)
                            C_inc.update({kc: c_kc})
                            N_inc.update({kc: Nchunk})

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_sending_C_inc'
                    data = {'C_inc': C_inc, 'N_inc': N_inc}
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}

                    '''
                    if MLmodel.worker_address == '3':
                        print('STOP AT while_computing_incC ----------------------  worker 3')
                        import code
                        code.interact(local=locals())
                    '''
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_C_inc' % (str(MLmodel.worker_address)))
                except Exception as err:
                    raise
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    print('ERROR AT while_computing_incC')
                    import code
                    code.interact(local=locals())
                    '''
                    '''
                return

        '''
        path = '../MMLL/models/POM' + str(self.pom) + '/' + self.model_type + '/' 
        filename = path + 'POM' + str(self.pom) + '_' + self.model_type + '_FSM_worker.pkl'
        with open(filename, 'rb') as f:
            [states_worker, transitions_worker] = pickle.load(f)
        '''
        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='computing_distXC', on_enter=['while_computing_distXC']),
            State(name='computing_incC', on_enter=['while_computing_incC'])
            ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_computing_distXC', 'waiting_order', 'computing_distXC'],

            ['go_computing_incC', 'computing_distXC', 'computing_incC'],
            ['done_computing_incC', 'computing_incC', 'waiting_order']

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

            if packet['action'] == 'send_C_encr':
                self.FSMworker.go_computing_distXC(self, packet)

            if packet['action'] == 'sent_argmin':
                self.FSMworker.go_computing_incC(self, packet)
                self.FSMworker.done_computing_incC(self)

        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Worker')
            import code
            code.interact(local=locals())
            '''
        return self.terminate
