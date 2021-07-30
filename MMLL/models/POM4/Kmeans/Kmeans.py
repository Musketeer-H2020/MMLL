# -*- coding: utf-8 -*-
'''
Kmeans model under POM4

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "May 2020"

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
        self.C = None
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
        self.message_counter = 100    # used to number the messages
        self.XTX_dict = {}
        self.XTy_dict = {}
        #self.encrypter = self.cr.get_encrypter()  # to be shared        # self.encrypter.encrypt(np.random.normal(0, 1, (2,3)))
        #self.decrypter = self.cr.get_decrypter()  # to be kept as secret  self.encrypter.decrypt()
        self.create_FSM_master()
        self.FSMmaster.master_address = master_address
        self.model = Model()
        self.added_bias = False
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
            State(name='store_Xyblinded', on_enter=['while_store_Xyblinded']),
            State(name='mult_XB', on_enter=['while_mult_XB']),
            State(name='decrypt_model', on_enter=['while_decrypt_model']),
            State(name='mult_AB', on_enter=['while_mult_AB']),
            State(name='compute_argmin', on_enter=['while_compute_argmin'])
        ]

        transitions_master = [
            ['go_store_Xyblinded', 'waiting_order', 'store_Xyblinded'],
            ['done_store_Xyblinded', 'store_Xyblinded', 'waiting_order'],

            ['go_mult_XB', 'waiting_order', 'mult_XB'],
            ['done_mult_XB', 'mult_XB', 'waiting_order'],

            ['go_decrypt_model', 'waiting_order', 'decrypt_model'],
            ['done_decrypt_model', 'decrypt_model', 'waiting_order'],

            ['go_mult_AB', 'waiting_order', 'mult_AB'],
            ['done_mult_AB', 'mult_AB', 'waiting_order'],

            ['go_compute_argmin', 'waiting_order', 'compute_argmin'],
            ['done_compute_argmin', 'compute_argmin', 'waiting_order']
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

            def while_compute_argmin(self, MLmodel, c2_2XTC_dict):
                try:

                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    # Adding blinding, different at every row, same for columns
                    MLmodel.bl_dict = {}
                    MLmodel.c2_2XTC_bl_dict = {}
                    for waddr in MLmodel.workers_addresses:
                        NP = c2_2XTC_dict[waddr].shape[0]
                        MLmodel.bl_dict.update({waddr: np.random.normal(0, 5, (NP, 1))})
                        MLmodel.c2_2XTC_bl_dict.update({waddr: c2_2XTC_dict[waddr] + MLmodel.bl_dict[waddr]})

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ask_argmin_bl'
                    data = {'c2_2XTC_bl_dict': MLmodel.c2_2XTC_bl_dict}
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.master_address}

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
                    print('ERROR AT Kmeans master while_compute_argmin')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return


            def while_decrypt_model(self, MLmodel, model_encr):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
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
                    
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    data = {'model_bl': model_encr_bl}
                    action = 'send_model_encr_bl'
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    
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
        self.display(self.name + ': Starting training')
        self.display('MASTER_INIT', verbose=False)

        self.NI = self.input_data_description['NI']
        self.C = np.random.normal(0, 0.5, (self.NC, self.NI))
        self.C_old = np.random.normal(0, 10, (self.NC, self.NI))

        self.selected_workers = self.workers_addresses

        # Data at:
        #self.X_encr_dict

        if self.Xval is not None:
            message = 'WARNING: Validation data is not used during training.'
            self.display(message, True)

        kiter = 0
        self.stop_training = False
        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)
            self.display('PROC_MASTER_START', verbose=False)

            self.C2 = np.sum(self.C * self.C, axis=1).reshape((self.NC, 1))

            self.c2_2XTC_dict = {}
            for waddr in self.workers_addresses:
                X = self.X_encr_dict[waddr]
                # X contains bias, removing it
                #X = X[:,1:]

                XTC = np.dot(X, self.C.T)
                c2_2XTC = self.C2.T - 2 * XTC
                self.c2_2XTC_dict.update({waddr: c2_2XTC})

            self.display('PROC_MASTER_END', verbose=False)

            # Añadiendo blinding y enviado a Crypto para calcular el minimo
            self.FSMmaster.go_compute_argmin(self, self.c2_2XTC_dict)
            self.run_Master()
            # self.argmin_dict

            self.display('PROC_MASTER_START', verbose=False)

            newC_encr = self.encrypter.encrypt(np.zeros((self.NC, self.NI)))
            TotalP = np.zeros((self.NC, 1))

            #self.X_encr_dict
            # self.argmin_dict
            for waddr in self.workers_addresses:
                for kc in range(0, self.NC):
                    try:
                        which = self.argmin_dict[waddr] == kc
                        aux = self.X_encr_dict[waddr][which, :]
                        # aux contains bias, removing it
                        #aux = aux[:,1:]
                        Nselected = aux.shape[0]
                        newC_encr[kc, :] += np.sum(aux, axis=0)
                        TotalP[kc] += Nselected
                    except:
                        pass
            self.display('PROC_MASTER_END', verbose=False)
            
            # Decrypting the model
            newC = self.decrypt_model({'C': newC_encr})['C']

            self.display('PROC_MASTER_START', verbose=False)

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
            Underpop = list(np.where(TotalP < 0.3 * Nmean)[0])
            for which_under in Underpop:
                which_over = Overpop[np.random.randint(0, len(Overpop))]
                newc = self.C[which_over, :] + np.random.normal(0, 0.01, (1, self.NI))
                self.C[which_under, :] = newc.copy()
            self.display(str(Overpop) + ' ' + str(Underpop))
            self.display(str(len(Overpop)) + ' ' + str(len(Underpop)))
            self.display('---------------')

            inc_normc = np.linalg.norm(self.C_old - self.C)
            self.display(self.name + ': INC_C norm = %s' % str(inc_normc)[0:7])

            # Stop if convergence is reached
            if inc_normc < self.conv_stop:
                self.stop_training = True
            if kiter == self.Nmaxiter:
                self.stop_training = True

            message = 'Maxiter = %d, iter = %d, inc = %f' % (self.Nmaxiter, kiter, inc_normc)
            print(message)
            #self.display(message, True)
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
            #sender = self.receive_from[sender]
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

            if packet['action'] == 'ACK_compute_argmin':
                self.argmin_dict = packet['data']['argmin_dict']
                self.FSMmaster.done_compute_argmin(self)

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

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None, cryptonode_address=None):
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
        self.NPtr = Xtr_b.shape[0]
        #self.ytr = ytr
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
                    packet = {'action': action, 'sender': MLmodel.worker_address, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}
                    
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
        if packet['action'] not in ['STOP']: 
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
class Kmeans_Crypto(Common_to_all_POMs):
    '''
    Class implementing Logistic Regression, run at Crypto

    '''

    def __init__(self, cryptonode_address, master_address, model_type, comms, logger, verbose=True):
        """
        Create a :class:`Kmeans_Crypto` instance.

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
        self.name = model_type + '_Worker'    # Name
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

        transitions_crypto = [
        ]

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
        if packet['action'] not in ['STOP']:
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

        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Crypto')
            import code
            code.interact(local=locals())
            '''

        return self.terminate
