# -*- coding: utf-8 -*-
'''
Kmeans under POM6

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Jan 2021"

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
        self.is_trained = True
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
        XTC = np.dot(X, self.c.T)
        x2 = np.sum(X * X, axis=1).reshape((-1, 1))
        c2 = np.sum(self.c * self.c, axis=1).reshape((1, -1))
        D = x2 - 2 * XTC + c2
        predictions = np.argmin(D, axis=1)
        return predictions

    def transform(self, X):
        """
        Transform data to distance space

        Parameters
        ----------
        X: ndarray
            Matrix with the input values

        Returns
        -------
        transformed data: ndarray

        """
        XTC = np.dot(X, self.c.T)
        x2 = np.sum(X * X, axis=1).reshape((-1, 1))
        c2 = np.sum(self.c * self.c, axis=1).reshape((1, -1))
        D2 = x2 - 2 * XTC + c2
        transf_X = np.sqrt(D2)
        return transf_X

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
    This class implements the Kmeans, run at Master node. It inherits from Common_to_all_POMs.
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

        self.model = Model()
        
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
            State(name='computing_DXTC', on_enter=['while_computing_DXTC']),
            State(name='computing_Cinc', on_enter=['while_computing_Cinc']),

            #State(name='sending_C', on_enter=['while_sending_C']),
            State(name='updating_C', on_enter=['while_updating_C']),
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_computing_DXTC', 'waiting_order', 'computing_DXTC'],
            ['go_waiting_order', 'computing_DXTC', 'waiting_order'],

            ['go_computing_Cinc', 'waiting_order', 'computing_Cinc'],
            ['go_waiting_order', 'computing_Cinc', 'waiting_order'],

            #['go_sending_C', 'waiting_order', 'sending_C'],
            #['go_waiting_order', 'sending_C', 'waiting_order'],

            ['go_updating_C', 'waiting_order', 'updating_C'],
            ['go_waiting_order', 'updating_C', 'waiting_order'],
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

            def while_computing_DXTC(self, MLmodel):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    action = 'computing_DXTC'
                    MLmodel.ACxaxb_dict = {}
                    xaxbP_dict = {}

                    for kc in range(MLmodel.NC):
                        x = MLmodel.model.c_dict[kc].T
                        NItrain = x.shape[1]
                        K = int(NItrain / 2)
                        # Guardar
                        tmp_dict = {}
                        tmp_dict.update({'A': np.random.uniform(-10, 10, K).reshape((1, K))})
                        tmp_dict.update({'C': np.random.uniform(-10, 10, K).reshape((1, K))})
                        tmp_dict.update({'xa': x[:, 0:K]})
                        tmp_dict.update({'xb': x[:, K:]})
                        MLmodel.ACxaxb_dict.update({kc: tmp_dict})

                        # Enviar
                        #xa_ = MLmodel.xa + MLmodel.A  
                        #xb_ = MLmodel.xb + MLmodel.C
                        #P = MLmodel.A + MLmodel.C   # warning, check the sum is nonzero (low prob...)
                        try:
                            tmp_dict = {}
                            tmp_dict.update({'xa_': MLmodel.ACxaxb_dict[kc]['xa'] + MLmodel.ACxaxb_dict[kc]['A']})
                            tmp_dict.update({'xb_': MLmodel.ACxaxb_dict[kc]['xb'] + MLmodel.ACxaxb_dict[kc]['C']})
                            tmp_dict.update({'P': MLmodel.ACxaxb_dict[kc]['A'] + MLmodel.ACxaxb_dict[kc]['C']})
                            xaxbP_dict.update({kc: tmp_dict})
                        except:
                            print('ERR HERE')
                            import code
                            code.interact(local=locals())

                            pass
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    # broadcasts xaxbP_dict
                    action = 'sending_xaxbP'
                    data = {'xaxbP_dict': xaxbP_dict, 'NC': MLmodel.NC}
                    del xaxbP_dict

                    #message_id = MLmodel.master_address + '_' + str(MLmodel.message_counter)
                    #packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address, 'message_id': message_id}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    del data

                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)
                   
                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': computing_DXTC with all Workers')
                    else:
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
                        MLmodel.display(MLmodel.name + ': computing_DXTC with Workers: %s' % str(MLmodel.selected_workers))
                
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    MLmodel.display('ERROR AT while_computing_DXTC')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_computing_Cinc(self, MLmodel):
                # received self.s_dict SDP for Xtr, one from every worker
                # MLmodel.ACxaxb_dict locally stored xa, xb, A, C for every binary classifier
                try:
                    action = 'computing_Cinc'
                    C2 = np.sum(MLmodel.model.c ** 2, axis=1)

                    for addr in MLmodel.workers_addresses:

                        MLmodel.display('PROC_MASTER_START', verbose=False)

                        NPtr = MLmodel.s_dict[addr]['ya_'].shape[0]
                        DXC = np.zeros((NPtr, MLmodel.NC))
                        for kc in range(MLmodel.NC):
                            xa = MLmodel.ACxaxb_dict[kc]['xa']
                            xb = MLmodel.ACxaxb_dict[kc]['xb']
                            A = MLmodel.ACxaxb_dict[kc]['A']
                            C = MLmodel.ACxaxb_dict[kc]['C']
                            #MLmodel.display('PROC_MASTER_START', verbose=False)
                            #U = MLmodel.s_dict[addr]['ya_'] * (MLmodel.xa + 2 * MLmodel.A) + MLmodel.s_dict[addr]['yb_'] * (MLmodel.xb + 2 * MLmodel.C) + MLmodel.s_dict[addr]['Q'] * (MLmodel.A + 2 * MLmodel.C)
                            U = MLmodel.s_dict[addr]['ya_'] * (xa + 2 * A) + MLmodel.s_dict[addr]['yb_'] * (xb + 2 * C) + MLmodel.s_dict[addr]['Q'] * (A + 2 * C)
                            del xa, xb, A, C
                            u = np.sum(U, axis=1)  
                            del U
                            s = u + MLmodel.s_dict[addr]['v_dict'][kc]
                            del u
                            DXC[:, kc] = C2[kc] - 2 * s
                    
                        nearest = np.argmin(DXC, axis = 1)
                        MLmodel.display('PROC_MASTER_END', verbose=False)
                        
                        action = 'sending_nearest'
                        data = {'nearest': nearest}
                        packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}

                        message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                        packet.update({'message_id': message_id})
                        MLmodel.message_counter += 1
                        size_bytes = asizeof.asizeof(dill.dumps(packet))
                        MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, addr, message_id, str(size_bytes)), verbose=False)

                        del data

                        MLmodel.comms.send(packet, MLmodel.send_to[addr])
                
                except Exception as err:
                    raise
                    '''
                    message = "ERROR: %s %s" % (str(err), str(type(err)))
                    MLmodel.display('\n ' + '='*50 + '\n' + message + '\n ' + '='*50 + '\n' )
                    MLmodel.display('ERROR AT while_computing_Cinc')
                    import code
                    code.interact(local=locals())
                    '''
                return

            '''
            def while_sending_C(self, MLmodel):
                try:
                    action = 'sending_C'
                    data = {'C': MLmodel.model.c}
                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.comms.broadcast(packet, MLmodel.selected_workers)
                    if MLmodel.selected_workers is None: 
                        MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')
                    else:
                        MLmodel.display(MLmodel.name + ': broadcasted C to Workers: %s' % str([MLmodel.receive_from[w] for w in MLmodel.selected_workers]))
                except Exception as err:
                    MLmodel.display('ERROR: %s %s' % (str(err), str(type(err))))
                    MLmodel.display('ERROR AT while_sending_C')
                    import code
                    code.interact(local=locals())         
                return
            '''

            def while_updating_C(self, MLmodel):
                #MLmodel.C_inc_dict
                try:

                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    NC = MLmodel.NC
                    NI = MLmodel.NI
                    newC = np.zeros((NC, NI))
                    TotalP = np.zeros((NC, 1))
                    #Dacum = np.zeros((NC, 1))

                    for user in MLmodel.workers_addresses:
                        cinc = MLmodel.Cinc_dict[user]
                        for kc in range(0, NC):
                            TotalP[kc] += cinc['Ninc'][kc]
                            if cinc['C_inc'][kc] is not None:
                                newC[kc, :] += cinc['C_inc'][kc]
                                '''
                                try:
                                    newC[kc, :] += cinc['C_inc'][kc]
                                except:
                                    pass
                                    print('ERR HERE')
                                    import code
                                    code.interact(local=locals())
                                '''


                    for kc in range(0, NC):
                        if TotalP[kc] > 0:
                            newC[kc, :] = newC[kc, :] / TotalP[kc]
                    
                    MLmodel.model.c = newC

                    MLmodel.display('---------------')
                    l = [int(p) for p in list(TotalP.ravel())]
                    MLmodel.display(str(l))
                    '''
                    print('---------------')
                    print(sum(TotalP))
                    print('---------------')
                    '''
                    # Evaluate if centroid realocation is needed
                    Nmean = np.mean(TotalP)
                    Overpop = list(np.where(TotalP > Nmean * 1.5)[0])
                    Maxpop = np.argmax(TotalP)
                    Overpop = list(set(Overpop + [Maxpop]))
                    Underpop = list(np.where(TotalP < 0.3 * Nmean)[0])
                    for which_under in Underpop:
                        which_over = Overpop[np.random.randint(0, len(Overpop))]
                        newc = MLmodel.model.c[which_over, :] + np.random.normal(0, 0.01, (1, NI))
                        MLmodel.model.c[which_under, :] = newc
                    MLmodel.display(str(Overpop) + ' ' + str(Underpop))
                    MLmodel.display(str(len(Overpop)) + ' ' + str(len(Underpop)))
                    MLmodel.display('---------------')
                    # passing MLmodel.model.c to dict
                    for kc in range(MLmodel.NC):
                        MLmodel.model.c_dict[kc] = MLmodel.model.c[kc, :].reshape((-1, 1))

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                except Exception as err:
                    raise
                    '''
                    MLmodel.display('ERROR: %s %s' % (str(err), str(type(err))))
                    MLmodel.display('ERROR AT while_updating_C')
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
        self.model.c = np.random.normal(0, 0.3, (self.NC, self.NI))      # centroids
        self.normx_dict = {}                                            # dictionary storing the norm2 of patterns
        self.Distc_dict = {}
        self.Cinc_dict = {}
        self.c_old = np.zeros((self.NC, self.NI))
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

        self.c_orig = np.copy(self.model.c)

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
            #if self.Xval_b is not None: 
            #    self.Xval_b = self.add_bias(self.Xval_b).astype(float)
            #    self.yval = self.yval.astype(float)

        self.Bob_data_s = False

       # Checking dimensions
        if int(self.NI / 2) != self.NI / 2: # add one value 
            self.c_orig_size = self.NI
            self.NItrain = self.NI + 1
            self.reset(self.NItrain)
        else:
            self.c_orig_size = self.NI
            self.NItrain = self.NI

        self.NI = self.NItrain

        self.model.c_dict = {}
        self.c_old_dict = {}

        for kc in range(self.NC):
            self.model.c_dict.update({kc: np.random.normal(0, 0.3, (self.NItrain, 1))})
            self.c_old_dict.update({kc: np.random.normal(0, 1.0, (self.NItrain, 1))})

        if self.Xval is not None:
            message = 'WARNING: Validation data is not used during training.'
            self.display(message, True)

        stop_training = False
        kiter = 0
        while not stop_training:
            self.display('MASTER_ITER_START', verbose=False)
            self.c_old = np.copy(self.model.c)

            # We XTw
            self.FSMmaster.go_computing_DXTC(self)
            self.run_Master()
            # We receive  self.s_dict, self.Ztr_dict (once)

            self.FSMmaster.go_computing_Cinc(self)
            self.run_Master()

            self.FSMmaster.go_updating_C(self)
            self.FSMmaster.go_waiting_order(self)

            inc_normc = np.linalg.norm(self.c_old - self.model.c) / np.linalg.norm(self.model.c)
            self.display(self.name + ': INC_C norm = %s' % str(inc_normc)[0:7])
            #print(self.Nmaxiter, kloop)
            message = 'Maxiter = %d, iter = %d, inc = %f' % (self.Nmaxiter, kiter, inc_normc)
            #self.display(message)
            print(message)
            if kiter == self.Nmaxiter or inc_normc < self.conv_stop:
                stop_training = True
            else:
                kiter += 1
            self.display('MASTER_ITER_END', verbose=False)

        self.display(self.name + ': Training is done')
        # reduciendo a dimensión original
        self.model.c = self.model.c[:, 0:self.c_orig_size]
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

        if self.chekAllStates('ACK_update_tr_data'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sending_XTC'):
            if not self.Bob_data_s:
                self.Bob_data_s = True
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sending_C_inc'):
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
            self.display(self.name + ': received ACK from %s: %s' % (str(sender), packet['action']))
            self.state_dict[sender] = packet['action']
            try:
                self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass

        if packet['action'] == 'ACK_update_tr_data':
            #print('ProcessReceivedPacket_Master ACK_update_tr_data from %s' % str(sender))
            self.newNI_dict.update({sender: packet['data']['newNI']})

        if packet['action'] == 'ACK_sending_XTC':
            if not self.Bob_data_s:
                self.s_dict.update({sender: {'ya_': packet['data']['ya_'], 'yb_': packet['data']['yb_'], 'Q': packet['data']['Q'], 'v_dict': packet['data']['v_dict']}})
                #self.Ztr_dict.update({sender: packet['data']['Ztr']})
            else:
                self.s_dict[sender]['v_dict'] = packet['data']['v_dict']

        if packet['action'] == 'ACK_sending_C_inc':
            self.Cinc_dict.update({sender : {'C_inc': packet['data']['C_inc'], 'Ninc': packet['data']['Ninc']}})

        return


#===============================================================
#                 Worker
#===============================================================
class Kmeans_Worker(Common_to_all_POMs):
    '''
    Class implementing Kmeans (public model), run at Worker

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
        #self.ytr = ytr
        self.NPtr = Xtr_b.shape[0]
        self.create_FSM_worker()
        self.message_id = 0    # used to number the messages
        self.cryptonode_address = None
        self.Bob_data_s = False
        self.Bob_data_grad = False
        self.added_bias=False
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
                    #raise
                    import code
                    code.interact(local=locals())
                    #MLmodel.display('ERROR AT while_computing_XTDaX')
                    '''


            def while_computing_XTC(self, MLmodel, packet):

                MLmodel.display('PROC_WORKER_START', verbose=False)

                #packet['data']['xaxbP_dict']
                #packet['data']['classes']
                MLmodel.NC = packet['data']['NC']

                try:
                    #MLmodel.display('PROC_WORKER_START', verbose=False)
                    # Only once
                    if not MLmodel.Bob_data_s:
                        # Checking NI
                        NI = MLmodel.Xtr_b.shape[1]
                        MLmodel.NPtr = MLmodel.Xtr_b.shape[0]
                        self.NI = NI

                        if NI/2 != int(NI/2):
                            MLmodel.Xtr_b = np.hstack((MLmodel.Xtr_b, np.random.normal(0, 0.01, (MLmodel.NPtr, 1))))

                        NPtr_train = MLmodel.Xtr_b.shape[0]
                        NI_train = MLmodel.Xtr_b.shape[1]
                        self.NI = NI_train

                        # RMD
                        #MLmodel.Cmat = np.random.normal(0, 1, (NI_train, NI_train))
                        #MLmodel.Dmat = np.linalg.inv(MLmodel.Cmat)
                        #MLmodel.Ztr = np.dot(MLmodel.Xtr_b, MLmodel.Cmat)

                        K = int(NI_train / 2)

                        #y = MLmodel.Xtr_b
                        MLmodel.yas = MLmodel.Xtr_b[:, 0:K]
                        MLmodel.ybs = MLmodel.Xtr_b[:, K:]
                        #del y

                        MLmodel.Bs = np.random.uniform(-10, 10, (NPtr_train, K))
                        MLmodel.Ds = np.random.uniform(-10, 10, (NPtr_train, K))
                        # Send once
                        MLmodel.Qs = MLmodel.Bs - MLmodel.Ds    # warning, check the sum is nonzero (low prob...)
                        MLmodel.ya_s = MLmodel.Bs - MLmodel.yas  
                        MLmodel.yb_s = MLmodel.Ds - MLmodel.ybs

                    r = np.random.uniform(-10, 10, MLmodel.NPtr)
                    v_dict = {}
                    for kc in range(MLmodel.NC):
                        xa_ = packet['data']['xaxbP_dict'][kc]['xa_']
                        xb_ = packet['data']['xaxbP_dict'][kc]['xb_']
                        P = packet['data']['xaxbP_dict'][kc]['P']
                        V = xa_ * (2 * MLmodel.yas - MLmodel.Bs) + xb_ * (2 * MLmodel.ybs - MLmodel.Ds) + P * (MLmodel.Ds - 2 * MLmodel.Bs)
                        # Dot product masking
                        v = np.sum(V, axis=1) + r
                        v_dict.update({kc: v})
                    del xa_, xb_, P, V

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    # send to Master ya_, yb_, Q, v
                    action = 'ACK_sending_XTC'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    if not MLmodel.Bob_data_s:
                        data = {'ya_': MLmodel.ya_s, 'yb_': MLmodel.yb_s, 'Q': MLmodel.Qs, 'v_dict': v_dict}
                        MLmodel.Bob_data_s = True
                    else:
                        data = {'v_dict': v_dict}

                    del v_dict

                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    del data

                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    del packet#, size_bytes
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_XTC' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_XTC')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_computing_incC(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    pred = packet['data']['nearest']
                    MLmodel.display(MLmodel.name + ' %s: Computing incC' % (str(MLmodel.worker_address)))
                    
                    C_inc = []
                    Ninc = []
                    #Dist_acum = []

                    for kc in range(0, MLmodel.NC):
                        cuales = pred == kc
                        Nchunk = np.sum(cuales)

                        if Nchunk > 2:
                            c_kc = np.sum(MLmodel.Xtr_b[cuales,:], axis=0)
                            C_inc.append(c_kc)
                            Ninc.append(Nchunk)
                        else: # preserve privacy not sending updates averaged with less than 3 patterns
                            c_kc = None
                            C_inc.append(c_kc)
                            # we send just the number of patterns per cluster
                            Ninc.append(Nchunk)

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_sending_C_inc'
                    data = {'C_inc': C_inc, 'Ninc':Ninc}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}

                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_sending_C_inc' % (str(MLmodel.worker_address)))
                except Exception as err:
                    raise
                    '''
                    MLmodel.display('ERROR: %s %s' % (str(err), str(type(err))))
                    MLmodel.display('ERROR AT while_computing_incC')
                    import code
                    code.interact(local=locals())   
                    '''      
                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='computing_XTC', on_enter=['while_computing_XTC']),

            State(name='computing_incC', on_enter=['while_computing_incC']),
            State(name='Exit', on_enter=['while_Exit']),
           ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_computing_XTC', 'waiting_order', 'computing_XTC'],
            ['done_computing_XTC', 'computing_XTC', 'waiting_order'],

            ['go_computing_incC', 'waiting_order', 'computing_incC'],
            ['done_computing_incC', 'computing_incC', 'waiting_order'],

            ['go_exit', 'waiting_order', 'Exit']
            ]

        self.FSMworker = FSM_worker()
        self.grafmachine_worker = GraphMachine(model=self.FSMworker,
            states=states_worker,
            transitions=transitions_worker,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of worker No. %s" % str(self.worker_address),
            show_conditions=True)
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

        if packet['action'] == 'sending_xaxbP':
            self.FSMworker.go_computing_XTC(self, packet)
            self.FSMworker.done_computing_XTC(self)

        if packet['action'] == 'sending_nearest':
            self.FSMworker.go_computing_incC(self, packet)
            self.FSMworker.done_computing_incC(self)

        if packet['action'] == 'update_tr_data':
            # We update the training data
            self.FSMworker.go_setting_tr_data(self, packet)
            self.FSMworker.done_setting_tr_data(self)

        return self.terminate
