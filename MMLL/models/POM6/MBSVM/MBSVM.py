# -*- coding: utf-8 -*-
'''
Multiclass Budget Support Vector Machine under POM6

'''
__author__ = "Angel Navia-Vázquez"
__date__ = "Apr. 2021"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
from sklearn.metrics import roc_curve, auc
from pympler import asizeof #asizeof.asizeof(my_object)
import pickle
import dill
import time

class Model():
    """
    Multiclass Budget Support Vector Machine.
    """
    def __init__(self):
        self.C = None
        self.w_dict = {}
        self.sigma = None
        self.Csvm = None
        self.classes = None
        self.is_trained = False
        self.supported_formats = ['pkl']
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
        # bias
        KXC = np.hstack( (np.ones((NP, 1)), KXC))

        preds_dict = {}
        NCLA = len(self.classes)
        O = []
        for cla in self.classes:
            o = np.dot(KXC, self.w_dict[cla]).ravel()
            preds_dict.update({cla: o})
            O.append(o)

        O = np.array(O)
        winners = list(np.argmax(O, axis=0))
        o = np.array([self.classes[pos] for pos in winners]).ravel() 
        return o

    def predict_soft(self, X):
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
        # bias
        KXC = np.hstack( (np.ones((NP, 1)), KXC))

        preds_dict = {}
        NCLA = len(self.classes)
        O = []
        for cla in self.classes:
            o = np.dot(KXC, self.w_dict[cla]).ravel()
            preds_dict.update({cla: o})
            O.append(o)

        O = np.array(O)
        winners = list(np.argmax(O, axis=0))
        o = np.array([self.classes[pos] for pos in winners]).ravel() 
        return preds_dict

    def save(self, filename=None):
        """
        Saves the trained model to file. The valid file extensions are:            
            - "pkl": saves the model as a Python3 pickle file       

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
                    except:
                        print('=' * 80)
                        print('Model Save Error: model cannot be saved at %s, please check the provided path/filename.' %filename)
                        print('=' * 80)
                        raise

class MBSVM_Master(Common_to_all_POMs):
    """
    This class implements the Multiclass Budget Support Vector Machine, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`BSVM_Master` instance.

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
        self.epsilon = 0.00000001  # to avoid log(0)
        self.landa = 0.5
        
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
        self.model.Csvm = self.Csvm
        self.Kacum_dict = {}
        self.grady_dict = {}
        self.s0_dict = {}
        self.s1_dict = {}
        self.grads_dict = {}
        self.Ztr_dict = {}
        self.NPtr_dict = {}
        self.eps = 0.0000001

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

            State(name='computing_XTw', on_enter=['while_computing_XTw']),
            State(name='computing_oi', on_enter=['while_computing_oi']),

            State(name='updating_w', on_enter=['while_updating_w'])
        ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_selecting_C', 'waiting_order', 'selecting_C'],
            ['go_waiting_order', 'selecting_C', 'waiting_order'],

            ['go_sending_C', 'waiting_order', 'sending_C'],
            ['go_waiting_order', 'sending_C', 'waiting_order'],

            ['go_computing_oi', 'waiting_order', 'computing_oi'],
            ['go_waiting_order', 'computing_oi', 'waiting_order'],

            ['go_getting_KTK', 'waiting_order', 'getting_KTK'],
            ['go_waiting_order', 'getting_KTK', 'waiting_order'],

            ['go_computing_XTw', 'waiting_order', 'computing_XTw'],
            ['go_waiting_order', 'computing_XTw', 'waiting_order'],


            ['go_updating_w', 'waiting_order', 'updating_w'],
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

                    if MLmodel.selected_workers is None: 
                        MLmodel.comms.broadcast(packet)
                        MLmodel.display(MLmodel.name + ': broadcasted C to all Workers')
                    else:
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
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
                    data = {'C': MLmodel.model.C, 'sigma': MLmodel.model.sigma, 'Csvm': MLmodel.model.Csvm, 'classes': MLmodel.classes}
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
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
                        MLmodel.display(MLmodel.name + ': broadcasted C to Workers: %s' % str([MLmodel.receive_from[w] for w in MLmodel.selected_workers]))

                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_sending_C')
                    import code
                    code.interact(local=locals())
                    '''         
                return

            def while_computing_XTw(self, MLmodel):
                try:
                    #action = 'computing_XTw'
                    MLmodel.ACxaxb_dict = {}
                    xaxbP_dict = {}

                    for cla in MLmodel.classes:
                        MLmodel.display('PROC_MASTER_START', verbose=False)
                        MLmodel.x = MLmodel.model.w_dict[cla].T
                        NItrain = MLmodel.x.shape[1]
                        K = int(NItrain / 2)
                        # Guardar
                        tmp_dict = {}
                        tmp_dict.update({'A': np.random.uniform(-10, 10, K).reshape((1, K))})
                        tmp_dict.update({'C': np.random.uniform(-10, 10, K).reshape((1, K))})
                        tmp_dict.update({'xa': MLmodel.x[:, 0:K]})
                        tmp_dict.update({'xb': MLmodel.x[:, K:]})
                        MLmodel.ACxaxb_dict.update({cla: tmp_dict})

                        # Enviar
                        #xa_ = MLmodel.xa + MLmodel.A  
                        #xb_ = MLmodel.xb + MLmodel.C
                        #P = MLmodel.A + MLmodel.C   # warning, check the sum is nonzero (low prob...)
                        tmp_dict = {}
                        tmp_dict.update({'xa_': MLmodel.ACxaxb_dict[cla]['xa'] + MLmodel.ACxaxb_dict[cla]['A']})
                        tmp_dict.update({'xb_': MLmodel.ACxaxb_dict[cla]['xb'] + MLmodel.ACxaxb_dict[cla]['C']})
                        tmp_dict.update({'P': MLmodel.ACxaxb_dict[cla]['A'] + MLmodel.ACxaxb_dict[cla]['C']})
                        xaxbP_dict.update({cla: tmp_dict})
                        MLmodel.display('PROC_MASTER_END', verbose=False)

                    # broadcasts xaxbP_dict
                    action = 'sending_xaxbP'
                    data = {'xaxbP_dict': xaxbP_dict, 'classes': MLmodel.classes}
                    del xaxbP_dict

                    packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                    del data
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

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
                # MLmodel.s0_dict, MLmodel.s1_dict 
                try:
                    MLmodel.o_dict = {}
                    
                    for addr in MLmodel.workers_addresses:
                        MLmodel.display('PROC_MASTER_START', verbose=False)
                        # We need to compute these and send them to every worker:                    
                        Rzz_dict = {}
                        rzt_dict = {}

                        for cla in MLmodel.classes:
                            #MLmodel.display('PROC_MASTER_START', verbose=False)

                            U0 = MLmodel.s0_dict[addr]['ya_0_dict'][cla] * (MLmodel.ACxaxb_dict[cla]['xa'] + 2 * MLmodel.ACxaxb_dict[cla]['A']) + MLmodel.s0_dict[addr]['yb_0_dict'][cla] * (MLmodel.ACxaxb_dict[cla]['xb'] + 2 * MLmodel.ACxaxb_dict[cla]['C']) + MLmodel.s0_dict[addr]['Q0_dict'][cla] * (MLmodel.ACxaxb_dict[cla]['A'] + 2 * MLmodel.ACxaxb_dict[cla]['C'])
                            u0 = np.sum(U0, axis=1)  
                            del U0
                            s0 = u0 + MLmodel.s0_dict[addr]['v0_dict'][cla]
                            del u0

                            NPtr0 = s0.shape[0]
                            y0 = -np.ones(NPtr0)
                            e0 = y0 - s0
                            del s0

                            # Weighting values a
                            a0 = np.ones(NPtr0)
                            ey0 = e0 * y0
                            which0 = ey0 >= MLmodel.eps
                            a0[which0] = 2 * MLmodel.Csvm / ey0[which0]
                            which0 = ey0 < MLmodel.eps
                            a0[which0] = 2 * MLmodel.Csvm / MLmodel.eps
                            which0 = ey0 < 0
                            a0[which0] = 0
                            a0 = a0.reshape((-1, 1))
                            del ey0, e0, which0

                            Rzz0 = MLmodel.Ztr_dict[addr]['Ztr0_dict'][cla] * a0
                            rzt0 = np.dot(Rzz0.T, y0)
                            Rzz0 = np.dot(Rzz0.T, MLmodel.Ztr_dict[addr]['Ztr0_dict'][cla])
                            del a0, y0

                            #U1 = MLmodel.s1_dict[addr]['ya_1'] * (MLmodel.xa + 2 * MLmodel.A) + MLmodel.s1_dict[addr]['yb_1'] * (MLmodel.xb + 2 * MLmodel.C) + MLmodel.s1_dict[addr]['Q1'] * (MLmodel.A + 2 * MLmodel.C)
                            U1 = MLmodel.s1_dict[addr]['ya_1_dict'][cla] * (MLmodel.ACxaxb_dict[cla]['xa'] + 2 * MLmodel.ACxaxb_dict[cla]['A']) + MLmodel.s1_dict[addr]['yb_1_dict'][cla] * (MLmodel.ACxaxb_dict[cla]['xb'] + 2 * MLmodel.ACxaxb_dict[cla]['C']) + MLmodel.s1_dict[addr]['Q1_dict'][cla] * (MLmodel.ACxaxb_dict[cla]['A'] + 2 * MLmodel.ACxaxb_dict[cla]['C'])

                            u1 = np.sum(U1, axis=1)  
                            del U1
                            s1 = u1 + MLmodel.s1_dict[addr]['v1_dict'][cla]
                            del u1

                            NPtr1 = s1.shape[0]
                            y1 = np.ones(NPtr1)
                            e1 = y1 - s1
                            del s1

                            # Weighting values a
                            a1 = np.ones(NPtr1)
                            ey1 = e1 * y1
                            which1 = ey1 >= MLmodel.eps
                            a1[which1] = 2 * MLmodel.Csvm / ey1[which1]
                            which1 = ey1 < MLmodel.eps
                            a1[which1] = 2 * MLmodel.Csvm / MLmodel.eps
                            which1 = ey1 < 0
                            a1[which1] = 0
                            a1 = a1.reshape((-1, 1))
                            del ey1, e1, which1

                            Rzz1 = MLmodel.Ztr_dict[addr]['Ztr1_dict'][cla] * a1
                            rzt1 = np.dot(Rzz1.T, y1)
                            Rzz1 = np.dot(Rzz1.T, MLmodel.Ztr_dict[addr]['Ztr1_dict'][cla])
                            del a1, y1

                            Rzz_dict.update({cla: Rzz0 + Rzz1})
                            rzt_dict.update({cla: rzt0 + rzt1})

                            # Needed ?
                            #MLmodel.NPtr_dict.update({addr: NPtr0 + NPtr1})
                        MLmodel.display('PROC_MASTER_END', verbose=False)

                        action = 'sending_Rzz_rzt'
                        data = {'Rzz_dict': Rzz_dict, 'rzt_dict': rzt_dict}
                        #del Rzz0, Rzz1, rzt0, rzt1
                         
                        packet = {'action': action, 'to': 'MLmodel', 'data': data, 'sender': MLmodel.master_address}
                        del data

                        message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                        packet.update({'message_id': message_id})
                        MLmodel.message_counter += 1
                        size_bytes = asizeof.asizeof(dill.dumps(packet))
                        MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, addr, message_id, str(size_bytes)), verbose=False)

                        MLmodel.comms.send(packet, MLmodel.send_to[addr])
                        #del packet, size_bytes, message_id
                        del packet
                        MLmodel.display(MLmodel.name + ' %s: sent sending_Rzz to %s' % (str(MLmodel.master_address), str(addr)))
                    
                    #del MLmodel.xa, MLmodel.xb, MLmodel.A, MLmodel.C

                except:
                    raise
                    '''
                    print('ERROR AT while_computing_oi')
                    import code
                    code.interact(local=locals())
                    '''
                return


            def while_getting_KTK(self, MLmodel):
                try:
                    action = 'compute_KTK'
                    data = {'w': MLmodel.model.w}
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
                        recipients = [MLmodel.send_to[w] for w in MLmodel.selected_workers]
                        MLmodel.comms.broadcast(packet, recipients)
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

                MLmodel.display('PROC_MASTER_START', verbose=False)

                MLmodel.w_old_dict = dict(MLmodel.model.w_dict)
                try:
                    MLmodel.w_new_dict = {}
                    for cla in MLmodel.classes:
                        MLmodel.KTK_accum = np.zeros((MLmodel.NItrain, MLmodel.NItrain))
                        MLmodel.KTy_accum = np.zeros((MLmodel.NItrain, 1))
                        for waddr in MLmodel.workers_addresses:
                            MLmodel.KTK_accum += MLmodel.KTK_dict[waddr][cla]
                            MLmodel.KTy_accum += MLmodel.KTy_dict[waddr][cla].reshape((-1, 1))

                        #MLmodel.model.w_dict[cla] = np.dot(np.linalg.inv(MLmodel.KTK_accum + MLmodel.Kcc), MLmodel.KTy_accum)        
                        MLmodel.w_new_dict[cla] = np.dot(np.linalg.inv(MLmodel.KTK_accum + MLmodel.Kcc), MLmodel.KTy_accum)        
                
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

        self.FSMmaster.go_update_tr_data(self)
        self.run_Master()

        self.display('PROC_MASTER_START', verbose=False)

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

        '''
        self.FSMmaster.go_selecting_C(self)
        self.run_Master()

        # Selecting centroids with largest projection
        Ncandidates = self.C.shape[0]
        Kacum_total = np.zeros(Ncandidates)
        for addr in self.workers_addresses:
            Kacum_total += self.Kacum_dict[addr].ravel()

        index = np.argsort(-Kacum_total)

        self.C = self.C[index[0: self.NC], :]
        '''
        self.model.C = self.C
        self.NC = self.C.shape[0]       

        # computing Kcc, only once
        X = self.model.C
        XC2 = -2 * np.dot(X, self.model.C.T)
        XC2 += np.sum(np.multiply(X, X), axis=1).reshape((self.NC, 1))
        XC2 += np.sum(np.multiply(self.model.C, self.model.C), axis=1).reshape((1, self.NC))
        KCC = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2))
        self.Kcc = np.zeros((self.NC + 1, self.NC + 1))
        self.Kcc[1:, 1:] = KCC
        self.Kcc[0, 0] = 0.00001

        self.Bob_data_s = False
        self.Bob_data_grad = False

        self.NI = self.NC + 1
        # Checking dimensions
        if int(self.NI / 2) != self.NI / 2: # add one value 
            self.w_orig_size = self.NI
            self.NItrain = self.NI + 1
            # Adding row and column to Kcc
            self.Kcc = np.hstack((self.Kcc, np.zeros((self.NI, 1))))
            self.Kcc = np.vstack((self.Kcc, np.zeros((1, self.NI + 1))))
            self.Kcc[self.NI, self.NI] = 1.0
        else:
            self.w_orig_size = self.NI
            self.NItrain = self.NI


        # Computing and storing KXC_val
        if self.Xval is not None:
            XC2 = -2 * np.dot(self.Xval, self.C.T)
            XC2 += np.sum(np.multiply(self.Xval, self.Xval), axis=1).reshape((-1, 1))
            XC2 += np.sum(np.multiply(self.C, self.C), axis=1).reshape((1, self.NC))
            # Gauss
            KXC_val = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2))
            self.KXC_val = np.hstack( (np.ones((self.Xval.shape[0], 1)), KXC_val)) # NP_val x NC + 1

            #self.yval.astype(float).reshape((-1, 1))

        self.stop_training = False
        self.kiter = 0

        self.display('PROC_MASTER_END', verbose=False)

        self.FSMmaster.go_sending_C(self)
        self.run_Master()

        self.ceval_acum = 100

       # Checking dimensions
        if int(self.NI / 2) != self.NI / 2: # add one value 
            self.w_orig_size = self.NI
            self.NItrain = self.NI + 1
        else:
            self.w_orig_size = self.NI
            self.NItrain = self.NI

        self.model.w_dict = {}
        self.w_old_dict = {}
        self.model.classes = self.classes

        for cla in self.classes:
            self.model.w_dict.update({cla: np.random.normal(0, 0.001, (self.NItrain, 1))})
            self.w_old_dict.update({cla: np.random.normal(0, 0.001, (self.NItrain, 1))})

        self.ACC_val = 0
        self.ACC_val_old = 0

        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)

            self.FSMmaster.go_computing_XTw(self)
            self.run_Master()
            # We receive  self.s_dict, self.Ztr_dict (once)

            self.FSMmaster.go_computing_oi(self)
            self.run_Master()

            # Updating w
            self.FSMmaster.go_updating_w(self)
            self.FSMmaster.go_waiting_order(self)

            self.display('PROC_MASTER_START', verbose=False)

            self.kiter += 1
            # Stop if Maxiter is reached
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            if self.Xval is None:  # A validation set is not provided
                for cla in self.classes:
                    self.model.w_dict[cla] = (1 - self.landa)  * self.w_old_dict[cla] + self.landa * self.w_new_dict[cla]

                inc_w = 0
                for cla in self.classes:
                    inc_w += np.linalg.norm(self.model.w_dict[cla] - self.w_old_dict[cla]) / np.linalg.norm(self.w_old_dict[cla])

                message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
                #self.display(message)
                print(message)
            
                if inc_w <= self.conv_stop:
                    self.stop_training = True

            else:
                self.ceval_acum_old = self.ceval_acum
                NIval = self.KXC_val.shape[1]

                O = []
                for cla in self.classes:                   
                    #w_ = self.model.w_dict[cla][0: NIval]
                    w_ = ((1 - self.landa) * self.w_old_dict[cla] + self.landa * self.w_new_dict[cla])[0: NIval]
                    o_val = np.dot(self.KXC_val, w_).ravel()
                    O.append(o_val)

                O = np.array(O)
                winners = list(np.argmax(O, axis=0))
                preds_val = np.array([self.classes[pos] for pos in winners]).ravel()

                ACC_val = np.mean(preds_val.ravel() == self.yval)
                if ACC_val > self.ACC_val_old: 
                    # retain the new
                    for cla in self.classes:
                        self.model.w_dict[cla] = (1 - self.landa) * self.w_old_dict[cla] + self.landa * self.w_new_dict[cla]
                    self.ACC_val_old = ACC_val
                    message = 'Maxiter = %d, iter = %d, ACC val = %f' % (self.Nmaxiter, self.kiter, ACC_val)
                    print(message)
                else: # restore the previous one and stop
                    self.model.w_dict = dict(self.w_old_dict)
                    self.stop_training = True

                '''
                self.ceval_acum = 0
                for cla in self.classes:
                    yval = np.array(self.yval == cla).astype(float).reshape((-1, 1))
                    w_ = self.model.w_dict[cla][0: NIval]
                    w_old_ = self.w_old_dict[cla][0: NIval]

                    CE_val = []
                    landas = np.arange(0, 1.0, 0.001)
                    
                    Xw = np.dot(self.KXC_val, w_)
                    Xw_old = np.dot(self.KXC_val, w_old_)
                    
                    for landa in landas:
                        w_tmp = landa * w_ + (1 - landa) * w_old_
                        o_tmp = landa * Xw + (1 - landa) * Xw_old
                        ce_val = np.mean(self.cross_entropy(self.sigm(o_tmp), yval, self.epsilon))
                        CE_val.append(ce_val)

                    min_pos = np.argmin(CE_val)
                    CEval_opt = CE_val[min_pos]
                    indices = np.array(range(len(landas)))[CE_val == CEval_opt]
                    min_pos = indices[0]  # first
                    landa_opt = landas[min_pos]
                    self.ceval_acum += CEval_opt
                    self.model.w_dict[cla] = (1.0 - landa_opt) * self.w_old_dict[cla] + landa_opt * self.model.w_dict[cla]
                    message = 'Class = %s, landa_opt = %f' % (cla, landa_opt)
                    self.display(message)
                self.ceval_acum = self.ceval_acum / len(self.classes)

                if self.ceval_acum < self.ceval_acum_old:  
                    message = 'Maxiter = %d, iter = %d, CE val = %f' % (self.Nmaxiter, self.kiter, self.ceval_acum)
                    print(message)
                    self.w_old_dict = dict(self.model.w_dict)
                else:
                    self.stop_training = True
                    # We retain the last weight values
                    self.model.w_dict = dict(self.w_old_dict)
                '''

            self.display('PROC_MASTER_END', verbose=False)

            self.display('MASTER_ITER_END', verbose=False)

        self.display(self.name + ': Training is done')
        self.model.niter = self.kiter
        self.model.is_trained = True

        # reduciendo a dimensión original
        for cla in self.classes:
            self.model.w_dict[cla] = self.model.w_dict[cla][0:self.w_orig_size, :]

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

        if self.chekAllStates('ACK_sending_s'):
            if not self.Bob_data_s:
                self.Bob_data_s = True
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_sending_KTK'):
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

                if packet['action'] == 'ACK_sending_s':
                    if not self.Bob_data_s:
                        self.s0_dict.update({sender: {'ya_0_dict': packet['data']['ya_0_dict'], 'yb_0_dict': packet['data']['yb_0_dict'], 'Q0_dict': packet['data']['Q0_dict'], 'v0_dict': packet['data']['v0_dict']}})
                        self.s1_dict.update({sender: {'ya_1_dict': packet['data']['ya_1_dict'], 'yb_1_dict': packet['data']['yb_1_dict'], 'Q1_dict': packet['data']['Q1_dict'], 'v1_dict': packet['data']['v1_dict']}})
                        self.Ztr_dict.update({sender: {'Ztr0_dict': packet['data']['Ztr0_dict'], 'Ztr1_dict': packet['data']['Ztr1_dict']}})
                    else:
                        self.s0_dict[sender]['v0_dict'] = packet['data']['v0_dict']
                        self.s1_dict[sender]['v1_dict'] = packet['data']['v1_dict']

                if packet['action'] == 'ACK_sending_KTK':
                    self.KTK_dict.update({sender: packet['data']['KTK_dict']})
                    self.KTy_dict.update({sender: packet['data']['KTy_dict']})


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
class MBSVM_Worker(Common_to_all_POMs):
    '''
    Class implementing Multiclass Budget Support Vector Machine, run at Worker

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=True, Xtr_b=None, ytr=None):
        """
        Create a :class:`BSVM_Worker` instance.

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
        self.message_id = 0    # used to number the messages
        self.eps = 0.0000001

        self.Bob_data_s = False
        self.Bob_data_grad = False
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

                    #Kacum contains the number of closest patterns                    
                    winners = list(np.argmax(KXC, axis=1))
                    Kacum = np.zeros((NC, 1))
                    for kc in range(NC):
                        Kacum[kc] = winners.count(kc)

                    #Kacum = np.sum(KXC, axis = 0)
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
                    MLmodel.Csvm = packet['data']['Csvm']
                    MLmodel.classes = packet['data']['classes']

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
                    # Poly
                    #KXC = 1 / (1 + (XC2 / 2.0 /  (MLmodel.sigma ** 2)  ) ) 
                    MLmodel.KXC = np.hstack( (np.ones((NP, 1)), KXC)) # NP x NC + 1
                    print('KXC min max', np.min(MLmodel.KXC), np.max(MLmodel.KXC))

                    # Checking NI
                    NI = MLmodel.KXC.shape[1]
                    NPtr = MLmodel.KXC.shape[0]

                    if NI/2 != int(NI/2):
                        MLmodel.KXC = np.hstack((MLmodel.KXC, np.random.normal(0, 0.01, (MLmodel.NPtr, 1))))

                    MLmodel.NPtr_train = MLmodel.KXC.shape[0]
                    MLmodel.NI_train = MLmodel.KXC.shape[1]

                    # RMD
                    MLmodel.Cmat_dict = {}
                    MLmodel.Dmat_dict = {}
                    MLmodel.Ztr0_dict = {}
                    MLmodel.Ztr1_dict = {}

                    for cla in MLmodel.classes:
                        MLmodel.Cmat_dict.update({cla: np.random.normal(0, 1, (MLmodel.NI_train, MLmodel.NI_train))})
                        MLmodel.Dmat_dict.update({cla: np.linalg.inv(MLmodel.Cmat_dict[cla])})
                        ytr = np.array(MLmodel.ytr == cla).astype(float).reshape((-1, 1))
                        which0 = (ytr == 0).ravel()
                        MLmodel.Ztr0_dict.update({cla: np.dot(MLmodel.KXC[which0, :], MLmodel.Cmat_dict[cla])})
                        which1 = (ytr == 1).ravel()
                        MLmodel.Ztr1_dict.update({cla: np.dot(MLmodel.KXC[which1, :], MLmodel.Cmat_dict[cla])})

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

            def while_computing_s(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    MLmodel.classes = packet['data']['classes']
                    if not MLmodel.Bob_data_s:
                        NI_train = MLmodel.KXC.shape[1]
                        NPtr_train = MLmodel.Xtr_b.shape[0]
                        # MLmodel.Cmat_dict
                        # MLmodel.Dmat_dict
                        # MLmodel.Ztr0_dict
                        # MLmodel.Ztr1_dict
                        K = int(NI_train / 2)

                        MLmodel.yas0_dict = {}
                        MLmodel.yas1_dict = {}
                        MLmodel.ybs0_dict = {}
                        MLmodel.ybs1_dict = {}
                        MLmodel.Bs0_dict = {}
                        MLmodel.Ds0_dict = {}
                        MLmodel.Bs1_dict = {}
                        MLmodel.Ds1_dict = {}
                        # Send once
                        MLmodel.Qs0_dict = {}
                        MLmodel.Qs1_dict = {}
                        MLmodel.ya_s0_dict = {}
                        MLmodel.yb_s0_dict = {}
                        MLmodel.ya_s1_dict = {}
                        MLmodel.yb_s1_dict = {}

                        for cla in MLmodel.classes:
                            ytr = np.array(MLmodel.ytr == cla).astype(float).reshape((-1, 1))
                            which0 = (ytr == 0).ravel()
                            NPtr0 = np.sum(which0)
                            which1 = (ytr == 1).ravel()
                            NPtr1 = np.sum(which1)
                            aux = MLmodel.KXC[:, 0:K]
                            MLmodel.yas0_dict.update({cla: aux[which0, :]})
                            MLmodel.yas1_dict.update({cla: aux[which1, :]})
                            aux = MLmodel.KXC[:, K:]
                            MLmodel.ybs0_dict.update({cla: aux[which0, :]})
                            MLmodel.ybs1_dict.update({cla: aux[which1, :]})

                            MLmodel.Bs0_dict.update({cla: np.random.uniform(-10, 10, (NPtr0, K))}) 
                            MLmodel.Ds0_dict.update({cla: np.random.uniform(-10, 10, (NPtr0, K))}) 
                            MLmodel.Bs1_dict.update({cla: np.random.uniform(-10, 10, (NPtr1, K))}) 
                            MLmodel.Ds1_dict.update({cla: np.random.uniform(-10, 10, (NPtr1, K))}) 
                            
                            # Send once
                            #MLmodel.Qs = MLmodel.Bs - MLmodel.Ds    # warning, check the sum is nonzero (low prob...)
                            MLmodel.Qs0_dict.update({cla: MLmodel.Bs0_dict[cla] - MLmodel.Ds0_dict[cla]})
                            MLmodel.Qs1_dict.update({cla: MLmodel.Bs1_dict[cla] - MLmodel.Ds1_dict[cla]})

                            #MLmodel.ya_s = MLmodel.Bs - MLmodel.yas  
                            MLmodel.ya_s0_dict.update({cla: MLmodel.Bs0_dict[cla] - MLmodel.yas0_dict[cla]})
                            MLmodel.ya_s1_dict.update({cla: MLmodel.Bs1_dict[cla] - MLmodel.yas1_dict[cla]})
                            
                            #MLmodel.yb_s = MLmodel.Ds - MLmodel.ybs
                            MLmodel.yb_s0_dict.update({cla: MLmodel.Ds0_dict[cla] - MLmodel.ybs0_dict[cla]})
                            MLmodel.yb_s1_dict.update({cla: MLmodel.Ds1_dict[cla] - MLmodel.ybs1_dict[cla]})

                    v0_dict = {}
                    v1_dict = {}
                    for cla in MLmodel.classes:
                        xa_ = packet['data']['xaxbP_dict'][cla]['xa_']
                        xb_ = packet['data']['xaxbP_dict'][cla]['xb_']
                        P = packet['data']['xaxbP_dict'][cla]['P']
                        V0 = xa_ * (2 * MLmodel.yas0_dict[cla] - MLmodel.Bs0_dict[cla]) + xb_ * (2 * MLmodel.ybs0_dict[cla] - MLmodel.Ds0_dict[cla]) + P * (MLmodel.Ds0_dict[cla] - 2 * MLmodel.Bs0_dict[cla])
                        v0 = np.sum(V0, axis=1)
                        v0_dict.update({cla: v0})
                        V1 = xa_ * (2 * MLmodel.yas1_dict[cla] - MLmodel.Bs1_dict[cla]) + xb_ * (2 * MLmodel.ybs1_dict[cla] - MLmodel.Ds1_dict[cla]) + P * (MLmodel.Ds1_dict[cla] - 2 * MLmodel.Bs1_dict[cla])
                        v1 = np.sum(V1, axis=1)
                        v1_dict.update({cla: v1})

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    # send to Master ya_, yb_, Q, v
                    action = 'ACK_sending_s'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    if not MLmodel.Bob_data_s:
                        data = {'ya_0_dict': MLmodel.ya_s0_dict, 'yb_0_dict': MLmodel.yb_s0_dict, 'Q0_dict': MLmodel.Qs0_dict, 'v0_dict': v0_dict, 'Ztr0_dict': MLmodel.Ztr0_dict, 'Ztr1_dict': MLmodel.Ztr1_dict}
                        data.update({'ya_1_dict': MLmodel.ya_s1_dict, 'yb_1_dict': MLmodel.yb_s1_dict, 'Q1_dict': MLmodel.Qs1_dict, 'v1_dict': v1_dict})
                        MLmodel.Bob_data_s = True
                    else:
                        data = {'v0_dict': v0_dict, 'v1_dict': v1_dict}

                    #del v0, v1

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


            def while_computing_KTK(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    KTK_dict = {}
                    KTy_dict = {}

                    for cla in MLmodel.classes:
                        KTK =  np.dot(MLmodel.Dmat_dict[cla].T, packet['data']['Rzz_dict'][cla]) 
                        KTK =  np.dot(KTK, MLmodel.Dmat_dict[cla])
                        KTK_dict.update({cla: KTK})
                        KTy =  np.dot(MLmodel.Dmat_dict[cla].T, packet['data']['rzt_dict'][cla]) 
                        KTy_dict.update({cla: KTy})

                    MLmodel.display('PROC_WORKER_END', verbose=False)
                  
                    action = 'ACK_sending_KTK'
                    data = {'KTK_dict': KTK_dict, 'KTy_dict': KTy_dict}
                    #del KTK, KTy
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    #MLmodel.comms.send(MLmodel.master_address, packet)
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

                return

        states_worker = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='setting_tr_data', on_enter=['while_setting_tr_data']),
            State(name='projecting_C', on_enter=['while_projecting_C']),
            State(name='storing_C', on_enter=['while_storing_C']),

            State(name='computing_s', on_enter=['while_computing_s']),

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

            ['go_computing_s', 'waiting_order', 'computing_s'],
            ['done_computing_s', 'computing_s', 'waiting_order'],

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
                    self.display('EXIT_WORKER')
                    self.terminate = True

                if packet['action'] == 'update_tr_data':
                    # We update the training data
                    self.FSMworker.go_setting_tr_data(self, packet)
                    self.FSMworker.done_setting_tr_data(self)

                if packet['action'] == 'compute_KTK':
                    self.FSMworker.go_computing_KTK(self, packet)          
                    self.FSMworker.done_computing_KTK(self)

                if packet['action'] == 'selecting_C':
                    #self.C = packet['data']['C']
                    self.FSMworker.go_projecting_C(self, packet)
                    self.FSMworker.done_projecting_C(self)

                if packet['action'] == 'sending_C':
                    #self.C = packet['data']['C']
                    self.FSMworker.go_storing_C(self, packet)
                    self.FSMworker.done_storing_C(self)

                if packet['action'] == 'sending_xaxbP':
                    self.FSMworker.go_computing_s(self, packet)
                    self.FSMworker.done_computing_s(self)

                if packet['action'] == 'sending_Rzz_rzt':
                    self.FSMworker.go_computing_KTK(self, packet)
                    self.FSMworker.done_computing_KTK(self)

            except Exception as err:
                raise
                '''
                print('ERROR AT CheckNewPacket_worker')
                import code
                code.interact(local=locals())
                '''

        return self.terminate
