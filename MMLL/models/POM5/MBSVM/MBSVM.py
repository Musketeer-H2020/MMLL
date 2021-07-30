# -*- coding: utf-8 -*-
'''
Multiclass Budget Support Vector Machine under POM5

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Apr. 2021"

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
    Multiclass Budget Support Vector Machine model.
    """
    def __init__(self):
        self.C = None
        self.w_dict = None
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
        X_b: ndarray
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
        o = [self.classes[pos] for pos in winners] 
        return preds_dict, o

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
    This class implements the Multiclass Budget Support Vector Machine model, run at Master node. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, model_type, comms, logger, verbose=True, **kwargs):
        """
        Create a :class:`MBSVM_Master` instance.

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
        self.landa = 0.5
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
        self.KTK_dict = {}
        self.KTy_dict = {}
        #self.NC = self.C.shape[0]
        self.NI = self.C.shape[1]
        self.newNI_dict = {}

        self.model = Model()
        self.model.C = self.C
        self.model.sigma = np.sqrt(self.NI) * self.fsigma 
        self.model.Csvm = self.Csvm
        self.NPtr_dict = {}
        self.eps = 0.0000001
        self.Kacum_dict = {}
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
            State(name='selecting_C', on_enter=['while_selecting_C']),
            State(name='sending_C', on_enter=['while_sending_C']),

            State(name='send_w_encr', on_enter=['while_send_w_encr']),
            
            State(name='crypto_loop', on_enter=['while_crypto_loop']),
            State(name='comp_ai', on_enter=['while_comp_ai']),
            
            State(name='updating_w', on_enter=['while_updating_w']),

            State(name='comp_exp_bl', on_enter=['while_comp_exp_bl']),
            State(name='comp_div_bl', on_enter=['while_comp_div_bl'])
         ]

        transitions_master = [
            ['go_update_tr_data', 'waiting_order', 'update_tr_data'],
            ['go_waiting_order', 'update_tr_data', 'waiting_order'],

            ['go_selecting_C', 'waiting_order', 'selecting_C'],
            ['go_waiting_order', 'selecting_C', 'waiting_order'],

            ['go_sending_C', 'waiting_order', 'sending_C'],
            ['go_waiting_order', 'sending_C', 'waiting_order'],

            ['go_send_w_encr', 'waiting_order', 'send_w_encr'],
            ['go_waiting_order', 'send_w_encr', 'waiting_order'],

            ['go_update_w', 'waiting_order', 'updating_w'],
            ['go_waiting_order', 'updating_w', 'waiting_order'],


            ['go_crypto_loop', 'waiting_order', 'crypto_loop'],
            ['done_crypto_loop', 'crypto_loop', 'waiting_order'],

            ['go_comp_ai', 'crypto_loop', 'comp_ai'],
            ['done_comp_ai', 'comp_ai', 'crypto_loop'],

            ['go_comp_exp_bl', 'crypto_loop', 'comp_exp_bl'],
            ['done_comp_exp_bl', 'comp_exp_bl', 'crypto_loop'],

            ['go_comp_div_bl', 'crypto_loop', 'comp_div_bl'],
            ['done_comp_div_bl', 'comp_div_bl', 'crypto_loop'],

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

            def while_send_w_encr(self, MLmodel):
                try:
                    data = {}
                    data.update({'w_encr_dict': MLmodel.w_encr_dict, 'classes': MLmodel.classes})
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


            def while_comp_ai(self, MLmodel, packet):
                try:
                    MLmodel.display(MLmodel.name + ' is computing ai...')
                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    s1_encr_dict = packet['data']['s1_encr_dict']
                    s_1_encr_dict = packet['data']['s_1_encr_dict']
                    sender = packet['sender']
                    del packet

                    a1_encr_dict = {}
                    a_1_encr_dict = {}

                    for cla in MLmodel.classes:
                        NPtr1 = s1_encr_dict[cla].shape[0]
                        y1 = -np.ones(NPtr1)
                        e1 = y1 - MLmodel.decrypter.decrypt(s1_encr_dict[cla]).ravel()
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
                        a1_encr_dict.update({cla: MLmodel.encrypter.encrypt(a1) })
                        del ey1, e1, a1, which1

                        NPtr_1 = s_1_encr_dict[cla].shape[0]
                        y_1 = -np.ones(NPtr_1)
                        e_1 = y_1 - MLmodel.decrypter.decrypt(s_1_encr_dict[cla]).ravel()
                        # Weighting values a
                        a_1 = np.ones(NPtr_1)
                        ey_1 = e_1 * y_1
                        which_1 = ey_1 >= MLmodel.eps
                        a_1[which_1] = 2 * MLmodel.Csvm / ey_1[which_1]
                        which_1 = ey_1 < MLmodel.eps
                        a_1[which_1] = 2 * MLmodel.Csvm / MLmodel.eps
                        which_1 = ey_1 < 0
                        a_1[which_1] = 0
                        a_1 = a_1.reshape((-1, 1))
                        a_1_encr_dict.update({cla: MLmodel.encrypter.encrypt(a_1) })
                        del ey_1, e_1, a_1, which_1

                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ACK_sent_ai_encr'                   
                    data = {'a1_encr_dict': a1_encr_dict, 'a_1_encr_dict': a_1_encr_dict}
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

                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_comp_ai')
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

                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    s_bl = MLmodel.decrypter.decrypt(s_encr_bl)
                    exp_s_bl = np.exp(-s_bl)
                    exp_s_bl_encr = MLmodel.encrypter.encrypt(exp_s_bl)
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'ACK_sent_exp_s_bl_encr'
                    data = {'exp_s_bl_encr': exp_s_bl_encr}
                    del exp_s_bl_encr
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
                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    MLmodel.display(MLmodel.name + ' is computing blinded division...')
                    num_bl = packet['data']['num_bl']
                    den_bl_encr = packet['data']['den_bl_encr']
                    sender = packet['sender']
                    del packet     
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


            def while_updating_w(self, MLmodel):
                try:
                    MLmodel.display('PROC_MASTER_START', verbose=False)
                    NC = MLmodel.model.C.shape[0]
                    MLmodel.w_old_dict = dict(MLmodel.model.w_dict)
                    MLmodel.w_new_dict = dict(MLmodel.model.w_dict)

                    for cla in MLmodel.classes:
                        KTDaK_accum = np.zeros((NC + 1, NC + 1))
                        KTDay_accum = np.zeros((NC + 1, 1))
                        for waddr in MLmodel.workers_addresses:
                            KTDaK_accum += MLmodel.decrypter.decrypt(MLmodel.KTDaK_dict[waddr][cla])
                            KTDay_accum += MLmodel.decrypter.decrypt(MLmodel.KTDay_dict[waddr][cla].reshape((-1, 1)))

                        MLmodel.w_new_dict[cla] = np.dot(np.linalg.inv(KTDaK_accum + MLmodel.Kcc), KTDay_accum)        

                    del KTDaK_accum, KTDay_accum
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                except Exception as err:
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
        
        self.display('PROC_MASTER_START', verbose=False)

        # computing Kcc, only once
        X = self.model.C
        XC2 = -2 * np.dot(X, self.model.C.T)
        XC2 += np.sum(np.multiply(X, X), axis=1).reshape((self.NC, 1))
        XC2 += np.sum(np.multiply(self.model.C, self.model.C), axis=1).reshape((1, self.NC))
        KCC = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2))
        self.Kcc = np.zeros((self.NC + 1, self.NC + 1))
        self.Kcc[1:, 1:] = KCC
        self.Kcc[0, 0] = 1.0

        # Computing and storing KXC_val
        if self.Xval is not None:
            XC2 = -2 * np.dot(self.Xval, self.C.T)
            XC2 += np.sum(np.multiply(self.Xval, self.Xval), axis=1).reshape((-1, 1))
            XC2 += np.sum(np.multiply(self.C, self.C), axis=1).reshape((1, self.NC))
            # Gauss
            KXC_val = np.exp(-XC2 / 2.0 /  (self.model.sigma ** 2))
            self.KXC_val = np.hstack( (np.ones((self.Xval.shape[0], 1)), KXC_val)) # NP_val x NC + 1

            self.yval.reshape((-1, 1))

        self.model.w_dict = {}
        self.w_encr_dict = {}
        self.w_old_dict = {}
        self.model.classes = self.classes
        for cla in self.classes:
            self.model.w_dict.update({cla: np.random.normal(0, 0.001, (self.NC + 1, 1))})
            self.w_encr_dict.update({cla: self.encrypter.encrypt(self.model.w_dict[cla])})
            #self.w_old_dict.update({cla: np.random.normal(0, 1.0, (self.NC + 1, 1))})

        self.display('PROC_MASTER_END', verbose=False)

        self.FSMmaster.go_sending_C(self)
        self.run_Master()     

        self.stop_training = False
        self.kiter = 0
        self.ACC_val = 0
        self.ACC_val_old = 0

        while not self.stop_training:
            self.display('MASTER_ITER_START', verbose=False)
            t_ini = time.time()
            self.w_old_dict = dict(self.model.w_dict)

            self.s1_dict = {}
            self.s_1_dict = {}
            self.KTDaK_dict = {}
            self.KTDay_dict = {}

            self.FSMmaster.go_send_w_encr(self)
            self.FSMmaster.go_waiting_order(self)

            # FSMaster en estado "crypto_loop", desde aquí puede responder a los workers
            # Los workers comienzan cómputo y pueden pedir operaciones al Master
            self.FSMmaster.go_crypto_loop(self)
            # El Master espera a servir de cryptoprocessor. Cuando tenga todos los ACK_grads, sale y sigue
            self.run_Master()

            self.FSMmaster.go_update_w(self)
            self.FSMmaster.go_waiting_order(self)

            self.display('PROC_MASTER_START', verbose=False)           

            self.kiter += 1
            # Stop if Maxiter is reached
            if self.kiter == self.Nmaxiter:
                self.stop_training = True

            if self.Xval is None:  # A validation set is not provided
                inc_w = 0
                for cla in self.classes:
                    self.model.w_dict[cla] = (1 - self.landa) * self.w_old_dict[cla] + self.landa * self.model.w_dict[cla]
                    inc_w += np.linalg.norm(self.model.w_dict[cla] - self.w_old_dict[cla]) / np.linalg.norm(self.w_old_dict[cla])

                # Stop if convergence is reached
                if inc_w < self.conv_stop:
                    self.stop_training = True     

                message = 'Maxiter = %d, iter = %d, inc_w = %f' % (self.Nmaxiter, self.kiter, inc_w)
                print(message)

            else:
                NIval = self.KXC_val.shape[1]
                # We keep the previos models
                w_dict = dict(self.model.w_dict)
                w_old_dict = dict(self.w_old_dict)

                # Updated model
                for cla in self.classes:
                    w_ = ((1 - self.landa) * w_old_dict[cla] + self.landa * self.w_new_dict[cla])[0: NIval]
                    self.model.w_dict[cla] = w_

                preds_dict = {}
                NCLA = len(self.classes)
                O = []
                for cla in self.classes:
                    o = np.dot(self.KXC_val, self.model.w_dict[cla]).ravel()
                    preds_dict.update({cla: o})
                    O.append(o)

                O = np.array(O)
                winners = list(np.argmax(O, axis=0))
                preds_val = [self.classes[pos] for pos in winners] 
                ACC_val = np.mean(np.array(preds_val).ravel() == (self.yval.ravel()))

                if ACC_val > self.ACC_val_old: 
                    # retain the new
                    #self.model.w = (1 - self.landa) * self.w_old + self.landa * self.new_w
                    self.ACC_val_old = ACC_val
                    self.w_old_dict = dict(self.model.w_dict)
                    message = 'Maxiter = %d, iter = %d, ACC val = %f' % (self.Nmaxiter, self.kiter, ACC_val)
                    print(message)
                else: # restore the previous one and stop
                    self.model.w_dict = dict(self.w_old_dict)
                    self.stop_training = True

            for cla in self.classes:
                self.w_encr_dict[cla] = self.encrypter.encrypt(self.model.w_dict[cla])

            self.display('PROC_MASTER_END', verbose=False)
            self.display('MASTER_ITER_END', verbose=False)
            elapsed_time = time.time() - t_ini
            print('Time iter (minutes) = %f' % (elapsed_time/60.0))

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

        if self.chekAllStates('ACK_update_tr_data'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_projecting_C'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_storing_C'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_KTDaK'):
            self.FSMmaster.done_crypto_loop(self)


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
            
            if packet['action'] == 'ACK_update_tr_data':
                #print('ProcessReceivedPacket_Master ACK_update_tr_data')
                self.newNI_dict.update({sender: packet['data']['newNI']})

            if packet['action'] == 'ACK_projecting_C':
                self.Kacum_dict.update({sender: packet['data']['Kacum']})

            if packet['action'] == 'ACK_KTDaK':
                self.KTDaK_dict.update({sender: packet['data']['KTDaK_dict']})
                self.KTDay_dict.update({sender: packet['data']['KTDay_dict']})

            if packet['action'] == 'ask_ai':
                self.FSMmaster.go_comp_ai(self, packet)
                self.FSMmaster.done_comp_ai(self)

            if packet['action'] == 'ask_exp_bl':
                self.FSMmaster.go_comp_exp_bl(self, packet)
                self.FSMmaster.done_comp_exp_bl(self)

            if packet['action'] == 'ask_div_bl':
                self.FSMmaster.go_comp_div_bl(self, packet)
                self.FSMmaster.done_comp_div_bl(self)

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
                    #MLmodel.Xtr_b = MLmodel.add_bias(MLmodel.Xtr_b).astype(float)
                    #MLmodel.Xtr_b = MLmodel.Xtr_b.astype(float)
                    #MLmodel.ytr = (MLmodel.ytr * 2.0 - 1.0).astype(float)
                    #MLmodel.which1 = (MLmodel.ytr == 1).ravel()
                    #MLmodel.which_1 = (MLmodel.ytr == -1).ravel()

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
                    #MLmodel.display('ERROR AT while_computing_KTDaK')
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
                    print(MLmodel.KXC.shape)
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

            def while_compute_states(self, MLmodel, packet):
                try:
                    check = False
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    MLmodel.display(MLmodel.name + ' %s: computing states...' % (str(MLmodel.worker_address)))
                    w_encr_dict = packet['data']['w_encr_dict']
                    MLmodel.classes = packet['data']['classes']

                    del packet
                    #X = MLmodel.KXC
                    #MLmodel.display('PROC_WORKER_START', verbose=False)
                    #y = MLmodel.ytr.reshape(-1, 1)
                    #MLmodel.s1_encr = np.dot(MLmodel.KXC[MLmodel.which1, :], w_encr)
                    #MLmodel.s_1_encr = np.dot(MLmodel.KXC[MLmodel.which_1, :], w_encr)
                    MLmodel.s1_encr_dict = {}
                    MLmodel.s_1_encr_dict = {}

                    for cla in MLmodel.classes:
                        ytr = ((np.array(MLmodel.ytr == cla)).astype(float)).reshape((-1, 1))
                        which1 = (ytr == 1).ravel()
                        which_1 = (ytr == 0).ravel()
                        MLmodel.s1_encr_dict.update({cla: np.dot(MLmodel.KXC[which1, :], w_encr_dict[cla])  })
                        MLmodel.s_1_encr_dict.update({cla: np.dot(MLmodel.KXC[which_1, :], w_encr_dict[cla])  })

                    if check:
                        MLmodel.w = MLmodel.decrypter.decrypt(w_encr)
                        MLmodel.s = MLmodel.decrypter.decrypt(MLmodel.s_encr)
                        MLmodel.s_ok = np.dot(MLmodel.Xtr_b, MLmodel.w)
                        e = np.linalg.norm(MLmodel.s_ok - MLmodel.s)
                        print('Error in s', e)

                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_ai'
                    data = {'s1_encr_dict': MLmodel.s1_encr_dict, 's_1_encr_dict': MLmodel.s_1_encr_dict}
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

            def while_compute_KTDay(self, MLmodel, packet):
                try:
                    check = False
                    MLmodel.display(MLmodel.name + ' %s: computing KTDay...' % (str(MLmodel.worker_address)))
                    MLmodel.display('PROC_WORKER_START', verbose=False)

                    a1_encr_dict = packet['data']['a1_encr_dict']
                    a_1_encr_dict = packet['data']['a_1_encr_dict']                   
                    del packet

                    KTDaK_dict = {}
                    KTDay_dict = {}
                    for cla in MLmodel.classes:
                        ytr = ((np.array(MLmodel.ytr == cla)).astype(float)).reshape((-1, 1))
                        which1 = (ytr == 1).ravel()
                        which_1 = (ytr == 0).ravel()

                        KTDaK1 = MLmodel.KXC[which1, :]  * a1_encr_dict[cla]
                        KTDaK1 = np.dot(MLmodel.KXC[which1, :].T, KTDaK1)
                        KTDay1 = ytr[which1].reshape((-1, 1))  * a1_encr_dict[cla]
                        KTDay1 = np.dot(MLmodel.KXC[which1, :].T, KTDay1)

                        KTDaK_1 = MLmodel.KXC[which_1, :]  * a_1_encr_dict[cla]
                        KTDaK_1 = np.dot(MLmodel.KXC[which_1, :].T, KTDaK_1)
                        KTDay_1 = (ytr[which_1] - 1).reshape((-1, 1)) * a_1_encr_dict[cla]
                        KTDay_1 = np.dot(MLmodel.KXC[which_1, :].T, KTDay_1)

                        KTDaK_dict.update({cla: KTDaK1 + KTDaK_1})
                        del KTDaK1, KTDaK_1
                        KTDay_dict.update({cla: KTDay1 + KTDay_1})
                        del KTDay1, KTDay_1

                    del a1_encr_dict, a_1_encr_dict
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_KTDaK'
                    data = {'KTDaK_dict': KTDaK_dict, 'KTDay_dict': KTDay_dict}
                    del KTDaK_dict, KTDay_dict
                    packet = {'action': action, 'data': data, 'to': 'MLmodel', 'sender': MLmodel.worker_address}
                    del data

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
                    print('ERROR AT while_compute_KTDay')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_compute_exp(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    check = False
                    exp_s_bl_encr = packet['data']['exp_s_bl_encr']
                    NP = exp_s_bl_encr.shape[0]
                    del packet
                    #MLmodel.display('PROC_WORKER_START', verbose=False)

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
                    #MLmodel.display('PROC_WORKER_END', verbose=False)
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ask_div_bl'
                    message_id = 'none'
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
                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    check = False
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
                    grad_encr = np.sum(Xe_encr, axis=0).reshape((NI, 1))
                    #MLmodel.display('PROC_WORKER_END', verbose=False)
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
            State(name='storing_Pk', on_enter=['while_storing_Pk']),
            State(name='projecting_C', on_enter=['while_projecting_C']),
            State(name='storing_C', on_enter=['while_storing_C']),

            State(name='compute_states', on_enter=['while_compute_states']),
            State(name='compute_KTDay', on_enter=['while_compute_KTDay']),

            State(name='compute_exp', on_enter=['while_compute_exp']),
            State(name='compute_gradients', on_enter=['while_compute_gradients'])
             ]

        transitions_worker = [
            ['go_setting_tr_data', 'waiting_order', 'setting_tr_data'],
            ['done_setting_tr_data', 'setting_tr_data', 'waiting_order'],

            ['go_storing_Pk', 'waiting_order', 'storing_Pk'],
            ['done_storing_Pk', 'storing_Pk', 'waiting_order'],

            ['go_projecting_C', 'waiting_order', 'projecting_C'],
            ['done_projecting_C', 'projecting_C', 'waiting_order'],

            ['go_storing_C', 'waiting_order', 'storing_C'],
            ['done_storing_C', 'storing_C', 'waiting_order'],

            ['go_compute_states', 'waiting_order', 'compute_states'],
            ['go_compute_KTDay', 'compute_states', 'compute_KTDay'],
            ['done_compute_KTDay', 'compute_KTDay', 'waiting_order']

            #            ['go_compute_exp', 'compute_states', 'compute_exp'],
            #            ['go_compute_gradients', 'compute_exp', 'compute_gradients'],
            #            ['done_compute_gradients', 'compute_gradients', 'waiting_order'],

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

            if packet['action'] == 'selecting_C':
                #self.C = packet['data']['C']
                self.FSMworker.go_projecting_C(self, packet)
                self.FSMworker.done_projecting_C(self)

            if packet['action'] == 'sending_C':
                #self.C = packet['data']['C']
                self.FSMworker.go_storing_C(self, packet)
                self.FSMworker.done_storing_C(self)

            if packet['action'] == 'send_w_encr':
                self.FSMworker.go_compute_states(self, packet)

            if packet['action'] == 'ACK_sent_ai_encr':
                self.FSMworker.go_compute_KTDay(self, packet)
                self.FSMworker.done_compute_KTDay(self)

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
