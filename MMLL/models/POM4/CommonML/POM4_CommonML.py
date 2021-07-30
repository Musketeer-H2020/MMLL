# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM4

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Febr. 2021"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
import pickle
from transitions import State
from transitions.extensions import GraphMachine
import time
import math
from collections import Counter 
from tqdm import tqdm   # pip install tqdm
import pickle
from pympler import asizeof #asizeof.asizeof(my_object)
import dill
import time

class POM4_CommonML_Master(Common_to_all_POMs):
    """
    This class implements ML operations common to POM5 algorithms, run at Master node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.
    """

    def __init__(self, master_address, workers_addresses, comms, logger, verbose=False, **kwargs):
        """
        Create a :class:`POM5_CommonML_Master` instance.

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
        self.name = 'POM4_CommonML_Master'               # Name
        self.pom = 4
        self.master_address = master_address
        self.cryptonode_address = None
        
        # Convert workers_addresses -> '0', '1', + send_to dict
        self.broadcast_addresses = workers_addresses
        self.Nworkers = len(workers_addresses)                    # Nworkers
        self.workers_addresses = list(range(self.Nworkers))
        self.workers_addresses = [str(x) for x in self.workers_addresses]
        self.receivers_list = None

        self.send_to = {}
        self.receive_from = {}
        for k in range(self.Nworkers):
            self.send_to.update({str(k): workers_addresses[k]})
            self.receive_from.update({workers_addresses[k]: str(k)})

        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.verbose = verbose                      # print on screen when true

        self.process_kwargs(kwargs)

        self.state_dict = None                      # State of the main script
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.NI_dict = {}
        self.X_encr_dict = {}
        self.y_encr_dict = {}

        for addr in self.workers_addresses:
            self.state_dict.update({addr: ''})
        self.create_FSM_master()
        self.message_counter = 0    # used to number the messages
        self.worker_names = {} # dictionary with the mappings worker_id -> pseudo_id
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
        self.display(self.name + ': creating Common ML FSM, POM4')

        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='asking_encrypter', on_enter=['while_asking_encrypter']),
            State(name='asking_encr_data', on_enter=['while_asking_encr_data']),
            State(name='sending_bl_data', on_enter=['while_sending_bl_data']),

            State(name='sending_prep_object', on_enter=['while_sending_prep_object']),
            State(name='asking_local_prep', on_enter=['while_asking_local_prep']),
            #State(name='sending_roundrobin', on_enter=['while_sending_roundrobin']),
            State(name='getting_Npc', on_enter=['while_getting_Npc']),
            State(name='getting_sumX', on_enter=['while_getting_sumX']),
            State(name='getting_sumX_roundrobin', on_enter=['while_getting_sumX_roundrobin']),
            State(name='getting_X_minus_mean_squared', on_enter=['while_getting_X_minus_mean_squared']),
            State(name='getting_X_minus_mean_squared_roundrobin', on_enter=['while_getting_X_minus_mean_squared_roundrobin']),

            State(name='getting_minX', on_enter=['while_getting_minX']),

            State(name='getting_sumXy', on_enter=['while_getting_sumXy']),
            State(name='getting_stats', on_enter=['while_getting_stats']),

            State(name='getting_Rxyb_rxyb_direct', on_enter=['while_getting_Rxyb_rxyb_direct']),
            State(name='getting_Rxyb_rxyb_roundrobin', on_enter=['while_getting_Rxyb_rxyb_roundrobin']),

            State(name='getting_vocab_direct', on_enter=['while_getting_vocab_direct']),
            State(name='getting_tf_df_direct', on_enter=['while_getting_tf_df_direct']),

            State(name='getting_feat_count_direct', on_enter=['while_getting_feat_count_direct']),

            State(name='getting_hashids_direct', on_enter=['while_getting_hashids_direct']),

            State(name='record_linkage', on_enter=['while_record_linkage']),

            State(name='sending_check_data', on_enter=['while_sending_check_data']),

            State(name='asking_data2num_at_workers_V', on_enter=['while_asking_data2num_at_workers_V']),

            State(name='sending_prep_object_V', on_enter=['while_sending_prep_object_V']),

            State(name='sending_ping', on_enter=['while_sending_ping']),

            State(name='terminating_workers', on_enter=['while_terminating_workers']),
            State(name='Exit', on_enter=['while_Exit'])
        ]

        transitions_master = [
            ['go_asking_encrypter', 'waiting_order', 'asking_encrypter'],
            ['go_waiting_order', 'asking_encrypter', 'waiting_order'],

            ['go_asking_encr_data', 'waiting_order', 'asking_encr_data'],
            ['go_waiting_order', 'asking_encr_data', 'waiting_order'],

            ['go_sending_bl_data', 'waiting_order', 'sending_bl_data'],
            ['go_waiting_order', 'sending_bl_data', 'waiting_order'],

            ['go_sending_prep_object', 'waiting_order', 'sending_prep_object'],
            #['go_asking_local_prep', 'sending_prep_object', 'asking_local_prep'],
            ['go_waiting_order', 'sending_prep_object', 'waiting_order'],

            ['go_getting_Npc', 'waiting_order', 'getting_Npc'],
            ['go_waiting_order', 'getting_Npc', 'waiting_order'],

            ['go_getting_sumXy', 'waiting_order', 'getting_sumXy'],
            ['go_waiting_order', 'getting_sumXy', 'waiting_order'],

            ['go_getting_sumX', 'waiting_order', 'getting_sumX'],
            ['go_waiting_order', 'getting_sumX', 'waiting_order'],

            ['go_getting_minX', 'waiting_order', 'getting_minX'],
            ['go_waiting_order', 'getting_minX', 'waiting_order'],

            ['go_getting_sumX_roundrobin', 'waiting_order', 'getting_sumX_roundrobin'],
            ['go_waiting_order', 'getting_sumX_roundrobin', 'waiting_order'],

            ['go_getting_X_minus_mean_squared', 'waiting_order', 'getting_X_minus_mean_squared'],
            ['go_waiting_order', 'getting_X_minus_mean_squared', 'waiting_order'],

            ['go_getting_X_minus_mean_squared_roundrobin', 'waiting_order', 'getting_X_minus_mean_squared_roundrobin'],
            ['go_waiting_order', 'getting_X_minus_mean_squared_roundrobin', 'waiting_order'],

            ['go_getting_stats', 'waiting_order', 'getting_stats'],
            ['go_waiting_order', 'getting_stats', 'waiting_order'],

            ['go_getting_Rxyb_rxyb_direct', 'waiting_order', 'getting_Rxyb_rxyb_direct'],
            ['go_waiting_order', 'getting_Rxyb_rxyb_direct', 'waiting_order'],

            ['go_getting_Rxyb_rxyb_roundrobin', 'waiting_order', 'getting_Rxyb_rxyb_roundrobin'],
            ['go_waiting_order', 'getting_Rxyb_rxyb_roundrobin', 'waiting_order'],

            ['go_getting_vocab_direct', 'waiting_order', 'getting_vocab_direct'],
            ['go_waiting_order', 'getting_vocab_direct', 'waiting_order'],

            ['go_getting_tf_df_direct', 'waiting_order', 'getting_tf_df_direct'],
            ['go_waiting_order', 'getting_tf_df_direct', 'waiting_order'],

            ['go_sending_check_data', 'waiting_order', 'sending_check_data'],
            ['go_waiting_order', 'sending_check_data', 'waiting_order'],

            ['go_getting_feat_count_direct', 'waiting_order', 'getting_feat_count_direct'],
            ['go_waiting_order', 'getting_feat_count_direct', 'waiting_order'],

            ['go_getting_hashids_direct', 'waiting_order', 'getting_hashids_direct'],
            ['go_waiting_order', 'getting_hashids_direct', 'waiting_order'],

            ['go_record_linkage', 'waiting_order', 'record_linkage'],
            ['go_waiting_order', 'record_linkage', 'waiting_order'],

            ['go_asking_data2num_at_workers_V', 'waiting_order', 'asking_data2num_at_workers_V'],
            ['go_waiting_order', 'asking_data2num_at_workers_V', 'waiting_order'],

            ['go_sending_prep_object_V', 'waiting_order', 'sending_prep_object_V'],
            ['go_waiting_order', 'sending_prep_object_V', 'waiting_order'],

            ['go_sending_ping', 'waiting_order', 'sending_ping'],
            ['go_waiting_order', 'sending_ping', 'waiting_order'],

            ['go_terminating_workers', 'waiting_order', 'terminating_workers'],
            ['go_waiting_order', 'terminating_workers', 'waiting_order'],

            ['go_Exit', 'waiting_order', 'Exit'],
            ['go_waiting_order', 'Exit', 'waiting_order']
        ]

        class FSM_master(object):

            def while_Exit(self, MLmodel):
                #print(MLmodel.name + 'while_Exit')
                return

            def while_waiting_order(self, MLmodel):
                '''
                try:
                    MLmodel.display(MLmodel.name + ' is waiting...')
                except:
                    print('STOP AT while_waiting_order')
                    import code
                    code.interact(local=locals())
                '''
                return

            def while_asking_encrypter(self, MLmodel):
                try:
                    action = 'ask_encrypter'
                    data = {}
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}

                    destination = MLmodel.cryptonode_address
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    #MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)
                    destination = 'ca'
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)
                                        
                    #MLmodel.comms.send(packet, MLmodel.cryptonode_address)
                    # We dont know the address of the cryptonode, we boradcast.
                    #MLmodel.comms.broadcast(packet, MLmodel.receivers_list)                   
                    MLmodel.comms.send(packet, MLmodel.send_to[MLmodel.cryptonode_address])

                    MLmodel.display(MLmodel.name + ' asking encrypter to cryptonode')
                except:
                    raise
                    '''
                    print('ERROR AT while_asking_encrypter')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_asking_encr_data(self, MLmodel, use_bias, classes):

                try:
                    # Communicating encrypter to workers
                    data = {'encrypter': MLmodel.encrypter, 'use_bias': use_bias, 'classes': classes}
                    # For checking, REMOVE
                    #data.update({'decrypter': MLmodel.decrypter})
                    action = 'ask_encr_data'
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.broadcast(packet)
                    MLmodel.display(MLmodel.name + ' sent encrypter to all Workers and asked encr_data')
                except:
                    raise
                    '''
                    print('ERROR AT while_asking_encr_data')
                    import code
                    code.interact(local=locals())
                    '''
                return


            '''
            def while_bcasting_encrypter(self, MLmodel):
                try:
                    # Communicating encrypter to workers
                    data = {'encrypter': MLmodel.cr.encrypter}
                    # For checking, REMOVE
                    data.update({'decrypter': MLmodel.cr.decrypter})
                    packet = {'action': 'send_encrypter', 'to': 'CommonML', 'data': data}
                    MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                    MLmodel.display(MLmodel.name + ' sent encrypter to all Workers')
                except:
                    print('ERROR AT while_bcasting_encrypter')
                    import code
                    code.interact(local=locals())
                return
            '''

            def while_sending_bl_data(self, MLmodel, classes):
                try:                    

                    MLmodel.display('PROC_MASTER_START', verbose=False)

                    # Encrypted data at MLmodel.X_encr_dict, MLmodel.y_encr_dict
                    # To store at MLmodel
                    MLmodel.BX_dict = {}
                    MLmodel.By_dict = {}
                    # To send to crypto
                    MLmodel.X_bl_dict = {}
                    MLmodel.y_bl_dict = {}
                    
                    for waddr in MLmodel.workers_addresses:
                        X = MLmodel.X_encr_dict[waddr]
                        NP, NI = X.shape
                        BX = np.random.normal(0, 1, (NP, NI))
                        MLmodel.BX_dict.update({waddr: BX})
                        MLmodel.X_bl_dict.update({waddr: X + BX})
                        
                        if classes is None: # binary case
                            y = MLmodel.y_encr_dict[waddr].reshape((-1, 1))
                            By = np.random.normal(0, 1, (NP, 1))
                            MLmodel.By_dict.update({waddr: By})
                            MLmodel.y_bl_dict.update({waddr: y + By})
                        else: # multiclass
                            y = MLmodel.y_encr_dict[waddr] # dict cla
                            y_bl_dict = {}
                            By_dict = {}
                            for cla in classes:
                                By_dict.update({cla: np.random.normal(0, 1, (NP, 1))})
                                y_bl_dict.update({cla: y[cla] + By_dict[cla]})
                            MLmodel.By_dict.update({waddr: By_dict})
                            MLmodel.y_bl_dict.update({waddr: y_bl_dict})
                    MLmodel.display('PROC_MASTER_END', verbose=False)

                    action = 'send_Xy_bl'
                    data = {'X_bl_dict': MLmodel.X_bl_dict, 'y_bl_dict': MLmodel.y_bl_dict}
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address}
                    MLmodel.display(MLmodel.name + ' sending Xy data blinded to cryptonode...')
                    
                    destination = 'ca'
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_SEND %s to %s, id = %s, bytes=%s' % (action, destination, message_id, str(size_bytes)), verbose=False)                   

                    MLmodel.comms.send(packet,  MLmodel.send_to[MLmodel.cryptonode_address])

                except:
                    raise
                    '''
                    print('ERROR AT while_sending_bl_data')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_sending_prep_object(self, MLmodel, model):              
                action = 'do_local_prep'
                data = {'prep_object': model}
                packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.worker_errors = {}
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display(MLmodel.name + ' is sending preprocessing object of size %d Kbytes' % int(size_bytes/1024.0))
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent preprocessing object to all Workers')
                return

            def while_asking_local_prep(self, MLmodel):
                packet = {'action': 'do_local_prep', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent do_local_prep to all Workers')
                return

            def while_getting_Npc(self, MLmodel):
                packet = {'action': 'get_Npc', 'to': 'CommonML', 'classes': MLmodel.classes, 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' broadcasted get_Npc to all Workers')
                return

            def while_getting_sumXy(self, MLmodel):
                packet = {'action': 'get_sumXy', 'to': 'CommonML', 'classes': MLmodel.classes, 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' broadcasted get_sumXy to all Workers')
                return

            def while_getting_sumX(self, MLmodel, input_data_description, which_variables):
                data = {'input_data_description': input_data_description, 'which_variables': which_variables}
                packet = {'action': 'get_sumX', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' broadcasted get_sumX to all Workers')
                return

            def while_getting_minX(self, MLmodel, input_data_description):
                data = {'input_data_description': input_data_description}
                packet = {'action': 'get_minX', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' broadcasted get_minX to all Workers')
                return

            def while_getting_sumX_roundrobin(self, MLmodel, input_data_description, x_init, NP_init, which_variables):
                data = {'input_data_description': input_data_description, 'x_init': x_init, 'NP_init': NP_init, 'which_variables': which_variables}
                packet = {'action': 'get_sumX_roundrobin', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.roundrobin(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' Started roundrobin: get_sumX_roundrobin')             
                return

            def while_getting_X_minus_mean_squared(self, MLmodel, mean_values, input_data_description, which_variables):
                try:
                    data = {'mean_values': mean_values, 'input_data_description': input_data_description, 'which_variables': which_variables}
                    packet = {'action': 'get_X_minus_mean_squared', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                    MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                    MLmodel.display(MLmodel.name + ' broadcasted get_X_minus_mean_squared to all Workers')
                except:
                    raise
                    '''
                    print('ERROR AT while_getting_X_minus_mean_squared, POM6_Common')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_getting_X_minus_mean_squared_roundrobin(self, MLmodel, mean_values, input_data_description, x2_init, which_variables):
                try:
                    data = {'mean_values': mean_values, 'input_data_description': input_data_description, 'x2_init': x2_init, 'which_variables': which_variables}
                    packet = {'action': 'get_X_minus_mean_squared_roundrobin', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                    MLmodel.comms.roundrobin(packet, MLmodel.workers_addresses)
                    MLmodel.display(MLmodel.name + 'Started roundrobin: get_X_minus_mean_squared_roundrobin')
                except:
                    raise
                    '''
                    print('ERROR AT while_getting_X_minus_mean_squared_roundrobin, POM6_Common')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return
            '''
            def while_sending_roundrobin(self, MLmodel, roundrobin_addresses, action, NI=None, xmean=None):
                if action == 'count_patterns':
                    self.N_random_ini = np.random.randint(1000000)
                    destiny_adress = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': action, 'N': self.N_random_ini, 'remain_addresses': remain_addresses, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(destiny_adress, packet)

                if action == 'sum_patterns':
                    self.X_sum_ini = np.random.randint(0, 10000, (NI, 1)).ravel()
                    destiny_adress = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': action, 'X_sum': self.X_sum_ini, 'remain_addresses': remain_addresses, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(destiny_adress, packet)

                if action == 'squared_sum_patterns':
                    self.Xsquared_sum_ini = np.random.randint(0, 10000, (NI, 1)).ravel()
                    destiny_adress = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': action, 'to': 'CommonML', 'Xsquared_sum': self.Xsquared_sum_ini, 'remain_addresses': remain_addresses, 'xmean': xmean, 'sender': MLmodel.master_address}
                    MLmodel.comms.send(destiny_adress, packet)
                return
            '''
            def while_getting_stats(self, MLmodel, stats_list):
                data = {'stats_list': stats_list}
                packet = {'action': 'get_stats', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' broadcasted get_stats to all Workers')
                return

            def while_getting_Rxyb_rxyb_direct(self, MLmodel):   
                action = 'get_Rxyb_rxyb_direct'
                packet = {'action': action, 'to': 'CommonML', 'sender': MLmodel.master_address}
                
                message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                packet.update({'message_id': message_id})
                MLmodel.message_counter += 1
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent get_Rxyb_rxyb_direct to all Workers')
                return

            def while_getting_Rxyb_rxyb_roundrobin(self, MLmodel, Rxyb_aggr, rxyb_aggr):   
                data = {'Rxyb_aggr': Rxyb_aggr, 'rxyb_aggr': rxyb_aggr}
                packet = {'action': 'get_Rxyb_rxyb_roundrobin', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.roundrobin(packet, MLmodel.workers_addresses)
                MLmodel.display(MLmodel.name + ' sent get_Rxyb_rxyb_roundrobin')
                return

            def while_getting_vocab_direct(self, MLmodel):   
                packet = {'action': 'get_vocab_direct', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent get_vocab_direct to all Workers')
                return

            def while_getting_tf_df_direct(self, MLmodel, vocab):
                data = {'vocab': vocab}   
                packet = {'action': 'get_tf_df_direct', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent get_tf_df_direct to all Workers')
                return

            def while_sending_check_data(self, MLmodel, input_data_description, target_data_description):              
                action = 'check_data'
                data = {'input_data_description': input_data_description, 'target_data_description': target_data_description}
                packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}
                #MLmodel.worker_errors = {}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent check_data to all Workers')
                return

            def while_getting_feat_count_direct(self, MLmodel):   
                packet = {'action': 'get_feat_count_direct', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent get_feat_count_direct to all Workers')
                return

            def while_getting_hashids_direct(self, MLmodel):   
                packet = {'action': 'get_hashids_direct', 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent get_hashids_direct to all Workers')
                return

            def while_record_linkage(self, MLmodel, hashids, linkage_type):   
                data = {'hashids': hashids, 'linkage_type': linkage_type}
                packet = {'action': 'record_linkage', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent record_linkage to all Workers')
                return

            def while_asking_data2num_at_workers_V(self, MLmodel):
                data = {}
                packet = {'action': 'data2num_at_worker_V', 'to': 'CommonML', 'sender': MLmodel.master_address, 'data': data}
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent data2num_at_worker_V to all Workers')

                return

            def while_sending_prep_object_V(self, MLmodel, model):              
                action = 'do_local_prep_V'
                data = {'prep_object_V': model}
                packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}
                MLmodel.worker_errors = {}
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display(MLmodel.name + ' is sending preprocessing object of size %d Kbytes' % int(size_bytes/1024.0))
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent preprocessing object to all Workers')
                return

            def while_sending_ping(self, MLmodel):              
                action = 'ping'
                data = None
                packet = {'action': action, 'data': data, 'to': 'CommonML', 'sender': MLmodel.master_address}
                
                '''
                message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                packet.update({'message_id': message_id})
                MLmodel.message_counter += 1
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)
                '''
                message_id = 'empty'
                packet.update({'message_id': message_id})
                
                MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                MLmodel.display(MLmodel.name + ' sent ping to all Workers')
                return

            def while_Exit(self, MLmodel):
                try:
                    print(MLmodel.name + ' COMPLETED')
                except:
                    raise
                    '''
                    print('STOP AT while_Exit')
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
            title="Finite State Machine modelling CommonML at master",
            show_conditions=False)
        return

    '''
    def terminate_Workers(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        users_addresses_terminate: list of strings
            addresses of the workers to be terminated

        """
        message_id = self.master_address + str(self.message_counter)
        packet = {'action': 'STOP', 'to': 'CommonML', 'sender': self.master_address, 'message_id': message_id}
        if workers_addresses_terminate is None:  # We terminate all of them
            workers_addresses_terminate = self.workers_addresses
            self.display('CommonML_Master sent STOP to all Workers')
        else:
            self.display('CommonML_Master sent STOP to %d Workers' % len(workers_addresses_terminate))

        self.comms.broadcast(packet, workers_addresses_terminate)

        # Updating the list of active users
        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)
        #self.FSMmaster.go_Exit(self)
        #self.FSMmaster.go_waiting_order(self)
    '''

    def get_vocabulary(self):
        """
        Gets the workers vocabulary

        Parameters
        ----------
        None

        Returns
        -------
        vocab: list of strings
            Vocabulary.

        """
        if self.aggregation_type == 'direct':
            self.vocab_dict = {}
            self.display(self.name + ': Asking workers their vocabulary directly')
            self.FSMmaster.go_getting_vocab_direct(self)
            self.run_Master()

            self.global_tf_dict = {}
            for waddr in self.workers_addresses:
                self.global_tf_dict = dict(Counter(self.global_tf_dict) + Counter(self.vocab_dict[waddr]))
            self.display(self.name + ': length of tf dictionary: %d' % len(self.global_tf_dict))

            self.global_tf_dict_filtered = {}
            for key in self.global_tf_dict.keys(): # at least 10 times, and less than 5000
                if self.global_tf_dict[key] > 10 and self.global_tf_dict[key] < 5000:
                    self.global_tf_dict_filtered.update({key: self.global_tf_dict[key]})

            self.display(self.name + ': length of filtered tf dictionary: %d' % len(self.global_tf_dict_filtered))

            vocab = list(self.global_tf_dict_filtered.keys())
            vocab.sort()
            self.vocab = vocab
        '''
        if self.aggregation_type == 'roundrobin':
            self.display(self.name + ': Getting sumX with roundrobin')
            # pending generate random numbers here...
            self.x_ini = np.random.uniform(-9e5, 9e5, (1, self.NI))
            self.NP_ini = np.random.uniform(-9e5, 9e5)
            self.FSMmaster.go_getting_sumX_roundrobin(self, input_data_description, self.x_ini, self.NP_ini)
            self.run_Master()
            self.total_sumX = self.sumX_roundrobin - self.x_ini
            self.total_NP =  self.NP_roundrobin - self.NP_ini   
        '''
        
        self.display(self.name + ': getting vocab is done')

    def get_df(self, vocab):
        """
        Gets df and filters vocabulary

        Parameters
        ----------
        vocab: list of strings
            Vocabulary to be used
        """
        if self.aggregation_type == 'direct':
            self.display(self.name + ': Asking workers their tf df directly')
            self.Ndocs_dict = {}
            self.df_dict = {}
            self.FSMmaster.go_getting_tf_df_direct(self, vocab)
            self.run_Master()


            self.global_df_dict = {}
            for waddr in self.workers_addresses:
                self.global_df_dict = dict(Counter(self.global_df_dict) + Counter(self.df_dict[waddr]))

            self.Ndocs = 0
            for waddr in self.workers_addresses:
                self.Ndocs += self.Ndocs_dict[waddr]

            self.display(self.name + ': length of df dictionary: %d' % len(self.global_df_dict))

            self.global_df_dict_filtered = {}
            for key in self.global_df_dict.keys():
                if self.global_df_dict[key] > 10: # at least in 10 docs
                    self.global_df_dict_filtered.update({key: self.global_df_dict[key]})

            self.display(self.name + ': length of filtered df dictionary: %d' % len(self.global_df_dict_filtered))

            vocab = list(self.global_df_dict_filtered.keys())
            vocab.sort()
            self.vocab = vocab
        '''
        if self.aggregation_type == 'roundrobin':
            self.display(self.name + ': Getting sumX with roundrobin')
            # pending generate random numbers here...
            self.x_ini = np.random.uniform(-9e5, 9e5, (1, self.NI))
            self.NP_ini = np.random.uniform(-9e5, 9e5)
            self.FSMmaster.go_getting_sumX_roundrobin(self, input_data_description, self.x_ini, self.NP_ini)
            self.run_Master()
            self.total_sumX = self.sumX_roundrobin - self.x_ini
            self.total_NP =  self.NP_roundrobin - self.NP_ini   
        '''
        self.display(self.name + ': getting df is done')


    def get_feat_count(self):
        """
        Gets feature frequency from workers

        Parameters
        ----------
        None

        Returns:
        ----------
        count: array
            Features count.
        NP: integer
            Total count.
        """
        if self.aggregation_type == 'direct':
            self.count_dict = {}
            self.NP_dict = {}
            self.display(self.name + ': Asking workers their features count directly')

            self.FSMmaster.go_getting_feat_count_direct(self)
            self.run_Master()

            NP = np.sum(np.array(list(self.NP_dict.values())))
            count = np.sum(np.array(list(self.count_dict.values())), axis=0)
            
        if self.aggregation_type == 'roundrobin':
            print('ERROR AT get_feat_count, pending roundrobin')
            #import code
            #code.interact(local=locals())
        
        self.display(self.name + ': getting feat freq is done')
        return count, NP

    def get_hashids(self, linkage_type):
        """
        Gets hashids from workers

        Parameters
        ----------
        linkage_type: string
            Type of linkage (full or join).

        Returns
        -------
        hashids_global: list of hashids
            Global hashids.
        """
        hashids_global = None

        if self.aggregation_type == 'direct':
            self.hashids_dict = {}
            self.display(self.name + ': Asking workers their hashids directly')
            self.FSMmaster.go_getting_hashids_direct(self)
            self.run_Master()

            unique_hashids = []
            for waddr in self.workers_addresses:
                unique_hashids = list(set(unique_hashids + self.hashids_dict[waddr]))

            if linkage_type == 'full':
                # Option 1: Filtering out hashids that are not in all workers
                Nhashids = len(unique_hashids)
                hashids_count = {}
                for khash in range(Nhashids):
                    hashids_count.update({unique_hashids[khash]: 0})

                for waddr in self.workers_addresses:
                    hashids = self.hashids_dict[waddr]
                    for hashid in hashids:
                        hashids_count[hashid] += 1

                Nworkers = len(self.workers_addresses)
                hashids_global = []
                for hashid in hashids_count:
                    if hashids_count[hashid] == Nworkers:
                        hashids_global.append(hashid)

            if linkage_type == 'join':
                # Option 2: take all ids into account, some features may be missing
                Nhashids = len(unique_hashids)
                hashids_global = unique_hashids

        if self.aggregation_type == 'roundrobin':
            print('get_hashids: Pending roundrobin implementation ')
            #import code
            #code.interact(local=locals())

        self.display(self.name + ': getting hashids is done')
        return hashids_global


    def linkage_hashids_transform_workers(self, hashids, linkage_type):
        """
        Record linkage at workers

        Parameters
        ----------
        hashids: list of strings
            Global hashids
        linkage_type: string
            Type of linkage (full or join).

        Returns
        -------
        input_data_description_dict: dict
            Updated description of the input data.

        target_data_description_dict: dict
            Updated description of the target data.
        """

        self.input_data_description_dict = {}
        self.target_data_description_dict = {}
        self.display(self.name + ': Asking workers record-linkage directly')
        self.FSMmaster.go_record_linkage(self, hashids, linkage_type)
        self.run_Master()

        self.display(self.name + ': record linkage at workers is done')
        return self.input_data_description_dict, self.target_data_description_dict

    def data2num_at_workers_V(self):
        """
        Asks workers to transform their data into numeric, vertical partition

        Parameters
        ----------
        None

        Returns
        -------
        input_data_description_dict: dict
            Updated description of the input data.

        target_data_description_dict: dict
            Updated description of the target data.
        """

        # New values
        self.input_data_description_dict = {}
        self.target_data_description_dict = {}
        self.errors_dict = {}

        self.display(self.name + ': Asking workers to transform data to numeric')
        self.FSMmaster.go_asking_data2num_at_workers_V(self)
        self.run_Master()

        self.display(self.name + ': data2num at workers is done')
        return self.input_data_description_dict, self.target_data_description_dict, self.errors_dict

    def send_ping_workers(self):
        """
        Ping workers to identify the cryptonode

        Parameters
        ----------
            None
        """
        self.display(self.name + ' : Sending ping to workers')
        self.FSMmaster.go_sending_ping(self)
        self.run_Master()
        self.display(self.name + ' : ping is done')


    def ask_encrypter(self):
        """
        Obtain encrypter from crypto, under POM4. The crypto also shares with the workers.

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_asking_encrypter(self)
        # We wait for the answer from the cryptonode
        self.run_Master()

    def get_cryptdata(self, use_bias=True, classes=None):
        """, classes=None
        Get encrypted data from workers, under POM4

        Parameters
        ----------
        use_bias: bool
            Indicates if bias must be used
        classes: list of strings
            list of classes
        """

        # Le mandamos al estado para activar el run_Master, pero no hace nada.
        self.FSMmaster.go_asking_encr_data(self, use_bias, classes)
        print('---------- waiting for cryptdata')
        self.run_Master()

        # Add blinding and share with cryptonode
        self.FSMmaster.go_sending_bl_data(self, classes)

        # We need to wait for the cryptonode to store the data
        crypto_OK = False
        print('Waiting for the cryptonode...')
        while not crypto_OK:
            time.sleep(1)
            packet, sender = self.CheckNewPacket_master()
            if packet is not None:
                if packet['action'] == 'ACK_storing_Xy_bl':
                    crypto_OK = True
        print('Cryptonode is ready!')

        self.FSMmaster.go_waiting_order(self)


    def reset(self, NI):
        """
        Create some empty variables needed by the Master Node

        Parameters
        ----------
        NI: integer
            Number of input features
        """

        #self.w_decr = np.random.normal(0, 0.001, (self.NI + 1, 1))      # weights in plaintext, first value is bias
        self.NI = NI
        self.w_decr = None      # weights in plaintext, first value is bias
        self.R_central_decr = np.zeros((self.NI + 1, self.NI + 1))    # Cov. matrix in plaintext
        self.r_central_decr = np.zeros((self.NI + 1, 1))              # Cov. matrix in plaintext
        self.preds_dict = {}                                           # dictionary storing the prediction errors
        self.AUCs_dict = {}                                           # dictionary storing the prediction errors
        self.R_dict = {}
        self.r_dict = {}
        self.Pk_dict = {}
        self.Data_dict = {}
        self.display('CommonML_Master: Resetting local data')

    def local_prep_Master(self, prep_object):
        """
        This is the local preprocessing loop, it runs the following actions until 
        the stop condition is met:
            - Update the execution state
            - Process the received packets
            - Perform actions according to the state

        Parameters
        ----------
            prep_object: object instance
                Instance of the preprocessing object
        """
        self.prep = prep_object
        self.FSMmaster.go_sending_prep_object(self)
        self.display('CommonML_Master: Sending Preprocessing object')
        self.run_Master()
        self.display('CommonML_Master: Local Preprocessing is done')

    def Update_State_Master(self):
        """
        We update control the flow given some conditions and parameters

        Parameters
        ----------
            None
        """
        if self.chekAllStates('ACK_stored_prep'):
            self.FSMmaster.go_asking_local_prep(self)

        if self.chekAllStates('ACK_local_prep'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_encr_data'):
           self.FSMmaster.go_waiting_order(self)

        #if self.chekAllStates('ACK_sent_encrypter'):
        #    self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_Npc'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_sumXy'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_sumX'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_minX'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('get_sumX_roundrobin'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('get_X_minus_mean_squared_roundrobin'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_X2_mean_sum'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_stats'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_Rxyb_rxyb_direct'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('get_Rxyb_rxyb_roundrobin'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_vocab_direct'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_tf_df_direct'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_checking_data'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_feat_count_direct'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_hashids_direct'):
            self.FSMmaster.go_waiting_order(self)
            
        if self.chekAllStates('ACK_record_linkage'):
            self.FSMmaster.go_waiting_order(self)          

        if self.chekAllStates('ACK_data2num_at_worker_V'):
            self.FSMmaster.go_waiting_order(self)          

        if self.chekAllStates('ACK_local_prep_V'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_send_ping'):
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
            self.display(self.name + ' received %s from %s' % (packet['action'], str(sender)))
            #self.state_dict[sender] = packet['action']

            if packet['action'][0:3] == 'ACK':
                #self.display('Master received ACK from %s: %s at common' % (str(sender), packet['action']))
                if sender != self.cryptonode_address:  # The cryptonode is to be excluded from the broadcast
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


            if packet['action'] == 'ACK_sent_encrypter':
                #print('#### WARNING delete decrypter, CommonML #####')
                #self.decrypter = packet['data']['decrypter']
                self.display('Storing encrypter')
                self.encrypter = packet['data']['encrypter']
                '''
                try:
                    self.decrypter = packet['data']['decrypter']
                    print('WARNING storing DECRYPTER ONLY FOR DEBUGGING')
                except:
                    pass
                '''
                # Not needed, processed at ping
                #self.broadcast_addresses = list(set(self.workers_addresses) -set([sender]))
                
                '''
                # Not needed now, the ping function has already identified the cryptonode
                self.workers_addresses = list(set(self.workers_addresses) -  set([sender]))
                self.display('Identified workers: ' + str(self.workers_addresses))
                self.cryptonode_address = sender
                self.display('Identified cryptonode as worker %s: address %s'% (str(self.cryptonode_address), str(self.send_to[sender])))
                #we update self.state_dict with the new list of workers_addresses
                self.state_dict = {}
                for waddr in self.workers_addresses:
                    self.state_dict.update({waddr: ''})
                '''

                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'ACK_send_encr_data':                  
                self.X_encr_dict.update({sender: packet['data']['Xtr_b_encr']})
                #self.y_encr_dict.update({sender: packet['data']['ytr_encr'].reshape((-1, 1))})
                # In the multiclass case, y_encr_dict is a dict, keys are the classes
                self.y_encr_dict.update({sender: packet['data']['ytr_encr']})

            if packet['action'] == 'ACK_local_prep':
                if packet['error'] is not None:
                    self.worker_errors.update({sender: packet['error']})

            if packet['action'] == 'count_patterns':
                self.Ncount = packet['N']
                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'sum_patterns':
                self.X_sum = packet['X_sum']
                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'squared_sum_patterns':
                self.Xsquared_sum = packet['Xsquared_sum']
                self.FSMmaster.go_waiting_order(self)

            if packet['action'] == 'ACK_computing_AUCs':
                self.AUCs_dict.update({sender: packet['data']['aucs'][0]})

            if packet['action'] == 'ACK_send_Npc':
                self.Npc_dict.update({sender: packet['npc_dict']})

            if packet['action'] == 'ACK_send_sumXy':
                self.sumX_dict.update({sender: packet['data']['X_sum']})
                self.sumy_dict.update({sender: packet['data']['y_sum']})
                self.NP_dict.update({sender: packet['data']['NP']})

            if packet['action'] == 'ACK_send_sumX':
                self.sumX_dict.update({sender: packet['data']['X_sum']})
                self.NP_dict.update({sender: packet['data']['NP']})

            if packet['action'] == 'ACK_send_minX':
                self.minX_dict.update({sender: packet['data']['X_min']})
                self.maxX_dict.update({sender: packet['data']['X_max']})
                self.NP_dict.update({sender: packet['data']['NP']})

            if packet['action'] == 'get_sumX_roundrobin':
                # mark all workers as received
                for kworker in self.workers_addresses:
                    self.state_dict[kworker] = packet['action']
                self.sumX_roundrobin = packet['data']['x_init']
                self.NP_roundrobin = packet['data']['NP_init']               

            if packet['action'] == 'get_X_minus_mean_squared_roundrobin':
                # mark all workers as received
                for kworker in self.workers_addresses:
                    self.state_dict[kworker] = packet['action']
                self.sumX2_roundrobin = packet['data']['x2_init']

            if packet['action'] == 'ACK_send_X2_mean_sum':
                self.sumXminusmeansquared_dict.update({sender: packet['data']['X_mean_squared_sum']})

            if packet['action'] == 'ACK_send_stats':
                self.stats_dict.update({sender: packet['data']['stats_dict']})

            if packet['action'] == 'ACK_send_Rxyb_rxyb_direct':
                self.Rxyb_dict.update({sender: packet['data']['Rxyb']})
                self.rxyb_dict.update({sender: packet['data']['rxyb']})

            if packet['action'] == 'get_Rxyb_rxyb_roundrobin':
                # mark all workers as received
                for kworker in self.workers_addresses:
                    self.state_dict[kworker] = packet['action']
                self.Rxyb_roundrobin = packet['data']['Rxyb_aggr']
                self.rxyb_roundrobin = packet['data']['rxyb_aggr']

            if packet['action'] == 'ACK_send_vocab_direct':
                self.vocab_dict.update({sender: packet['data']['vocab_dict']})

            if packet['action'] == 'ACK_send_tf_df_direct':
                self.Ndocs_dict.update({sender: packet['data']['Ndocs']})
                self.df_dict.update({sender: packet['data']['df_dict']})

            if packet['action'] == 'ACK_checking_data':
                #worker = self.receive_from[packet['sender']]
                #print(worker, sender, packet['data']['err'])
                self.worker_errors_dict.update({sender: packet['data']['err']})

            if packet['action'] == 'ACK_send_feat_count_direct':
                self.NP_dict.update({sender: packet['data']['NP']})
                self.count_dict.update({sender: packet['data']['count']})

            if packet['action'] == 'ACK_send_hashids_direct':
                self.hashids_dict.update({sender: packet['data']['hashids']})

            if packet['action'] == 'ACK_record_linkage':
                self.input_data_description_dict.update({sender: packet['data']['input_data_description']})
                self.target_data_description_dict.update({sender: packet['data']['target_data_description']})

            if packet['action'] == 'ACK_data2num_at_worker_V':
                self.input_data_description_dict.update({sender: packet['data']['input_data_description']})
                self.target_data_description_dict.update({sender: packet['data']['target_data_description']})
                self.errors_dict.update({sender: packet['data']['error']})

            if packet['action'] == 'ACK_local_prep_V':
                if packet['error'] is not None:
                    self.worker_errors.update({sender: packet['error']})
                # try to recover mean / std
                try:
                    self.mean_dict.update({sender: packet['data']['X_mean']})
                    self.std_dict.update({sender: packet['data']['X_std']})
                except:
                    pass

            if packet['action'] == 'ACK_send_ping':
                #self.display('ACK_send_ping Master %s %s %s' % (str(sender), packet['action'], packet['data']))               
                self.ping_dict.update({sender: packet['data']})
                #self.display('ACK_send_ping Master %s %s %s' % (str(sender), packet['action'], packet['data']))               

        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Master')
            import code
            code.interact(local=locals())
            pass
            '''
        return


class POM4_CommonML_Worker(Common_to_all_POMs):
    '''
    This class implements ML operations common to POM5 algorithms, run at Worker node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None, cryptonode_address=None):
        """
        Create a :class:`POM4_CommonML_Worker` instance.

        Parameters
        ----------
        master_address: string
            address of the master node

        worker_address: string
            id of the worker

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
        self.master_address = master_address
        self.worker_address = worker_address                    # The id of this Worker
        self.cryptonode_address = cryptonode_address
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.comms = comms                      # The comms library
        self.model_type = model_type
        #self.cr = cr
        self.logger = logger                    # logger
        self.name = 'POM4_CommonML_Worker'           # Name
        self.verbose = verbose                  # print on screen when true
        self.Xtr_b = Xtr_b
        self.Xtr_orig = Xtr_b
        if model_type != 'Kmeans':
            self.ytr = ytr
            self.ytr_orig = ytr
        else:
            self.ytr = None
            self.ytr_orig = None            
        self.create_FSM_worker()
        self.message_counter = 0 # used to number the messages
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
        self.display(self.name + ' %s: creating FSM' % (str(self.worker_address)))

        class FSM_worker(object):

            # Enter/exit callbacks are defined here

            def while_waiting_order(self, MLmodel):
                try:
                    MLmodel.display(MLmodel.name + ' %s is waiting...' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('STOP AT while_waiting_order')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_storing_prep_object(self, MLmodel, packet):
                try:
                    MLmodel.prep = packet['data']['prep_object']
                    MLmodel.display(MLmodel.name + ' %s: stored preprocessing object' % (str(MLmodel.worker_address)))
                    action = 'ACK_stored_prep'
                    packet = {'action': action, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_stored_prep' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('STOP AT while_storing_prep_object')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_local_preprocessing(self, MLmodel):
                try:
                    X = np.copy(MLmodel.Xtr_b)
                    new_Xtr_b = MLmodel.prep.transform(X)
                    MLmodel.Xtr_b = np.copy(new_Xtr_b)
                    MLmodel.display(MLmodel.name + ' %s: locally preprocessing data...' % (str(MLmodel.worker_address)))
                    action = 'ACK_local_prep'
                    packet = {'action': action, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('STOP AT while_local_preprocessing')
                    import code
                    code.interact(local=locals())
                    '''
                return
            
            def while_sending_encr_data(self, MLmodel, packet):
                try:

                    MLmodel.display('PROC_WORKER_START', verbose=False)
                    MLmodel.encrypter = packet['data']['encrypter']
                    use_bias = packet['data']['use_bias']
                    MLmodel.classes = packet['data']['classes']

                    MLmodel.display(MLmodel.name + ': stored encrypter, encrypting data...')
                    
                    if use_bias:  # we add bias
                        NP = MLmodel.Xtr_b.shape[0]
                        MLmodel.Xtr_b = np.hstack((np.ones((NP, 1)), MLmodel.Xtr_b))

                    # Encrypting data
                    MLmodel.Xtr_b_encr = MLmodel.encrypter.encrypt(MLmodel.Xtr_b.astype(float))

                    try:
                        if MLmodel.classes is None: # Binary case
                            MLmodel.ytr_encr = MLmodel.encrypter.encrypt(MLmodel.ytr.astype(float))

                        if MLmodel.classes is not None: # Multiclass
                            MLmodel.ytr_encr={}
                            for cla in MLmodel.classes:
                                ytr = (np.array(MLmodel.ytr == cla).reshape(-1,1)).astype(float)
                                MLmodel.ytr_encr.update({cla: MLmodel.encrypter.encrypt(ytr)})
                    except:
                        # When no targets are provided, we pass zeros, not to be used
                        NPtr = MLmodel.Xtr_b.shape[0]
                        MLmodel.ytr_encr = MLmodel.encrypter.encrypt(np.zeros((NPtr, 1)))
                        pass
                    MLmodel.display('PROC_WORKER_END', verbose=False)

                    action = 'ACK_send_encr_data'
                    data = {'Xtr_b_encr': MLmodel.Xtr_b_encr, 'ytr_encr': MLmodel.ytr_encr}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.worker_address}
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_send_encr_data')
                except:
                    raise
                    '''
                    print('ERROR AT while_sending_encr_data')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_storing_prep_object(self, MLmodel, packet):
                #print('STOP AT while_storing_prep_object worker')
                #import code
                #code.interact(local=locals())
                MLmodel.prep = packet['data']['prep_object']
                MLmodel.display(MLmodel.name + ' %s: stored preprocessing object' % (str(MLmodel.worker_address)))
                action = 'ACK_stored_prep'
                packet = {'action': action, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(self.name + ' %s: sent ACK_stored_prep' % (str(MLmodel.worker_address)))
                return

            def while_local_preprocessing(self, MLmodel, prep_model):
                # We preprocess the original data
                X = MLmodel.Xtr_orig
                error = 'data not processed'

                try:
                    #if prep_model.name in ['data2num', 'normalization', 'logscale', 'feature_extract', 'outlier_clipping', 'deep_learning']: # we replace the original data
                    if prep_model.name == 'image_to_vector':
                        X_prep, new_input_format = prep_model.transform(X)
                    else:
                        X_prep = prep_model.transform(X)

                    MLmodel.Xtr_b = X_prep
                    MLmodel.display(MLmodel.name + ' %s: locally preprocessing data with %s...' % (str(MLmodel.worker_address), prep_model.name))
                    MLmodel.display(MLmodel.name + ' %s: new data shape is %d x %d.' % (str(MLmodel.worker_address), X_prep.shape[0], X_prep.shape[1]))
                    MLmodel.Xtr_orig = X_prep
                    error = None
                except Exception as err:
                    raise
                    '''
                    error = err
                    # Comment later
                    print('ERROR AT while_local_preprocessing worker')
                    import code
                    code.interact(local=locals())
                    pass
                    '''

                #print(np.mean(MLmodel.Xtr_b, axis = 0))
                #print(np.std(MLmodel.Xtr_b, axis = 0))

                action = 'ACK_local_prep'
                packet = {'action': action, 'sender': MLmodel.worker_address, 'error': error}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep' % (str(MLmodel.worker_address)))
                return
            '''
            def while_computing_roundrobin(self, MLmodel, packet):
                print('STOP AT while_computing_roundrobin worker')
                import code
                code.interact(local=locals())
                if packet['action'] == 'count_patterns':
                    NPtr = MLmodel.Xtr_b.shape[0]
                    N = packet['N'] + NPtr
                    roundrobin_addresses = packet['remain_addresses']
                    destiny_address = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': packet['action'], 'N': N, 'remain_addresses': remain_addresses, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, destiny_address)

                if packet['action'] == 'sum_patterns':
                    x_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()
                    X_sum = packet['X_sum'] + x_sum

                    roundrobin_addresses = packet['remain_addresses']
                    destiny_address = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': packet['action'], 'X_sum': X_sum, 'remain_addresses': remain_addresses, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, destiny_address)

                if packet['action'] == 'squared_sum_patterns':
                    xmean = packet['xmean']
                    X_mean = MLmodel.Xtr_b - xmean
                    X_squared = np.power(X_mean, 2)
                    X_squared_sum = np.sum(X_squared, axis=0).ravel()
                    Xsquared_sum = packet['Xsquared_sum'] + X_squared_sum
                    roundrobin_addresses = packet['remain_addresses']
                    destiny_address = roundrobin_addresses[0]
                    remain_addresses = roundrobin_addresses[1:]
                    packet = {'action': packet['action'], 'Xsquared_sum': Xsquared_sum, 'remain_addresses': remain_addresses, 'xmean': xmean, 'sender': MLmodel.worker_address}
                    MLmodel.comms.send(packet, destiny_address)
            '''
            def while_computing_Npc(self, MLmodel, packet):
                npc_dict = {}
                ytr_str = MLmodel.ytr_orig
                ytr_str = [str(int(y)) for y in ytr_str]
                for cla in packet['classes']:
                    npc_dict.update({cla: ytr_str.count(cla)})
                packet = {'action': 'ACK_send_Npc', 'npc_dict': npc_dict, 'sender': MLmodel.worker_address}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_Npc' % (str(MLmodel.worker_address)))
                return

            def while_computing_sumXy(self, MLmodel, packet):
                X_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()
                y_sum = np.sum(MLmodel.ytr, axis=0).ravel()
                NP = MLmodel.Xtr_b.shape[0]  
                data = {'X_sum':X_sum , 'y_sum':y_sum, 'NP':NP}
                packet = {'action': 'ACK_send_sumXy', 'sender': MLmodel.worker_address, 'data':data}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_sumXy' % (str(MLmodel.worker_address)))
                return

            def while_computing_sumX(self, MLmodel, packet):
                try:
                    input_data_description = packet['data']['input_data_description']
                    which_variables = packet['data']['which_variables']
                    #X_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()      

                    X_sum = []
                    for kinput in range(input_data_description['NI']):
                        if input_data_description['input_types'][kinput]['type'] == 'num' or (input_data_description['input_types'][kinput]['type'] == 'bin' and which_variables=='all'):
                            aux = np.array(MLmodel.Xtr_orig)[:, kinput].astype(float)
                            X_sum.append(np.sum(aux))
                        else:
                            X_sum.append(np.NaN)

                    X_sum = np.array(X_sum).reshape(1, -1)

                    NP = np.array(MLmodel.Xtr_orig).shape[0]  
                    data = {'X_sum':X_sum , 'NP':NP}
                    packet = {'action': 'ACK_send_sumX', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_sumX' % (str(MLmodel.worker_address)))
                except Exception as err:
                    raise
                    '''
                    print('ERROR AT while_computing_sumX worker')
                    import code
                    code.interact(local=locals())
                    pass
                    '''

                return

            def while_computing_X_minus_mean_squared(self, MLmodel, packet):
                try:
                    mean_values = packet['data']['mean_values']
                    input_data_description = packet['data']['input_data_description']
                    which_variables = packet['data']['which_variables']

                    X_mean_squared_sum = []

                    for kinput in range(input_data_description['NI']):
                        if input_data_description['input_types'][kinput]['type'] == 'num' or (input_data_description['input_types'][kinput]['type'] == 'bin' and which_variables=='all'):
                            aux = np.array(MLmodel.Xtr_orig)[:, kinput].astype(float)
                            aux = aux - mean_values[0, kinput]
                            aux = aux ** 2
                            aux = np.sum(aux)
                            X_mean_squared_sum.append(np.sum(aux))
                        else:
                            X_mean_squared_sum.append(np.NaN)

                    X_mean_squared_sum = np.array(X_mean_squared_sum).reshape(1, -1)

                    #X2_mean_squared = (MLmodel.Xtr_b - mean_values) ** 2
                    #X2_mean_sum = np.sum(X2_mean, axis=0).ravel()
                    data = {'X_mean_squared_sum': X_mean_squared_sum}
                    packet = {'action': 'ACK_send_X2_mean_sum', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_X2_mean_sum' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_X_minus_mean_squared worker POM6_Common')
                    import code
                    code.interact(local=locals())
                    '''
                    pass
                return

            def while_computing_minX(self, MLmodel, packet):               
                try:
                    input_data_description = packet['data']['input_data_description']
                    #X_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()


                    X_min = []
                    X_max = []                    
                    for kinput in range(input_data_description['NI']):
                        if input_data_description['input_types'][kinput]['type'] == 'num':
                            aux = np.array(MLmodel.Xtr_orig)[:, kinput].astype(float)
                            X_min.append(np.min(aux))
                            X_max.append(np.max(aux))
                        else:
                            X_min.append(np.NaN)
                            X_max.append(np.NaN)

                    X_min = np.array(X_min).reshape(1, -1)
                    X_max = np.array(X_max).reshape(1, -1)

                    NP = np.array(MLmodel.Xtr_orig).shape[0]  
                    data = {'X_min':X_min , 'X_max':X_max, 'NP':NP}
                    packet = {'action': 'ACK_send_minX', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_minX' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_minX')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_computing_sumX_roundrobin(self, MLmodel, packet):
                try:
                    input_data_description = packet['data']['input_data_description']
                    which_variables = packet['data']['which_variables']
                    #X_sum = np.sum(MLmodel.Xtr_b, axis=0).ravel()
                    x_init = packet['data']['x_init']
                    NP_init = packet['data']['NP_init']

                    X_sum = []
                    for kinput in range(input_data_description['NI']):
                        if input_data_description['input_types'][kinput]['type'] == 'num'  or (input_data_description['input_types'][kinput]['type'] == 'bin' and which_variables=='all'):
                            aux = np.array(MLmodel.Xtr_orig)[:, kinput].astype(float)
                            X_sum.append(np.nansum(aux))
                        else:
                            X_sum.append(np.NaN)

                    X_sum = np.array(X_sum).reshape(1, -1)
                    NP = np.array(MLmodel.Xtr_orig).shape[0]  

                    x_init += X_sum
                    NP_init += NP                
                    data = {'input_data_description': input_data_description, 'x_init':x_init , 'NP_init':NP_init, 'which_variables':which_variables}
                    packet = {'action': 'get_sumX_roundrobin',  'to': 'CommonML', 'sender': MLmodel.worker_address, 'data':data}
                    # WARN, in localflask we do not send to ma...
                    MLmodel.comms.send(packet)
                    MLmodel.display(MLmodel.name + ' %s: sent get_sumX_roundrobin' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_sumX_roundrobin')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return


            def while_computing_X_minus_mean_squared_roundrobin(self, MLmodel, packet):
                try:
                    mean_values = packet['data']['mean_values']
                    input_data_description = packet['data']['input_data_description']
                    which_variables = packet['data']['which_variables']
                    x2_init = packet['data']['x2_init']
            
                    X_mean_squared_sum = []

                    for kinput in range(input_data_description['NI']):
                        if input_data_description['input_types'][kinput]['type'] == 'num'  or (input_data_description['input_types'][kinput]['type'] == 'bin' and which_variables=='all'):
                            aux = np.array(MLmodel.Xtr_orig)[:, kinput].astype(float)
                            aux = aux - mean_values[0, kinput]
                            aux = aux ** 2
                            aux = np.nansum(aux)
                            X_mean_squared_sum.append(np.sum(aux))
                        else:
                            X_mean_squared_sum.append(np.NaN)

                    X_mean_squared_sum = np.array(X_mean_squared_sum).reshape(1, -1)

                    x2_init += X_mean_squared_sum

                    data = {'input_data_description': input_data_description, 'x2_init':x2_init , 'mean_values':mean_values, 'which_variables':which_variables}
                    packet = {'action': 'get_X_minus_mean_squared_roundrobin',  'to': 'CommonML', 'sender': MLmodel.worker_address, 'data':data}
                    # WARN, in localflask we do not send to ma...
                    MLmodel.comms.send(packet)
                    MLmodel.display(MLmodel.name + ' %s: sent get_X_minus_mean_squared_roundrobin' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_X_minus_mean_squared_roundrobin worker POM6_Common')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_computing_stats(self, MLmodel, packet):

                stats_list = packet['data']['stats_list']

                stats_dict = MLmodel.compute_stats(MLmodel.Xtr_b, MLmodel.ytr.astype(float), stats_list)

                '''
                stats_dict = {}

                if 'rxy' in stats_list:
                    rxy = np.dot(MLmodel.Xtr_b.T, MLmodel.ytr.astype(float))
                    # unit norm
                    rxy = rxy / np.linalg.norm(rxy)
                    stats_dict.update({'rxy': rxy})

                if 'mx' in stats_list:
                    mx = np.mean(X, axis=0)
                    # unit norm
                    mx = mx / np.linalg.norm(mx)
                    stats_dict.update({'mx': mx})
                '''
                data = {'stats_dict':stats_dict}
                packet = {'action': 'ACK_send_stats', 'sender': MLmodel.worker_address, 'data':data}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_stats' % (str(MLmodel.worker_address)))
                return

            def while_computing_Rxyb_rxyb_direct(self, MLmodel, packet):
                MLmodel.display('PROC_WORKER_START', verbose=False)

                Xb = MLmodel.add_bias(MLmodel.Xtr_b)
                y = MLmodel.ytr.astype(float)
                
                Rxyb = np.dot(Xb.T, Xb)
                rxyb = np.dot(Xb.T, y)
                MLmodel.display('PROC_WORKER_END', verbose=False)

                data = {'Rxyb':Rxyb, 'rxyb': rxyb}
                action = 'ACK_send_Rxyb_rxyb_direct'
                packet = {'action': action, 'sender': MLmodel.worker_address, 'data':data}
                
                message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                packet.update({'message_id': message_id})
                MLmodel.message_counter += 1
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_Rxyb_rxyb_direct' % (str(MLmodel.worker_address)))
                return

            def while_computing_Rxyb_rxyb_roundrobin(self, MLmodel, packet):
                try:
                    Rxyb_aggr = packet['data']['Rxyb_aggr']
                    rxyb_aggr = packet['data']['rxyb_aggr']

                    Xb = MLmodel.add_bias(MLmodel.Xtr_b)
                    y = MLmodel.ytr.astype(float)
                    
                    Rxyb_aggr += np.dot(Xb.T, Xb)
                    rxyb_aggr += np.dot(Xb.T, y)

                    data = {'Rxyb_aggr':Rxyb_aggr, 'rxyb_aggr': rxyb_aggr}
                    packet = {'action': 'get_Rxyb_rxyb_roundrobin',  'to': 'CommonML', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent get_Rxyb_rxyb_roundrobin' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_Rxyb_rxyb_roundrobin worker POM6_Common')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_computing_vocab_direct(self, MLmodel, packet):
                try:
                    '''
                    # we compute tf here
                    vocab = []
                    for text in MLmodel.Xtr_b:
                        words = list(text[0].keys())
                        vocab = list(set(vocab + words))
                    '''
                    vocab_dict = {}
                    X = MLmodel.Xtr_b
                    Ndocs = X.shape[0]

                    for kdoc in tqdm(range(Ndocs)):
                        bow_dict = X[kdoc, 0]
                        vocab_dict = dict(Counter(vocab_dict) + Counter(bow_dict))

                    data = {'vocab_dict': vocab_dict}
                    packet = {'action': 'ACK_send_vocab_direct', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_vocab_direct' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_vocab_direct')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_computing_tf_df_direct(self, MLmodel, packet):
                try:
                    MLmodel.display(MLmodel.name + ' %s: computing df' % (str(MLmodel.worker_address)))

                    vocab = packet['data']['vocab']
                    X = MLmodel.Xtr_b
                    Ndocs = X.shape[0]
                    df_dict = {}

                    for kdoc in tqdm(range(Ndocs)):
                        bow_dict = X[kdoc, 0]
                        tmp_dict = {}
                        for key in bow_dict:
                            if key in vocab:
                                tmp_dict.update({key: 1})

                        df_dict = dict(Counter(df_dict) + Counter(tmp_dict))

                    data = {'Ndocs': Ndocs, 'df_dict': df_dict}
                    packet = {'action': 'ACK_send_tf_df_direct', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_tf_df_direct' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_df_direct')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_checking_data(self, MLmodel, packet):
                try:
                    MLmodel.display(MLmodel.name + ' %s: is checking data' % (str(MLmodel.worker_address)))
                    err = ''
                    try:
                        input_data_description = packet['data']['input_data_description']
                    except:
                        MLmodel.display(MLmodel.name + ' %s: input_data_description not available, not checking inputs.' % (str(MLmodel.worker_address)))
                        input_data_description = None
                        err += 'Missing input_data_description; '

                    try:
                        NI = MLmodel.Xtr_b.shape[1]
                    except:
                        MLmodel.display(MLmodel.name + ' %s: missing input_data.' % (str(MLmodel.worker_address)))
                        err += 'Missing input data; '
                        raise

                    try:
                        target_data_description = packet['data']['target_data_description']
                    except:
                        target_data_description = None

                    if target_data_description is None:
                        MLmodel.display(MLmodel.name + ' %s: target_data_description not available, not checking targets' % (str(MLmodel.worker_address)))

                    try:
                        NT = MLmodel.ytr.reshape(-1, 1).shape[1]
                    except:
                        NT = None    

                    if target_data_description is not None and NT is not None:
                        if NT != target_data_description['NT']:
                            err += 'Incorrect number of targets; '

                        try:
                            # Checking targets
                            for k in range(NT):
                                x = MLmodel.ytr[:, k]
                                xtype = target_data_description['output_types'][k]['type']
                                if xtype == 'num':
                                    try:
                                        x = x.astype(float)
                                    except:
                                        err += 'Target No. %d is not numeric; ' % k
                                        pass
                                if xtype == 'cat':
                                    try:
                                        x = set(x.astype(str))
                                        x = list(x - set(input_data_description['input_types'][k]['values']))
                                        if len(x) > 0:
                                            err += 'Target No. %d has an unrecognized categorical value; ' % k
                                    except:
                                        err += 'Target No. %d is not categorical; ' % k
                                        pass
                                if xtype == 'bin':
                                    try:
                                        x = x.astype(float)
                                        x = list(set(x) - set([0.0, 1.0]))
                                        if len(x) > 0:
                                            err += 'Target No. %d is not binary; ' % k
                                    except:
                                        err += 'Target No. %d is not binary; ' % k
                                        pass

                        except:
                            err += 'Unexpected error when processing targets; '
                            pass

                    if input_data_description is not None and NI is not None:
                        if NI != input_data_description['NI']:
                            err += 'Incorrect number of input features; '

                        if NT is not None:
                            if MLmodel.Xtr_b.shape[0] != MLmodel.ytr.shape[0]:
                                err += 'Different number of input and target patterns; '

                        try:
                            # Checking inputs
                            for k in range(NI):
                                x = MLmodel.Xtr_b[:, k]
                                xtype = input_data_description['input_types'][k]['type']
                                if xtype == 'num':
                                    try:
                                        x = x.astype(float)
                                    except:
                                        err += 'Input feature No. %d is not numeric; ' % k
                                        pass
                                if xtype == 'cat':
                                    try:
                                        x = set(x.astype(str))
                                        x = list(x - set(input_data_description['input_types'][k]['values']))
                                        if len(x) > 0:
                                            err += 'Input feature No. %d has an unrecognized categorical value; ' % k
                                    except:
                                        err += 'Input feature No. %d is not categorical; ' % k
                                        pass
                                if xtype == 'bin':
                                    try:
                                        x = x.astype(float)
                                        x = list(set(x) - set([0.0, 1.0]))
                                        if len(x) > 0:
                                            err += 'Input feature No. %d is not binary; ' % k
                                    except:
                                        err += 'INput feature No. %d is not binary; ' % k
                                        pass

                        except:
                            err += 'Unexpected error when processing input features; '
                            pass


                    data = {'err': err}
                    packet = {'action': 'ACK_checking_data', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_checking_data' % (str(MLmodel.worker_address)))
                except Exception as error:
                    raise
                    '''
                    print('ERROR AT while_checking_data')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_computing_feat_count_direct(self, MLmodel, packet):
                try:
                    X = MLmodel.Xtr_b
                    count = np.array(np.sum((X > 0).astype(float), axis=0)).ravel()
                    NP = X.shape[0]
                    data = {'NP': NP, 'count': count}

                    packet = {'action': 'ACK_send_feat_count_direct', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_feat_count_direct' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_vocab_direct')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_computing_hashids_direct(self, MLmodel, packet):
                try:
                    X = MLmodel.Xtr_b
                    ids = list(X[:,0])
                    MLmodel.hashids = [MLmodel.hash_sha256(id) for id in ids]

                    data = {'hashids': MLmodel.hashids}

                    packet = {'action': 'ACK_send_hashids_direct', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_send_hashids_direct' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_computing_hashids_direct')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_record_linkage(self, MLmodel, packet):
                try:
                    hashids = packet['data']['hashids']
                    Nhashids = len(hashids)

                    X = MLmodel.Xtr_b
                    y = MLmodel.ytr
                    NF = X.shape[1] - 1  # id is no longer stored

                    newX = np.empty((Nhashids, NF), dtype=np.dtype('U100'))
                    #newX[:] = np.nan

                    # warning, binary
                    newy = np.empty((Nhashids, 1), dtype=np.dtype('U100'))
                    #newy[:] = np.nan

                    for k, hashid in enumerate(hashids):
                        try:
                            pos = MLmodel.hashids.index(hashid)
                            newX[k, :] = X[pos, :][1:]
                            newy[k, :] = y[pos, :]
                        except:
                            pass
                    MLmodel.Xtr_b = newX
                    MLmodel.ytr = newy
                    MLmodel.display(MLmodel.name + ' %s: completed record linkage, input data has shape %dx%d' % (str(MLmodel.worker_address), newX.shape[0], newX.shape[1]))

                    # returning data description
                    data = {'input_data_description': MLmodel.input_data_description, 'target_data_description': MLmodel.target_data_description}
                    packet = {'action': 'ACK_record_linkage', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent ACK_record_linkage' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_record_linkage')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_data2num_at_worker_V(self, MLmodel, packet):
                try:
                    error = None
                    try:
                        model = MLmodel.get_data2num_model(MLmodel.input_data_description)
                        MLmodel.Xtr_b = model.transform(MLmodel.Xtr_b)
                        MLmodel.display(MLmodel.name + ' %s: transformed data to numeric, with shape %dx%d' % (str(MLmodel.worker_address), MLmodel.Xtr_b.shape[0], MLmodel.Xtr_b.shape[1]))
                        MLmodel.input_data_description = model.new_input_data_description
                    except Exception as err:
                        error = str(err)
                        pass
                    # returning data description
                    data = {'error': error, 'input_data_description': MLmodel.input_data_description, 'target_data_description': MLmodel.target_data_description}
                    packet = {'action': 'ACK_data2num_at_worker_V', 'sender': MLmodel.worker_address, 'data':data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ' %s: sent data2num_at_worker_V' % (str(MLmodel.worker_address)))
                except:
                    raise
                    '''
                    print('ERROR AT while_data2num_at_worker_V')
                    import code
                    code.interact(local=locals())                   
                    pass
                    '''
                return

            def while_local_preprocessing_V(self, MLmodel, prep_model):
                # We preprocess the original data
                X = MLmodel.Xtr_b
                error = 'data not processed'
                # We overwrite with the local data description
                prep_model.input_data_description = MLmodel.input_data_description

                data = {}
                if prep_model.name == 'normalization':
                    if prep_model.method == 'global_mean_std':
                        which_variables='all'
                        MLmodel.X_mean = []
                        MLmodel.X_std = []
                        for kinput in range(MLmodel.input_data_description['NI']):
                            if MLmodel.input_data_description['input_types'][kinput]['type'] == 'num' or (MLmodel.input_data_description['input_types'][kinput]['type'] == 'bin' and which_variables=='all'):
                                aux = np.array(MLmodel.Xtr_b)[:, kinput].astype(float)
                                MLmodel.X_mean.append(np.nanmean(aux))
                                MLmodel.X_std.append(np.nanstd(aux))
                            else:
                                MLmodel.X_mean.append(np.NaN)
                                MLmodel.X_std.append(np.NaN)

                        MLmodel.X_mean = np.array(MLmodel.X_mean).reshape(1, -1)
                        MLmodel.X_std = np.array(MLmodel.X_std).reshape(1, -1)
                        prep_model.mean = MLmodel.X_mean
                        prep_model.std = MLmodel.X_std
                        data = {'X_mean':MLmodel.X_mean, 'X_std':MLmodel.X_std}

                try:
                    #if prep_model.name in ['data2num', 'normalization', 'logscale', 'feature_extract', 'outlier_clipping', 'deep_learning']: # we replace the original data
                    X_prep = prep_model.transform(X)
                    MLmodel.Xtr_b = X_prep
                    MLmodel.display(MLmodel.name + ' %s: locally V preprocessing data with %s...' % (str(MLmodel.worker_address), prep_model.name))
                    MLmodel.display(MLmodel.name + ' %s: new data shape is %d x %d.' % (str(MLmodel.worker_address), X_prep.shape[0], X_prep.shape[1]))
                    MLmodel.Xtr_orig = X_prep
                    error = None
                except Exception as err:
                    raise
                    '''
                    error = err
                    # Comment later
                    print('ERROR AT while_local_preprocessing_V worker')
                    import code
                    code.interact(local=locals())
                    pass
                    '''

                if prep_model.name == 'missing_data_imputation_V':
                    data = {'X_mean':prep_model.means, 'X_std':None}

                action = 'ACK_local_prep_V'
                packet = {'action': action, 'sender': MLmodel.worker_address, 'error': error, 'data': data}
                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_local_prep_V' % (str(MLmodel.worker_address)))
                return

            def while_answering_ping(self, MLmodel):
                data = {'name': 'worker', 'address': MLmodel.worker_address}
                action = 'ACK_send_ping'
                packet = {'action': action, 'sender': MLmodel.worker_address, 'data':data}
                
                '''
                message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                packet.update({'message_id': message_id})
                MLmodel.message_counter += 1
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)
                '''

                message_id = 'empty'
                packet.update({'message_id': message_id})

                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_ping' % (str(MLmodel.worker_address)))
                return


            '''
            def while_sending_encr_data(self, MLmodel):
                try:
                    action = 'ACK_send_encr_data'
                    data = {'Xtr_b_encr': MLmodel.Xtr_b_encr, 'ytr_encr': MLmodel.ytr_encr}
                    packet = {'action': action, 'data': data}
                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_send_encr_data')
                except:
                    print('ERROR AT while_sending_encr_data')
                    import code
                    code.interact(local=locals())
                return
            '''

        states_worker = [
            'Exit',
            State(name='waiting_order', on_enter=['while_waiting_order']),

            #State(name='storing_prep_object', on_enter=['while_storing_prep_object']),
            #State(name='local_preprocessing', on_enter=['while_local_preprocessing']),

            State(name='storing_encrypt', on_enter=['while_storing_encrypt']),

            State(name='sending_encr_data', on_enter=['while_sending_encr_data']),

            State(name='storing_prep_object', on_enter=['while_storing_prep_object']),
            State(name='local_preprocessing', on_enter=['while_local_preprocessing']),
            #State(name='computing_roundrobin', on_enter=['while_computing_roundrobin']),
            State(name='computing_Npc', on_enter=['while_computing_Npc']),
            State(name='computing_sumXy', on_enter=['while_computing_sumXy']),
            State(name='computing_sumX', on_enter=['while_computing_sumX']),
            State(name='computing_X_minus_mean_squared', on_enter=['while_computing_X_minus_mean_squared']),
            State(name='computing_minX', on_enter=['while_computing_minX']),
            State(name='computing_sumX_roundrobin', on_enter=['while_computing_sumX_roundrobin']),
            State(name='computing_X_minus_mean_squared_roundrobin', on_enter=['while_computing_X_minus_mean_squared_roundrobin']),
            State(name='computing_stats', on_enter=['while_computing_stats']),
            State(name='computing_Rxyb_rxyb_direct', on_enter=['while_computing_Rxyb_rxyb_direct']),
            State(name='computing_Rxyb_rxyb_roundrobin', on_enter=['while_computing_Rxyb_rxyb_roundrobin']),
            State(name='computing_vocab_direct', on_enter=['while_computing_vocab_direct']),
            State(name='computing_tf_df_direct', on_enter=['while_computing_tf_df_direct']),
            State(name='computing_feat_count_direct', on_enter=['while_computing_feat_count_direct']),
            State(name='computing_hashids_direct', on_enter=['while_computing_hashids_direct']),
            State(name='record_linkage', on_enter=['while_record_linkage']),
            State(name='data2num_at_worker_V', on_enter=['while_data2num_at_worker_V']),
            State(name='checking_data', on_enter=['while_checking_data']),
            State(name='local_preprocessing_V', on_enter=['while_local_preprocessing_V']),
            State(name='answering_ping', on_enter=['while_answering_ping'])

        ]

        transitions_worker = [
            ['go_sending_encr_data', 'waiting_order', 'sending_encr_data'],
            ['done_sending_encr_data', 'sending_encr_data', 'waiting_order'],

            ['go_storing_prep_object', 'waiting_order', 'storing_prep_object'],
            ['done_storing_prep_object', 'storing_prep_object', 'waiting_order'],

            ['go_local_preprocessing', 'waiting_order', 'local_preprocessing'],
            ['done_local_preprocessing', 'local_preprocessing', 'waiting_order'],

            ['go_computing_Npc', 'waiting_order', 'computing_Npc'],
            ['done_computing_Npc', 'computing_Npc', 'waiting_order'],

            ['go_computing_sumXy', 'waiting_order', 'computing_sumXy'],
            ['done_computing_sumXy', 'computing_sumXy', 'waiting_order'],

            ['go_computing_sumX', 'waiting_order', 'computing_sumX'],
            ['done_computing_sumX', 'computing_sumX', 'waiting_order'],

            ['go_computing_X_minus_mean_squared', 'waiting_order', 'computing_X_minus_mean_squared'],
            ['done_computing_X_minus_mean_squared', 'computing_X_minus_mean_squared', 'waiting_order'],

            ['go_computing_minX', 'waiting_order', 'computing_minX'],
            ['done_computing_minX', 'computing_minX', 'waiting_order'],

            ['go_computing_sumX_roundrobin', 'waiting_order', 'computing_sumX_roundrobin'],
            ['done_computing_sumX_roundrobin', 'computing_sumX_roundrobin', 'waiting_order'],

            ['go_computing_X_minus_mean_squared_roundrobin', 'waiting_order', 'computing_X_minus_mean_squared_roundrobin'],
            ['done_computing_X_minus_mean_squared_roundrobin', 'computing_X_minus_mean_squared_roundrobin', 'waiting_order'],

            ['go_computing_stats', 'waiting_order', 'computing_stats'],
            ['done_computing_stats', 'computing_stats', 'waiting_order'],

            ['go_computing_Rxyb_rxyb_direct', 'waiting_order', 'computing_Rxyb_rxyb_direct'],
            ['done_computing_Rxyb_rxyb_direct', 'computing_Rxyb_rxyb_direct', 'waiting_order'],

            ['go_computing_Rxyb_rxyb_roundrobin', 'waiting_order', 'computing_Rxyb_rxyb_roundrobin'],
            ['done_computing_Rxyb_rxyb_roundrobin', 'computing_Rxyb_rxyb_roundrobin', 'waiting_order'],

            ['go_computing_vocab_direct', 'waiting_order', 'computing_vocab_direct'],
            ['done_computing_vocab_direct', 'computing_vocab_direct', 'waiting_order'],

            ['go_computing_tf_df_direct', 'waiting_order', 'computing_tf_df_direct'],
            ['done_computing_tf_df_direct', 'computing_tf_df_direct', 'waiting_order'],

            ['go_computing_feat_count_direct', 'waiting_order', 'computing_feat_count_direct'],
            ['done_computing_feat_count_direct', 'computing_feat_count_direct', 'waiting_order'],

            ['go_checking_data', 'waiting_order', 'checking_data'],
            ['done_checking_data', 'checking_data', 'waiting_order'],

            ['go_computing_hashids_direct', 'waiting_order', 'computing_hashids_direct'],
            ['done_computing_hashids_direct', 'computing_hashids_direct', 'waiting_order'],

            ['go_record_linkage', 'waiting_order', 'record_linkage'],
            ['done_record_linkage', 'record_linkage', 'waiting_order'],

            ['go_data2num_at_worker_V', 'waiting_order', 'data2num_at_worker_V'],
            ['done_data2num_at_worker_V', 'data2num_at_worker_V', 'waiting_order'],

            ['go_local_preprocessing_V', 'waiting_order', 'local_preprocessing_V'],
            ['done_local_preprocessing_V', 'local_preprocessing_V', 'waiting_order'],

            ['go_answering_ping', 'waiting_order', 'answering_ping'],
            ['done_answering_ping', 'answering_ping', 'waiting_order'],

            ['go_Exit', 'waiting_order', 'Exit']


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
        if packet['action'] not in ['ping', 'STOP']:
            try:
                self.display('COMMS_WORKER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('WORKER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass

        # Exit the process
        if packet['action'] == 'STOP':
            try: 
                with self.comms.commsffl:
                    self.comms.commsffl.leave_task()
            except:
                pass            
            self.display(self.name + ' %s: terminated by Master' % (str(self.worker_address)))
            self.display('EXIT_WORKER')
            self.terminate = True

        if packet['action'] == 'STOP_NOT_CLOSE_CONNECTION':
            self.display(self.name + ' %s: stopped by Master' % (str(self.worker_address)))
            self.terminate = True

        if packet['action'] == 'store_prep':
            self.FSMworker.go_storing_prep_object(self, packet)
            self.FSMworker.done_storing_prep_object(self)

        if packet['action'] == 'ask_encr_data':
            #self.encrypter = packet['data']['encrypter']
            #self.display('Worker stored encrypter')
            self.FSMworker.go_sending_encr_data(self, packet)
            self.FSMworker.done_sending_encr_data(self)

        if packet['action'] == 'do_local_prep':           
            self.FSMworker.go_local_preprocessing(self, packet['data']['prep_object'])
            self.FSMworker.done_local_preprocessing(self)

        if packet['action'] == 'count_patterns':
            self.FSMworker.go_computing_roundrobin(self, packet)
            self.FSMworker.done_computing_roundrobin(self)

        if packet['action'] == 'sum_patterns':
            self.FSMworker.go_computing_roundrobin(self, packet)
            self.FSMworker.done_computing_roundrobin(self)

        if packet['action'] == 'squared_sum_patterns':
            self.FSMworker.go_computing_roundrobin(self, packet)
            self.FSMworker.done_computing_roundrobin(self)

        if packet['action'] == 'get_Npc':
            self.FSMworker.go_computing_Npc(self, packet)
            self.FSMworker.done_computing_Npc(self)

        if packet['action'] == 'get_sumXy':           
            self.FSMworker.go_computing_sumXy(self, packet)
            self.FSMworker.done_computing_sumXy(self)

        if packet['action'] == 'get_sumX':           
            self.FSMworker.go_computing_sumX(self, packet)
            self.FSMworker.done_computing_sumX(self)

        if packet['action'] == 'get_X_minus_mean_squared':           
            self.FSMworker.go_computing_X_minus_mean_squared(self, packet)
            self.FSMworker.done_computing_X_minus_mean_squared(self)

        if packet['action'] == 'get_minX':           
            self.FSMworker.go_computing_minX(self, packet)
            self.FSMworker.done_computing_minX(self)

        if packet['action'] == 'get_sumX_roundrobin':           
            self.FSMworker.go_computing_sumX_roundrobin(self, packet)
            self.FSMworker.done_computing_sumX_roundrobin(self)

        if packet['action'] == 'get_X_minus_mean_squared_roundrobin':           
            self.FSMworker.go_computing_X_minus_mean_squared_roundrobin(self, packet)
            self.FSMworker.done_computing_X_minus_mean_squared_roundrobin(self)

        if packet['action'] == 'get_stats':           
            self.FSMworker.go_computing_stats(self, packet)
            self.FSMworker.done_computing_stats(self)

        if packet['action'] == 'get_Rxyb_rxyb_direct':           
            self.FSMworker.go_computing_Rxyb_rxyb_direct(self, packet)
            self.FSMworker.done_computing_Rxyb_rxyb_direct(self)

        if packet['action'] == 'get_Rxyb_rxyb_roundrobin':           
            self.FSMworker.go_computing_Rxyb_rxyb_roundrobin(self, packet)
            self.FSMworker.done_computing_Rxyb_rxyb_roundrobin(self)

        if packet['action'] == 'get_vocab_direct':           
            self.FSMworker.go_computing_vocab_direct(self, packet)
            self.FSMworker.done_computing_vocab_direct(self)

        if packet['action'] == 'get_tf_df_direct':           
            self.FSMworker.go_computing_tf_df_direct(self, packet)
            self.FSMworker.done_computing_tf_df_direct(self)

        if packet['action'] == 'get_feat_count_direct':           
            self.FSMworker.go_computing_feat_count_direct(self, packet)
            self.FSMworker.done_computing_feat_count_direct(self)

        if packet['action'] == 'get_hashids_direct':           
            self.FSMworker.go_computing_hashids_direct(self, packet)
            self.FSMworker.done_computing_hashids_direct(self)

        if packet['action'] == 'record_linkage':           
            self.FSMworker.go_record_linkage(self, packet)
            self.FSMworker.done_record_linkage(self)

        if packet['action'] == 'check_data':           
            self.FSMworker.go_checking_data(self, packet)
            self.FSMworker.done_checking_data(self)

        if packet['action'] == 'data2num_at_worker_V':           
            self.FSMworker.go_data2num_at_worker_V(self, packet)
            self.FSMworker.done_data2num_at_worker_V(self)

        if packet['action'] == 'do_local_prep_V':           
            self.FSMworker.go_local_preprocessing_V(self, packet['data']['prep_object_V'])
            self.FSMworker.done_local_preprocessing_V(self)

        if packet['action'] == 'ping':       
            self.FSMworker.go_answering_ping(self)
            self.FSMworker.done_answering_ping(self)

        return self.terminate


class POM4_CommonML_Crypto(Common_to_all_POMs):
    '''
    This class implements ML operations common to POM4 algorithms, run at Crypto node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.

    '''

    def __init__(self, cryptonode_address, master_address, model_type, comms, logger, verbose=False, cr=None):
        """
        Create a :class:`POM4_CommonML_Crypto` instance.

        Parameters
        ----------
        master_address: string
            address of the master node

        worker_address: string
            id of the worker

        model_type: string
            type of ML model

        comms: comms object instance
            object providing communications

        logger: class:`logging.Logger`
            logging object instance

        verbose: boolean
            indicates if messages are print or not on screen
   
        kwargs: Keyword arguments.

        """
        self.pom = 4
        self.master_address = master_address
        self.cryptonode_address = cryptonode_address                    # The id of this Worker
        self.comms = comms                      # The comms library
        self.model_type = model_type
        self.cr = cr
        self.encrypter = self.cr.get_encrypter()  # to be shared
        self.decrypter = self.cr.get_decrypter()  # to be kept as secret

        self.logger = logger                    # logger
        self.name = 'POM4_CommonML_Crypto'           # Name
        self.verbose = verbose                  # print on screen when true
        self.create_FSM_crypto()
        self.message_counter = 0 # used to number the messages
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
        self.display(self.name + ': creating FSM')

        class FSM_crypto(object):

            # Enter/exit callbacks are defined here

            def while_waiting_order(self, MLmodel):
                MLmodel.display('Crypto is waiting...')

            def while_sending_encrypter(self, MLmodel, packet):
                try:

                    '''
                    key_size = packet['data']['key_size']
                    crypt_library = packet['data']['crypt_library']
                    if crypt_library == 'phe':
                        from MMLL.crypto.crypt_PHE import Crypto as CR
                        MLmodel.cr = CR(key_size=key_size)
                        MLmodel.encrypter = MLmodel.cr.encrypter
                        MLmodel.decrypter = MLmodel.cr.decrypter
                    '''
                    # sending encrypter to workers and Master
                    action = 'ACK_sent_encrypter'
                    ############################################
                    # WARNING, remove decrypter, only for testing
                    #print('##### WARNING, sending decrypter, only for testing ####')
                    #data = {'decrypter': MLmodel.decrypter, 'encrypter': MLmodel.encrypter, 'sender': MLmodel.cryptonode_address}
                    ##########################################
                    data = {'encrypter': MLmodel.encrypter, 'sender': MLmodel.cryptonode_address}
                    '''
                    print('====== WARNING sending decrypter for debugging =======')
                    data.update({'decrypter': MLmodel.decrypter})
                    '''
                    packet = {'action': action, 'data': data, 'to': 'CommonML'}
                    # Sending encrypter to Master
                    
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)


                    MLmodel.comms.send(packet, MLmodel.master_address)
                    
                    # Sending params to Master
                    #MLmodel.comms.broadcast(packet, MLmodel.receivers_list)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_encrypter')
                except:
                    raise
                    '''
                    print('ERROR AT while_sending_encrypter')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_storing_Xy_bl(self, MLmodel, packet):
                try:

                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    MLmodel.X_bl_dict = {}
                    MLmodel.y_bl_dict = {}
                    MLmodel.X2_bl_encr_dict = {}
                    keys =  list(packet['data']['X_bl_dict'].keys())
                    for waddr in keys:
                        X_bl = MLmodel.decrypter.decrypt(packet['data']['X_bl_dict'][waddr])
                        MLmodel.X_bl_dict.update({waddr: X_bl})
                        MLmodel.X2_bl_encr_dict.update({waddr: MLmodel.encrypter.encrypt(X_bl * X_bl)})   
                        
                        y = packet['data']['y_bl_dict'][waddr]
                        if type(y) is not dict: 
                            MLmodel.y_bl_dict.update({waddr: MLmodel.decrypter.decrypt(packet['data']['y_bl_dict'][waddr])})   
                        if type(y) is dict:
                            y_bl_dict = {}
                            for cla in y.keys():
                                y_bl_dict.update({cla: MLmodel.decrypter.decrypt(y[cla])})
                            MLmodel.y_bl_dict.update({waddr: y_bl_dict})   
                        MLmodel.display('Decrypting blinded data from %s OK' % waddr)
                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    MLmodel.display(MLmodel.name + ': stored decrypted blinded data')
                    action = 'ACK_storing_Xy_bl'
                    data = {None}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_storing_Xy_bl')

                except:
                    raise
                    '''
                    print('ERROR AT while_storing_Xy_bl')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_storing_Kxc_bl(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    # We store the kernel values as the input training data
                    del MLmodel.X2_bl_encr_dict
                    MLmodel.X_bl_dict = {}

                    #Kxc_encr_bl_dict = packet['data']['Kxc_encr_bl_dict']
                    keys =  list(packet['data']['Kxc_encr_bl_dict'].keys())
                    for waddr in keys:
                        MLmodel.X_bl_dict.update({waddr: MLmodel.decrypter.decrypt(packet['data']['Kxc_encr_bl_dict'][waddr])})
                        MLmodel.display('Decrypting and storing blinded kernel data from %s OK' % waddr)

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_storing_Kxc_bl'
                    data = {None}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_storing_Kxc_bl')
                except:
                    raise
                    '''
                    print('ERROR AT while_storing_Xy_bl')
                    import code
                    code.interact(local=locals())
                    '''
                return


            def while_multiplying_XB(self, MLmodel, packet):


                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)
                    # Result in:
                    XB_bl_encr_dict = {}
                    MLmodel.display(MLmodel.name + ' is multiplying...')

                    B_bl_encr = packet['data']['B_bl']
                    is_empty = B_bl_encr is None  

                    if is_empty: # If it is empty, we return X_bl_squared
                        XB_bl_encr_dict = MLmodel.X2_bl_encr_dict
                    else:
                        is_dictionary = type(B_bl_encr) is dict

                        if is_dictionary:
                            MLmodel.workers_addresses = list(B_bl_encr.keys())

                        if not is_dictionary:
                            # Same value of B for all X
                            B_bl = MLmodel.decrypter.decrypt(B_bl_encr)
                            MQ, NQ = B_bl.shape

                        for waddr in MLmodel.workers_addresses:
                            if is_dictionary:
                                # Different values of B for every X
                                ### Overflow???
                                try:
                                    B_bl = MLmodel.decrypter.decrypt(B_bl_encr[waddr])
                                except:
                                    raise
                                    '''
                                    print('STOP AT while_multiplying_XB  overflow at crypto???--------')
                                    import code
                                    code.interact(local=locals())
                                    pass
                                    '''
                                MQ, NQ = B_bl.shape

                            X_bl = MLmodel.X_bl_dict[waddr]

                            MX, NX = X_bl.shape
                            if (MX == MQ and NQ == 1) or (MX == MQ and NQ == NX):
                                # B is of size MP, e.g., errors
                                XB_bl = X_bl * B_bl
                               
                            if (NX == NQ and MQ == 1):
                                # B is of size 1xNI, e.g., weights
                                XB_bl = B_bl * X_bl

                            try:
                                print('AT Common ML while_multiplying_XB')
                                print(MX, NX, MQ, NQ)
                                XB_bl_encr = MLmodel.encrypter.encrypt(XB_bl)
                            except: 
                                print('ERROR AT Common ML while_multiplying_XB')
                                import code
                                code.interact(local=locals())
                               
                            XB_bl_encr_dict.update({waddr: XB_bl_encr})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)
                        
                    action = 'ACK_sent_XB_bl_encr_dict'
                    data = {'XB_bl_encr_dict': XB_bl_encr_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_XB_bl_encr_dict')

                except:
                    raise
                    '''
                    print('ERROR AT while_multiplying_XB ############')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_multiplying_AB(self, MLmodel, packet):
                # A and B are two dictionaries
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)
                    # Result in:
                    AB_bl_encr_dict = {}
                    MLmodel.display(MLmodel.name + ' is multiplying...')
                    A_encr_bl_dict = packet['data']['A_encr_bl_dict']
                    B_encr_bl_dict = packet['data']['B_encr_bl_dict']

                    AB_bl_encr_dict = {}

                    for waddr in A_encr_bl_dict.keys():
                        A_bl = MLmodel.decrypter.decrypt(A_encr_bl_dict[waddr])
                        B_bl = MLmodel.decrypter.decrypt(B_encr_bl_dict[waddr])
                        AB_bl = A_bl * B_bl
                        AB_bl_encr_dict.update({waddr: MLmodel.encrypter.encrypt(AB_bl)})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_sent_AB_bl_encr_dict'
                    data = {'AB_bl_encr_dict': AB_bl_encr_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_AB_bl_encr_dict')

                except:
                    raise
                    '''
                    print('ERROR AT while_multiplying_AB ############')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_multiplying_XBM(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)
                    # Result in:
                    MLmodel.XB_bl_encr_dict = {}
                    MLmodel.display(MLmodel.name + ' is multiplying...')
                    
                    B_bl_encr = packet['data']['B_bl']

                    MLmodel.workers_addresses = list(B_bl_encr.keys())
                    MLmodel.classes = list(B_bl_encr[MLmodel.workers_addresses[0]].keys())

                    for waddr in MLmodel.workers_addresses:
                        X_bl = MLmodel.X_bl_dict[waddr]
                        MX, NX = X_bl.shape
                        XB_bl_encr_dict = {}
                        for cla in MLmodel.classes: 
                            B_bl = MLmodel.decrypter.decrypt(B_bl_encr[waddr][cla])
                            MB, NB = B_bl.shape

                            if (MX == MB and NB == 1) or (MX == MB and NB == NB):
                                # B is of size MP, e.g., errors
                                XB_bl = X_bl * B_bl

                            XB_bl_encr = MLmodel.encrypter.encrypt(XB_bl)
                            XB_bl_encr_dict.update({cla: XB_bl_encr})
                        MLmodel.XB_bl_encr_dict.update({waddr: XB_bl_encr_dict})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)
                        
                    action = 'ACK_sent_XBM_bl_encr_dict'
                    data = {'XB_bl_encr_dict': MLmodel.XB_bl_encr_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_XBM_bl_encr_dict')

                except:
                    raise
                    '''
                    print('ERROR AT while_multiplying_XB ############')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_decrypting_model(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    model_encr_bl = packet['data']['model_bl']
                    model_decr_bl = {}
                    for key in list(model_encr_bl.keys()):
                        model_decr_bl.update({key: MLmodel.cr.decrypter.decrypt(model_encr_bl[key])})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_sent_decr_bl_model'
                    data = {'model_decr_bl': model_decr_bl}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_decr_bl_model')
                except:
                    raise
                    '''
                    print('ERROR AT while_decr_model pom4commonml crypto')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_decrypting_modelM(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)
                    model_encr_bl_dict = packet['data']['model_bl']
                    model_decr_bl_dict = {}

                    for key in list(model_encr_bl_dict.keys()):
                        if key == 'wM':
                            classes = list(model_encr_bl_dict[key].keys())
                            tmp_dict = {}
                            for cla in classes: 
                                tmp_dict.update({cla: MLmodel.cr.decrypter.decrypt(model_encr_bl_dict[key][cla])})
                            model_decr_bl_dict.update({key: tmp_dict})
                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_sent_decr_bl_modelM'
                    data = {'model_decr_bl_dict': model_decr_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_decr_bl_modelM')
                except:
                    raise
                    '''
                    print('ERROR AT while_decr_modelM pom4commonml crypto')
                    import code
                    code.interact(local=locals())
                    '''
                return

            def while_compute_exp(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    s_encr_bl_dict = packet['data']['s_encr_bl_dict']
                    exps_bl_dict = {}
                    for waddr in s_encr_bl_dict.keys():
                        s_bl = MLmodel.decrypter.decrypt(s_encr_bl_dict[waddr])
                        exp_s_bl = np.exp(-s_bl)
                        which = exp_s_bl < 1e-10
                        exp_s_bl[which] = 0
                        exp_s_bl_encr = MLmodel.encrypter.encrypt(exp_s_bl)
                        exps_bl_dict.update({waddr: exp_s_bl_encr})
                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_exp_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'exps_bl_dict': exps_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common crypto while_compute_exp')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_expM(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    s_encr_bl_dict = packet['data']['s_encr_bl_dict']
                    exps_bl_dict = {}

                    workers = list(s_encr_bl_dict.keys())
                    classes = list(s_encr_bl_dict[workers[0]])
                    for waddr in workers:
                        cla_exps_bl_dict = {}
                        for cla in classes:
                            s_bl = MLmodel.decrypter.decrypt(s_encr_bl_dict[waddr][cla])
                            exp_s_bl = np.exp(-s_bl)
                            which = exp_s_bl < 1e-10
                            exp_s_bl[which] = 0
                            exp_s_bl_encr = MLmodel.encrypter.encrypt(exp_s_bl)
                            cla_exps_bl_dict.update({cla: exp_s_bl_encr})
                        exps_bl_dict.update({waddr: cla_exps_bl_dict})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_expM_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'exps_bl_dict': exps_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common crypto while_compute_exp')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_sort(self, MLmodel, packet):
                try: 
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    x_encr_bl = packet['data']['x_encr_bl']
                    x_bl = MLmodel.decrypter.decrypt(x_encr_bl)

                    index = np.argsort(-x_bl)

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_sort_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'index': index}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common while_compute_sort')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_div(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    num_bl_dict = packet['data']['num_bl_dict']
                    den_bl_dict = packet['data']['den_bl_dict']

                    sigm_encr_bl_dict = {}
                    for waddr in den_bl_dict.keys():
                        num_bl = MLmodel.decrypter.decrypt(num_bl_dict[waddr])
                        den_bl = MLmodel.decrypter.decrypt(den_bl_dict[waddr])
                        sigm_bl = num_bl / den_bl
                        sigm_encr_bl = MLmodel.encrypter.encrypt(sigm_bl)
                        sigm_encr_bl_dict.update({waddr: sigm_encr_bl})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_div_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'sigm_encr_bl_dict': sigm_encr_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common crypto while_compute_div')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_divM(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    num_bl_dict = packet['data']['num_bl_dict']
                    den_bl_dict = packet['data']['den_bl_dict']

                    workers = list(num_bl_dict.keys())
                    classes = list(num_bl_dict[workers[0]])
                    sigm_encr_bl_dict = {}
                    for waddr in workers:
                        cla_sigm_encr_bl_dict = {}
                        for cla in classes:
                            num_bl = MLmodel.decrypter.decrypt(num_bl_dict[waddr][cla])
                            den_bl = MLmodel.decrypter.decrypt(den_bl_dict[waddr][cla])
                            sigm_bl = num_bl / den_bl
                            sigm_encr_bl = MLmodel.encrypter.encrypt(sigm_bl)
                            cla_sigm_encr_bl_dict.update({cla: sigm_encr_bl})
                        sigm_encr_bl_dict.update({waddr: cla_sigm_encr_bl_dict})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)
                    
                    action = 'ACK_divM_bl'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'sigm_encr_bl_dict': sigm_encr_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common crypto while_compute_divM')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_argmin(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    c2_2XTC_bl_dict = packet['data']['c2_2XTC_bl_dict']
                    axis = 1
                    try:
                        axis = packet['data']['axis']
                    except: 
                        pass
                    argmin_dict = {}
                    for waddr in c2_2XTC_bl_dict.keys():
                        distXC_bl = MLmodel.decrypter.decrypt(c2_2XTC_bl_dict[waddr])
                        argmin_dict.update({waddr: np.argmin(distXC_bl, axis=axis)})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_compute_argmin'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'argmin_dict': argmin_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common crypto while_compute_argmin')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_compute_sign(self, MLmodel, packet):
                try:
                    MLmodel.display('PROC_CRYPTO_START', verbose=False)

                    sign_bl_dict = {}                    
                    for waddr in packet['data']['A_encr_bl_dict'].keys():
                        sign_bl = np.sign(MLmodel.decrypter.decrypt(packet['data']['A_encr_bl_dict'][waddr]))
                        sign_bl_dict.update({waddr: sign_bl})

                    MLmodel.display('PROC_CRYPTO_END', verbose=False)

                    action = 'ACK_compute_sign'
                    #message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    data = {'sign_bl_dict': sign_bl_dict}
                    packet = {'action': action, 'data': data, 'sender': MLmodel.cryptonode_address}
                    
                    message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    #del packet
                    MLmodel.display(MLmodel.name + ' %s: sent %s ' % (str(MLmodel.cryptonode_address), action))
                except:
                    raise
                    '''
                    print('ERROR AT Common crypto while_compute_sign')
                    import code
                    code.interact(local=locals())
                    pass
                    '''
                return

            def while_answering_ping(self, MLmodel):
                data = {'name': 'crypto', 'address': MLmodel.cryptonode_address}
                action = 'ACK_send_ping'
                packet = {'action': action, 'sender': MLmodel.cryptonode_address, 'data':data}
                
                '''
                message_id = 'crypto_' + MLmodel.cryptonode_address + '_' + str(MLmodel.message_counter)
                packet.update({'message_id': message_id})
                MLmodel.message_counter += 1
                size_bytes = asizeof.asizeof(dill.dumps(packet))
                MLmodel.display('COMMS_CRYPTO_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)
                '''
                message_id = 'empty'
                packet.update({'message_id': message_id})

                MLmodel.comms.send(packet, MLmodel.master_address)
                MLmodel.display(MLmodel.name + ' %s: sent ACK_send_ping' % (str(MLmodel.cryptonode_address)))
                return


        states_crypto = [
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='sending_encrypter', on_enter=['while_sending_encrypter']),
            State(name='storing_Xy_bl', on_enter=['while_storing_Xy_bl']),
            State(name='storing_Kxc_bl', on_enter=['while_storing_Kxc_bl']),
            State(name='multiplying_XB', on_enter=['while_multiplying_XB']),
            State(name='multiplying_AB', on_enter=['while_multiplying_AB']),
            State(name='multiplying_XBM', on_enter=['while_multiplying_XBM']),
            State(name='decrypting_model', on_enter=['while_decrypting_model']),
            State(name='decrypting_modelM', on_enter=['while_decrypting_modelM']),
            State(name='compute_exp', on_enter=['while_compute_exp']),
            State(name='compute_expM', on_enter=['while_compute_expM']),
            State(name='compute_div', on_enter=['while_compute_div']),
            State(name='compute_divM', on_enter=['while_compute_divM']),
            State(name='compute_sort', on_enter=['while_compute_sort']),
            State(name='answering_ping', on_enter=['while_answering_ping']),
            State(name='compute_argmin', on_enter=['while_compute_argmin']),
            State(name='compute_sign', on_enter=['while_compute_sign']),
            'Exit']

        transitions_crypto = [
            ['go_exit', 'waiting_order', 'Exit'],
            ['go_sending_encrypter', 'waiting_order', 'sending_encrypter'],
            ['go_waiting_order', 'sending_encrypter', 'waiting_order'],
            ['go_storing_Xy_bl', 'waiting_order', 'storing_Xy_bl'],
            ['go_storing_Kxc_bl', 'waiting_order', 'storing_Kxc_bl'],
            ['done_storing_Kxc_bl', 'storing_Kxc_bl', 'waiting_order'],
            ['go_waiting_order', 'storing_Xy_bl', 'waiting_order'],
            ['go_multiplying_XB', 'waiting_order', 'multiplying_XB'],
            ['done_multiplying_XB', 'multiplying_XB', 'waiting_order'],
            ['go_multiplying_AB', 'waiting_order', 'multiplying_AB'],
            ['done_multiplying_AB', 'multiplying_AB', 'waiting_order'],
            ['go_multiplying_XBM', 'waiting_order', 'multiplying_XBM'],
            ['done_multiplying_XBM', 'multiplying_XBM', 'waiting_order'],
            ['go_decrypting_model', 'waiting_order', 'decrypting_model'],
            ['done_decrypting_model', 'decrypting_model', 'waiting_order'],
            ['go_decrypting_modelM', 'waiting_order', 'decrypting_modelM'],
            ['done_decrypting_modelM', 'decrypting_modelM', 'waiting_order'],
            ['go_compute_exp', 'waiting_order', 'compute_exp'],
            ['done_compute_exp', 'compute_exp', 'waiting_order'],
            ['go_compute_expM', 'waiting_order', 'compute_expM'],
            ['done_compute_expM', 'compute_expM', 'waiting_order'],
            ['go_compute_div', 'waiting_order', 'compute_div'],
            ['done_compute_div', 'compute_div', 'waiting_order'],
            ['go_compute_divM', 'waiting_order', 'compute_divM'],
            ['done_compute_divM', 'compute_divM', 'waiting_order'],
            ['go_compute_sort', 'waiting_order', 'compute_sort'],
            ['done_compute_sort', 'compute_sort', 'waiting_order'],
            ['go_answering_ping', 'waiting_order', 'answering_ping'],
            ['done_answering_ping', 'answering_ping', 'waiting_order'],
            ['go_compute_argmin', 'waiting_order', 'compute_argmin'],
            ['done_compute_argmin', 'compute_argmin', 'waiting_order'],
            ['go_compute_sign', 'waiting_order', 'compute_sign'],
            ['done_compute_sign', 'compute_sign', 'waiting_order']
        ]

        self.FSMcrypto = FSM_crypto()
        self.grafmachine_crypto = GraphMachine(model=self.FSMcrypto,
            states=states_crypto,
            transitions=transitions_crypto,
            initial='waiting_order',
            show_auto_transitions=False,  # default value is False
            title="Finite State Machine modelling the behaviour of Common Cryptonode",
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
        #self.display(self.name + ': received %s from worker %s' % (packet['action'], sender), verbose=True)
        if packet['action'] not in ['update_tr_data', 'ping', 'ask_encr_data', 'STOP']:
            try:
                self.display('COMMS_CRYPTO_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
            except:
                self.display('CRYPTO MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                pass

        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' % (str(self.cryptonode_address)))
            self.terminate = True
            self.display('EXIT_CRYPTO')

        if packet['action'] == 'ask_encrypter':
            self.FSMcrypto.go_sending_encrypter(self, packet)
            self.FSMcrypto.go_waiting_order(self)

        if packet['action'] == 'send_Xy_bl':
            self.FSMcrypto.go_storing_Xy_bl(self, packet)
            self.FSMcrypto.go_waiting_order(self)

        if packet['action'] == 'store_Kxc_bl':
            self.FSMcrypto.go_storing_Kxc_bl(self, packet)
            self.FSMcrypto.done_storing_Kxc_bl(self)

        if packet['action'] == 'send_mult_XB':
            self.FSMcrypto.go_multiplying_XB(self, packet)
            self.FSMcrypto.done_multiplying_XB(self)

        if packet['action'] == 'send_mult_AB':
            self.FSMcrypto.go_multiplying_AB(self, packet)
            self.FSMcrypto.done_multiplying_AB(self)

        if packet['action'] == 'send_mult_XBM':
            self.FSMcrypto.go_multiplying_XBM(self, packet)
            self.FSMcrypto.done_multiplying_XBM(self)

        if packet['action'] == 'send_model_encr_bl':
            self.FSMcrypto.go_decrypting_model(self, packet)
            self.FSMcrypto.done_decrypting_model(self)

        if packet['action'] == 'send_modelM_encr_bl':
            self.FSMcrypto.go_decrypting_modelM(self, packet)
            self.FSMcrypto.done_decrypting_modelM(self)

        if packet['action'] == 'ask_exp_bl':
            self.FSMcrypto.go_compute_exp(self, packet)
            self.FSMcrypto.done_compute_exp(self)

        if packet['action'] == 'ask_expM_bl':
            self.FSMcrypto.go_compute_expM(self, packet)
            self.FSMcrypto.done_compute_expM(self)

        if packet['action'] == 'ask_div_bl':
            self.FSMcrypto.go_compute_div(self, packet)
            self.FSMcrypto.done_compute_div(self)

        if packet['action'] == 'ask_divM_bl':
            self.FSMcrypto.go_compute_divM(self, packet)
            self.FSMcrypto.done_compute_divM(self)

        if packet['action'] == 'ask_sort_bl':
            self.FSMcrypto.go_compute_sort(self, packet)
            self.FSMcrypto.done_compute_sort(self)

        if packet['action'] == 'ping':           
            self.FSMcrypto.go_answering_ping(self)
            self.FSMcrypto.done_answering_ping(self)

        if packet['action'] == 'ask_argmin_bl':
            self.FSMcrypto.go_compute_argmin(self, packet)
            self.FSMcrypto.done_compute_argmin(self)

        if packet['action'] == 'ask_sign_bl':
            self.FSMcrypto.go_compute_sign(self, packet)
            self.FSMcrypto.done_compute_sign(self)

        return self.terminate

