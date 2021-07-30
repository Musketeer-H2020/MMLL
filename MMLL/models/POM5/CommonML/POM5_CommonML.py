# -*- coding: utf-8 -*-
'''
Common ML operations to be used by all algorithms in POM5

'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Nov. 2020"

import numpy as np
from MMLL.models.Common_to_all_POMs import Common_to_all_POMs
from transitions import State
from transitions.extensions import GraphMachine
import math
from collections import Counter 
from tqdm import tqdm   # pip install tqdm
import pickle
from pympler import asizeof #asizeof.asizeof(my_object)
import dill
import time

class POM5_CommonML_Master(Common_to_all_POMs):
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
        self.name = 'POM5_CommonML_Master'               # Name
        self.master_address = master_address
        self.cryptonode_address = 'ca'

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

        self.cryptonode_address = None

        self.comms = comms                          # comms lib
        self.logger = logger                        # logger
        self.verbose = verbose                      # print on screen when true

        self.process_kwargs(kwargs)
        self.encrypter = self.cr.get_encrypter()  # to be shared
        self.decrypter = self.cr.get_decrypter()  # to be kept as secret
        self.message_counter = 1    # used to number the messages

        self.state_dict = None                      # State of the main script
        self.state_dict = {}                        # dictionary storing the execution state
        self.NI = None
        self.NI_dict = {}
        #self.Data_encr_dict = {}  # X data encrypted with PK from every user
        #self.Xq_prodpk_dict = {}  # X data reencypted with PRodPK
        #self.yq_prodpk_dict = {}  # y data reencypted with PRodPK

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
        self.display(self.name + ': creating Common ML FSM, POM5')

        states_master = [
            State(name='waiting_order', on_enter=['while_waiting_order']),

            State(name='sending_prep_object', on_enter=['while_sending_prep_object']),
            State(name='asking_local_prep', on_enter=['while_asking_local_prep']),

            State(name='bcasting_encrypter', on_enter=['while_bcasting_encrypter']),

            State(name='terminating_workers', on_enter=['while_terminating_workers']),
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

            State(name='Exit', on_enter=['while_Exit'])
        ]

        transitions_master = [
            ['go_sending_prep_object', 'waiting_order', 'sending_prep_object'],
            #['go_asking_local_prep', 'sending_prep_object', 'asking_local_prep'],
            ['go_waiting_order', 'sending_prep_object', 'waiting_order'],

            ['go_bcasting_encrypter', 'waiting_order', 'bcasting_encrypter'],
            ['go_waiting_order', 'bcasting_encrypter', 'waiting_order'],

            ['go_terminating_workers', 'waiting_order', 'terminating_workers'],
            ['go_waiting_order', 'terminating_workers', 'waiting_order'],

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

            ['go_Exit', 'waiting_order', 'Exit'],
            ['go_waiting_order', 'Exit', 'waiting_order']
        ]

        class FSM_master(object):

            def while_Exit(self, MLmodel):
                #print(MLmodel.name + 'while_Exit')
                return

            def while_waiting_order(self, MLmodel):
                MLmodel.display(MLmodel.name + ' is waiting...')
                return

            def while_bcasting_encrypter(self, MLmodel):
                try:
                    # Communicating encrypter to workers
                    data = {'encrypter': MLmodel.cr.encrypter}
                    # For checking, REMOVE
                    message_id = MLmodel.master_address + str(MLmodel.message_counter)
                    MLmodel.message_counter += 1
                    #data.update({'decrypter': MLmodel.cr.decrypter})
                    action = 'send_encrypter'
                    packet = {'action': action, 'to': 'CommonML', 'data': data, 'sender': MLmodel.master_address, 'message_id': message_id}
                    
                    message_id = MLmodel.master_address+'_'+str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.broadcast(packet)
                    MLmodel.display(MLmodel.name + ' broadcast encrypter to all Workers')
                except:
                    raise
                    '''
                    print('ERROR AT while_bcasting_encrypter')
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
                MLmodel.comms.broadcast(packet)
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
                action = 'get_Rxyb_rxyb_roundrobin'
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
        self.message_counter += 1            
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

    ######## UNCHECKED

    def send_encrypter(self):
        """
        Send encrypter to all nodes

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_bcasting_encrypter(self)
        self.run_Master()



    def get_cryptdata(self):
        """
        Get encrypted data from workers, under POM4

        Parameters
        ----------
        None
        """
        self.FSMmaster.go_asking_encrypted_data(self)
        self.run_Master()


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
            prep_object: preprocessing instance
                Preprocessing object to be used.
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

        if self.chekAllStates('ACK_sent_encrypter'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_stored_prep'):
            self.FSMmaster.go_asking_local_prep(self)
        """
        if self.chekAllStates('ACK_encr_preds'):
            self.FSMmaster.go_sending_sorting(self)
        """
        if self.chekAllStates('ACK_sending_qysv'):
            self.FSMmaster.go_sending_sorting(self)

        if self.chekAllStates('ACK_local_prep'):
            self.FSMmaster.go_waiting_order(self)

        if self.chekAllStates('ACK_computing_AUCs'):
            self.FSMmaster.go_waiting_order(self)

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
            self.state_dict[sender] = packet['action']

            if packet['action'][0:3] == 'ACK':
                self.display('Master received ACK from %s: %s' % (str(sender), packet['action']))
                self.state_dict[sender] = packet['action']
                try:
                    self.display('COMMS_MASTER_RECEIVED %s from %s, id=%s' % (packet['action'], sender, str(packet['message_id'])), verbose=False)
                except:
                    self.display('MASTER MISSING message_id in %s from %s' % (packet['action'], sender), verbose=False)                    
                    pass

            if packet['action'] == 'ACK_sent_encrypter':
                self.NI_dict.update({sender: packet['data']['NI']})
                # This part could be moved to a more general first step retrieving the feature characteristics...
                #self.send_to.update({sender: packet['pseudo_id']})
                #self.worker_names.update({sender: packet['sender_']})

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

        except:
            raise
            '''
            print('ERROR AT ProcessReceivedPacket_Master')
            import code
            code.interact(local=locals())
            pass
            '''

        return


class POM5_CommonML_Worker(Common_to_all_POMs):
    '''
    This class implements ML operations common to POM5 algorithms, run at Worker node. To be inherited by the specific ML models. It inherits from Common_to_all_POMs.

    '''

    def __init__(self, master_address, worker_address, model_type, comms, logger, verbose=False, Xtr_b=None, ytr=None):
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
        #self.workers_addresses = workers_addresses                    # The id of this Worker
        self.comms = comms                      # The comms library
        self.model_type = model_type
        #self.cr = cr
        self.logger = logger                    # logger
        self.name = 'POM5_CommonML_Worker'           # Name
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
                MLmodel.display(MLmodel.name + ' %s is waiting...' % (str(MLmodel.worker_address)))
                return

            def while_storing_prep_object(self, MLmodel, packet):
                try:
                    MLmodel.prep = packet['data']['prep_object']
                    MLmodel.display(MLmodel.name + ' %s: stored preprocessing object' % (str(MLmodel.worker_address)))
                    action = 'ACK_stored_prep'
                    packet = {'action': action}
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

            def while_storing_encrypter(self, MLmodel, packet):
                try:
                    MLmodel.encrypter = packet['data']['encrypter']
                    #MLmodel.decrypter = packet['data']['decrypter']
                    action = 'ACK_sent_encrypter'
                    NI = MLmodel.Xtr_b.shape[1]
                    data = {'NI': NI}
                    packet = {'action': action, 'data': data, 'sender': str(MLmodel.worker_address)}
                    # Sending params to Master
                    
                    message_id = 'worker_' + MLmodel.worker_address + '_' + str(MLmodel.message_counter)
                    packet.update({'message_id': message_id})
                    MLmodel.message_counter += 1
                    size_bytes = asizeof.asizeof(dill.dumps(packet))
                    MLmodel.display('COMMS_WORKER_SEND %s to %s, id = %s, bytes=%s' % (action, MLmodel.master_address, message_id, str(size_bytes)), verbose=False)

                    MLmodel.comms.send(packet, MLmodel.master_address)
                    MLmodel.display(MLmodel.name + ': sent ACK_sent_encrypter')
                except:
                    raise
                    '''
                    print('STOP AT while_storing_encrypter')
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

                Xb = MLmodel.add_bias(MLmodel.Xtr_b)
                y = MLmodel.ytr.astype(float)
                
                Rxyb = np.dot(Xb.T, Xb)
                rxyb = np.dot(Xb.T, y)

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

        '''
        with open('../MMLL/models/POM5/CommonML/POM5_CommonML_FSM_worker.pkl', 'rb') as f:
            [states_worker, transitions_worker] = pickle.load(f)
        '''

        states_worker = [
            'Exit',
            State(name='waiting_order', on_enter=['while_waiting_order']),
            State(name='storing_prep_object', on_enter=['while_storing_prep_object']),
            State(name='local_preprocessing', on_enter=['while_local_preprocessing']),
            State(name='storing_encrypter', on_enter=['while_storing_encrypter']),

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
        ]

        transitions_worker = [
            ['go_storing_prep_object', 'waiting_order', 'storing_prep_object'],
            ['done_storing_prep_object', 'storing_prep_object', 'waiting_order'],

            ['go_local_preprocessing', 'waiting_order', 'local_preprocessing'],
            ['done_local_preprocessing', 'local_preprocessing', 'waiting_order'],

            ['go_storing_encrypter', 'waiting_order', 'storing_encrypter'],
            ['done_storing_encrypter', 'storing_encrypter', 'waiting_order'],

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

        if packet['action'] == 'do_local_prep':           
            self.FSMworker.go_local_preprocessing(self, packet['data']['prep_object'])
            self.FSMworker.done_local_preprocessing(self)

        if packet['action'] == 'send_encrypter':
            self.FSMworker.go_storing_encrypter(self, packet)
            self.FSMworker.done_storing_encrypter(self)

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

        return self.terminate

