# -*- coding: utf-8 -*-
'''
Collection of methods common to POMs 1, 2 and 3. To be inherited by the ML classes

'''

__author__ = "Marcos Fernández Díaz"
__date__ = "December 2020"


import sys
import numpy as np
from collections import Counter 

from MMLL.Common_to_all_objects import Common_to_all_objects



class Common_to_POMs_123_Master(Common_to_all_objects):
    """
    This class implements basic methods and protocols common to POMs 1, 2 and 3 for the master.
    To be inherited by the specific ML models. Not every method is used by every POM.
    """

    def __init__(self):
        """
        Create a :class:`Common_to_POMs_123_Master` instance.
        Parameters
        ----------
        None
        """
        return



    def terminate_workers_(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        users_addresses_terminate: List of strings
            Addresses of the workers to be terminated

        """
        packet = {'action': 'STOP', 'to': 'CommonML'}

        if workers_addresses_terminate is None:  # We terminate all of them
            workers_addresses_terminate = self.workers_addresses
            self.display(self.name + ': Sent STOP to all workers')
            self.comms.broadcast(packet, workers_addresses_terminate)
        else:
            self.display(self.name + ': Sending STOP to %d workers' %len(workers_addresses_terminate))
            for worker in workers_addresses_terminate:
                if worker not in self.workers_addresses:
                    raise Exception('WorkerNotFound')
                self.comms.send(packet, worker)
                self.display(self.name + ': Sent STOP to worker %s' %worker)

        # Updating the list of active users
        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)
        for worker in workers_addresses_terminate: 
            self.state_dict.pop(worker)



    def checkAllStates(self, condition, state_dict):
        """
        Checks if all worker states satisfy a given condition

        Parameters
        ----------
        condition: String
            Condition to check
        state_dict: Dictionary
            Dictionary whose values need to be compared against condition

        Returns
        ----------
        all_active: Boolean
            Flag indicating if all values inside dictionary are equal to condition
        """
        all_active = True
        for worker in self.workers_addresses:
            if state_dict[worker] != condition:
                all_active = False
                break
        return all_active



    def train_Master(self):
        """
        This is the main training loop, it runs the following actions until the stop condition is met:
            - Update the execution state
            - Perform actions according to the state
            - Process the received packets

        Parameters
        ----------
        None
        """
        if self.Nworkers == 0:
            raise ValueError('Empty list of workers to train')

        else:
            self.iter = 0
            self.state_dict.update({'CN': 'START_TRAIN'})
            self.display(self.name + ': Starting training')

            if self.name.startswith('POM3'):
                self.train_Master_()

            else: # POMs 1 and 2
                while self.state_dict['CN'] != 'END':
                    self.Update_State_Master()
                    self.TakeAction_Master()
                    self.CheckNewPacket_Master()            
                self.display(self.name + ': Training is done')



    def CheckNewPacket_Master(self):
        """
        Checks if there is a new message in the Master queue

        Parameters
        ----------
            None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(timeout=0.1) # We only receive a dictionary at a time even if there are more than 1 workers
                try:  # For the pycloudmessenger cloud
                    sender = packet.notification['participant']
                except Exception: # For the pycloudmessenger local
                    self.counter = (self.counter + 1) % self.Nworkers
                    sender = self.workers_addresses[self.counter]
                    
                packet = packet.content
                self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                self.ProcessReceivedPacket_Master(packet, sender)
            except KeyboardInterrupt:
                self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                sys.exit()
            except Exception as err:
                if 'pycloudmessenger.ffl.fflapi.TimedOutException' in str(type(err)):
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise
        else: # Local flask
            packet = None
            sender = None
            for sender in self.workers_addresses:
                try:
                    packet = self.comms.receive(sender, timeout=0.1)
                    self.display(self.name + ': Received %s from worker %s' %(packet['action'], sender))
                    self.ProcessReceivedPacket_Master(packet, sender)
                except KeyboardInterrupt:
                    self.display(self.name + ': Shutdown requested by Keyboard...exiting')
                    sys.exit()
                except Exception as err:
                    if str(err).startswith('Timeout when receiving data'): # TimedOutException
                        pass
                    else:
                        self.display(self.name + ': Error %s' %err)
                        raise



    def ProcessReceivedPacket_Master(self, packet, sender):
        """
        Process the received packet at Master and take some actions, possibly changing the state

        Parameters
        ----------
        packet: Dictionary
            Packet received

        sender: String
            Id of the sender
        """
        if packet['action'][0:3] == 'ACK':
            self.state_dict[sender] = packet['action']
            if self.checkAllStates('ACK_FINAL_MODEL', self.state_dict): # Included here to avoid calling CheckNewPacket_Master after sending the final model (this call could imply significant delay if timeout is set to a high value)
                self.state_dict['CN'] = 'END'
            if packet['action'] == 'ACK_SEND_PREPROCESSOR':
                if packet['error'] is not None:
                    self.worker_errors[sender] = packet['error']
            if packet['action'] == 'ACK_GET_STATS':
                self.stats_dict[sender] = packet['data']['stats_dict']
            if packet['action'] == 'ACK_CHECK_DATA':
                self.worker_errors[sender] = packet['error']
            if packet['action'] == 'ACK_get_Rxyb_rxyb':
                self.Rxyb_dict[sender] = packet['data']['Rxyb']
                self.rxyb_dict[sender] = packet['data']['rxyb']
            if packet['action'] == 'ACK_GET_VOCABULARY':
                self.vocab_dict[sender] = packet['data']['vocab_dict']
            if packet['action'] == 'ACK_GET_DF':
                self.Ndocs_dict[sender] = packet['data']['Ndocs']
                self.df_dict[sender] = packet['data']['df_dict']
            if packet['action'] == 'ACK_GET_FEAT_COUNT':
                self.NP_dict[sender] = packet['data']['NP']
                self.count_dict[sender] = packet['data']['count']
            if packet['action'] == 'ACK_GET_HASH_IDS':
                self.hashids_dict[sender] = packet['data']['hashids']
            if packet['action'] == 'ACK_GET_RECORD_LINKAGE':
                self.input_data_description_dict[sender] = packet['data']['input_data_description']
                self.target_data_description_dict[sender] = packet['data']['target_data_description']
            if packet['action'] == 'ACK_DATA2NUM_V':
                self.input_data_description_dict[sender] = packet['data']['input_data_description']
                self.target_data_description_dict[sender] = packet['data']['target_data_description']
                self.errors_dict[sender] = packet['data']['error']
            if packet['action'] == 'ACK_SEND_PREPROCESSOR_V':
                self.mean_dict[sender] = packet['data']['X_mean']
                self.std_dict[sender] = packet['data']['X_std']
                self.worker_errors[sender] = packet['error']

        else:
            self.ProcessReceivedPacket_Master_(packet, sender) # Call individual method for each of the algorithms



    def send_preprocessor(self, prep_model):
        self.worker_errors = {}

        # Send message to every worker
        action = 'SEND_PREPROCESSOR'
        data = {'prep_model': prep_model}
        packet = {'action': action, 'data': data, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_SEND_PREPROCESSOR', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''
        return self.worker_errors



    def get_stats(self, stats_list):
        """
        Gets from workers their statistics

        Parameters
        ----------
        stats_list: List of strings
            List of statistics to be computed by every worker
        """
        self.stats_dict = {}

        # Send message to every worker
        action = 'GET_STATS'
        data = {'stats_list': stats_list}
        packet = {'action': action, 'data': data, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_GET_STATS', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''
        return self.stats_dict



    def send_check(self, input_data_description, target_data_description):
        self.worker_errors = {}

        # Send message to every worker
        action = 'CHECK_DATA'
        data = {'input_data_description': input_data_description, 'target_data_description': target_data_description}
        packet = {'action': action, 'data': data, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_CHECK_DATA', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''
        return self.worker_errors



    def get_Rxyb_rxyb(self):
        """
        Obtaining get_Rxyb_rxyb from workers
        Parameters
        ----------
            None
        """
        self.Rxyb_dict = {}
        self.rxyb_dict = {}

        # Send message to every worker
        action = 'get_Rxyb_rxyb'
        packet = {'action': action, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_get_Rxyb_rxyb', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        workers = list(self.Rxyb_dict.keys())
        Rxy_b = self.Rxyb_dict[workers[0]]
        rxy_b = self.rxyb_dict[workers[0]]
        for worker in workers[1:]:
            Rxy_b += self.Rxyb_dict[worker]
            rxy_b += self.rxyb_dict[worker]

        self.display(self.name + ': Computation of Rxyb_rxyb ready')
        return Rxy_b, rxy_b



    def get_vocabulary(self):
        """
        Gets from workers their vocabulary
        Parameters
        ----------
        None
        """
        self.vocab_dict = {}

        # Send message to every worker
        action = 'GET_VOCABULARY'
        packet = {'action': action, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_GET_VOCABULARY', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        self.global_tf_dict = {}
        for waddr in self.workers_addresses:
            self.global_tf_dict = dict(Counter(self.global_tf_dict) + Counter(self.vocab_dict[waddr]))
        self.display(self.name + ': Length of tf dictionary: %d' %len(self.global_tf_dict))

        self.global_tf_dict_filtered = {}
        for key in self.global_tf_dict.keys(): # At least 10 times, and less than 5000
            if self.global_tf_dict[key] > 10 and self.global_tf_dict[key] < 5000:
                self.global_tf_dict_filtered[key] = self.global_tf_dict[key]
        self.display(self.name + ': Length of filtered tf dictionary: %d' %len(self.global_tf_dict_filtered))

        vocab = list(self.global_tf_dict_filtered.keys())
        vocab.sort()
        self.vocab = vocab
        return self.vocab



    def get_df(self, vocab):
        """
        Gets df and Ndocs from workers
        Parameters
        ----------
        None
        """
        self.Ndocs_dict = {}
        self.df_dict = {}

        # Send message to every worker
        data = {'vocab': vocab} 
        action = 'GET_DF'
        packet = {'action': action, 'to': 'CommonML', 'data': data}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_GET_DF', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        self.global_df_dict = {}
        for waddr in self.workers_addresses:
            self.global_df_dict = dict(Counter(self.global_df_dict) + Counter(self.df_dict[waddr]))

        self.Ndocs = 0
        for waddr in self.workers_addresses:
            self.Ndocs += self.Ndocs_dict[waddr]
        self.display(self.name + ': Length of df dictionary: %d' % len(self.global_df_dict))

        self.global_df_dict_filtered = {}
        for key in self.global_df_dict.keys():
            if self.global_df_dict[key] > 10: # at least in 10 docs
                self.global_df_dict_filtered[key] = self.global_df_dict[key]
        self.display(self.name + ': Length of filtered df dictionary: %d' % len(self.global_df_dict_filtered))

        vocab = list(self.global_df_dict_filtered.keys())
        vocab.sort()
        self.vocab = vocab
        return self.vocab, self.global_df_dict_filtered



    def get_feat_count(self):
        """
        Gets feature frequency from workers
        Parameters
        ----------
        None
        """
        self.count_dict = {}
        self.NP_dict = {}

        # Send message to every worker
        action = 'GET_FEAT_COUNT'
        packet = {'action': action, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_GET_FEAT_COUNT', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''
            
        NP = np.sum(np.array(list(self.NP_dict.values())))
        count = np.sum(np.array(list(self.count_dict.values())), axis=0)
        self.display(self.name + ': Getting feat freq ready')

        return count, NP



    def data2num_at_workers_V(self):
        """
        Asks workers to transform their data  into numeric
        Parameters
        ----------
        None
        """
        self.input_data_description_dict = {}
        self.target_data_description_dict = {}
        self.errors_dict = {}

        # Send message to every worker
        action = 'DATA2NUM_V'
        packet = {'action': action, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_DATA2NUM_V', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        self.display(self.name + ': data2num at workers ready')
        return self.input_data_description_dict, self.target_data_description_dict, self.errors_dict



    def get_hashids(self, linkage_type):
        """
        Gets hashids from workers
        Parameters
        ----------
        None
        """
        hashids_global = None
        self.hashids_dict = {}

        # Send message to every worker
        action = 'GET_HASH_IDS'
        packet = {'action': action, 'to': 'CommonML'}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_GET_HASH_IDS', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        unique_hashids = []
        for waddr in self.workers_addresses:
            unique_hashids = list(set(unique_hashids + self.hashids_dict[waddr]))

        if linkage_type == 'full':
            # Option 1: Filtering out hashids that are not in all workers
            Nhashids = len(unique_hashids)
            hashids_count = {}
            for khash in range(Nhashids):
                hashids_count[unique_hashids[khash]] = 0

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

        self.display(self.name + ': Getting hashids ready')
        return hashids_global



    def linkage_hashids_transform_workers(self, hashids, linkage_type):
        """
        Record linkage at workers
        Parameters
        ----------
        Common Hashids, list of strings
        """

        self.input_data_description_dict = {}
        self.target_data_description_dict = {}

        # Send message to every worker
        action = 'GET_RECORD_LINKAGE'
        data = {'hashids': hashids, 'linkage_type': linkage_type}
        packet = {'action': action, 'to': 'CommonML', 'data': data}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_GET_RECORD_LINKAGE', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        self.display(self.name + ': Record linkage at workers ready')
        return self.input_data_description_dict, self.target_data_description_dict



    def send_preprocess_V(self, prep_model):
        """
        This is the local preprocessing loop, it runs the following actions:
            - It sends the preprocessing object to the workers 
            - It sends instruction to the workers to preprocess the data
        Parameters
        ----------
            None
        """
        self.worker_errors = {}
        self.mean_dict = {}
        self.std_dict = {}

        # Send message to every worker
        action = 'SEND_PREPROCESSOR_V'
        data = {'prep_model': prep_model}
        packet = {'action': action, 'to': 'CommonML', 'data': data}
        self.comms.broadcast(packet, self.workers_addresses)
        self.display(self.name + ': Sent ' + action + ' to all workers')

        # Wait for the reply from every worker
        while not self.checkAllStates('ACK_SEND_PREPROCESSOR_V', self.state_dict):
            self.CheckNewPacket_Master()
        for worker in self.workers_addresses:
            self.state_dict[worker] = ''

        self.display(self.name + ' : Local preprocessing vertical partition ready')
        return self.worker_errors





class Common_to_POMs_123_Worker(Common_to_all_objects):
    """
    This class implements basic methods and protocols common to POMs 1, 2 and 3 for the worker.
    To be inherited by the specific ML models. Not every method is used by every POM.
    """

    def __init__(self):
        """
        Create a :class:`Common_to_POMs_123_Worker` instance.
        Parameters
        ----------
        None
        """
        return



    def run_worker(self):
        """
        This is the training executed at every Worker

        Parameters
        ----------
        None
        """
        self.display(self.name + ' %s: READY and waiting instructions' %(self.worker_address))
        self.terminate = False

        while not self.terminate:
            self.CheckNewPacket_worker()



    def CheckNewPacket_worker(self):
        """
        Checks if there is a new message in the Worker queue

        Parameters
        ----------
        None
        """
        if self.platform == 'pycloudmessenger':
            packet = None
            sender = None
            try:
                packet = self.comms.receive_poms_123(timeout=0.1)
                packet = packet.content
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))

                if packet['to'] == 'Preprocessing':
                    self.ProcessPreprocessingPacket(packet)
                elif packet['to'] == 'CommonML':
                    self.ProcessCommonPacket(packet)
                else: # Message for training the ML model
                    self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception as err:
                if 'pycloudmessenger.ffl.fflapi.TimedOutException' in str(type(err)):
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise
        else: # Local flask
            packet = None
            sender = None
            try:
                packet = self.comms.receive(self.master_address, timeout=0.1)
                sender = 'Master'
                self.display(self.name + ' %s: Received %s from %s' % (self.worker_address, packet['action'], sender))

                if packet['to'] == 'Preprocessing':
                    self.ProcessPreprocessingPacket(packet)
                elif packet['to'] == 'CommonML':
                    self.ProcessCommonPacket(packet)
                else: # Message for training the ML model
                    self.ProcessReceivedPacket_Worker(packet)
            except KeyboardInterrupt:
                self.display(self.name + '%s: Shutdown requested by Keyboard...exiting' %self.worker_address)
                sys.exit()
            except Exception as err:
                if str(err).startswith('Timeout when receiving data'): # TimedOutException
                    pass
                else:
                    self.display(self.name + ': Error %s' %err)
                    raise



    def ProcessCommonPacket(self, packet):
        """
        Take an action after receiving a packet for the preprocessing

        Parameters
        ----------
        packet: Dictionary
            Packet received
        """
        # Exit the process
        if packet['action'] == 'STOP':
            self.display(self.name + ' %s: terminated by Master' %self.worker_address)
            self.terminate = True


        # Store and apply preprocessor
        if packet['action'] == 'SEND_PREPROCESSOR':
            self.display(self.name + ' %s: Receiving preprocessor' %self.worker_address)

            # Retrieve the preprocessing object
            prep_model = packet['data']['prep_model']

            # Apply the received object to Xtr_b and store back the result
            Xtr = self.Xtr_b
            self.display(self.name + ' %s: Shape of original dataset: %s' %(self.worker_address, self.Xtr_b.shape))
            error = 'data not processed'

            try:
                if prep_model.name == 'image_to_vector':
                    X_prep, new_input_format = prep_model.transform(Xtr)
                else:
                    X_prep = prep_model.transform(Xtr)

                self.Xtr_b = X_prep
                self.display(self.name + ' %s: Training set transformed using preprocessor %s' %(self.worker_address, prep_model.name))
                self.display(self.name + ' %s: Shape of transformed dataset: %s' %(self.worker_address, self.Xtr_b.shape))
                error = None
            except Exception as err:
                self.display(self.name + ' %s: ERROR when applying local preprocessing to worker' %self.worker_address)

            # Store the preprocessing object
            self.preprocessors.append(prep_model)
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

            # Send back the ACK
            action = 'ACK_SEND_PREPROCESSOR'
            packet = {'action': action, 'error': error}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Store and apply preprocessor for vertical partition
        if packet['action'] == 'SEND_PREPROCESSOR_V':
            self.display(self.name + ' %s: Receiving preprocessor for vertical partition' %self.worker_address)

            # Retrieve the preprocessing object
            prep_model = packet['data']['prep_model']

            # Apply the received object to Xtr_b and store back the result
            X = np.copy(self.Xtr_b)
            error = 'data not processed'
            # We overwrite with the local data description
            prep_model.input_data_description = self.input_data_description

            data = {}
            if prep_model.name == 'normalization':
                if prep_model.method == 'global_mean_std':
                    which_variables = 'all'
                    self.X_mean = []
                    self.X_std = []
                    for kinput in range(self.input_data_description['NI']):
                        if self.input_data_description['input_types'][kinput]['type']=='num' or (self.input_data_description['input_types'][kinput]['type']=='bin' and which_variables=='all'):
                            aux = np.array(self.Xtr_b)[:, kinput].astype(float)
                            self.X_mean.append(np.nanmean(aux))
                            self.X_std.append(np.nanstd(aux))
                        else:
                            self.X_mean.append(np.NaN)
                            self.X_std.append(np.NaN)

                    self.X_mean = np.array(self.X_mean).reshape(1, -1)
                    self.X_std = np.array(self.X_std).reshape(1, -1)
                    prep_model.mean = self.X_mean
                    prep_model.std = self.X_std
                    data = {'X_mean': self.X_mean, 'X_std': self.X_std}

            try:
                #if prep_model.name in ['data2num', 'normalization', 'logscale', 'feature_extract', 'outlier_clipping', 'deep_learning']: # we replace the original data
                X_prep = prep_model.transform(X)
                self.Xtr_b = np.copy(X_prep)
                self.display(self.name + ' %s: Locally V preprocessing data with %s...' % (str(self.worker_address), prep_model.name))
                self.display(self.name + ' %s: New data shape is %d x %d.' % (str(self.worker_address), X_prep.shape[0], X_prep.shape[1]))
                self.Xtr_orig = X_prep
                error = None
            except Exception as err:
                error = err
                # Comment later
                self.display(self.name + ' %s: ERROR when applying local preprocessing vertical partition to worker' %self.worker_address)

            if prep_model.name == 'missing_data_imputation_V':
                data = {'X_mean': prep_model.means, 'X_std': None}

            # Store the preprocessing object
            self.preprocessors.append(prep_model)
            self.display(self.name + ' %s: Final preprocessor stored' %self.worker_address)

            # Send back the ACK
            action = 'ACK_SEND_PREPROCESSOR_V'
            packet = {'action': action, 'error': error, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Compute statistics
        if packet['action'] == 'GET_STATS':
            self.display(self.name + ' %s: Computing statistics' %self.worker_address)
            stats_list = packet['data']['stats_list']
            stats_dict = self.compute_stats(self.Xtr_b, self.ytr.astype(float), stats_list)

            # Send back the ACK
            action = 'ACK_GET_STATS'
            data = {'stats_dict': stats_dict}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Compute Rxyb, rxyb
        if packet['action'] == 'get_Rxyb_rxyb':
            self.display(self.name + ' %s: Computing Rxyb and rxyb' %self.worker_address)
            Xb = self.add_bias(self.Xtr_b)
            y = self.ytr.astype(float)                
            Rxyb = np.dot(Xb.T, Xb)
            rxyb = np.dot(Xb.T, y)

            # Send back the ACK
            action = 'ACK_get_Rxyb_rxyb'
            data = {'Rxyb': Rxyb, 'rxyb': rxyb}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Get vocabulary
        if packet['action'] == 'GET_VOCABULARY':
            self.display(self.name + ' %s: Computing vocabulary' %self.worker_address)
            vocab_dict = {}
            X = self.Xtr_b
            Ndocs = X.shape[0]

            for kdoc in range(Ndocs):
                bow_dict = X[kdoc, 0]
                vocab_dict = dict(Counter(vocab_dict) + Counter(bow_dict))

            # Send back the ACK
            action = 'ACK_GET_VOCABULARY'
            data = {'vocab_dict': vocab_dict}
            packet = {'action': action, 'data':data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Get DF
        if packet['action'] == 'GET_DF':
            self.display(self.name + ' %s: Computing df' %self.worker_address)

            # Retrieve the vocabulary
            vocab = packet['data']['vocab']
            X = self.Xtr_b
            Ndocs = X.shape[0]
            df_dict = {}

            for kdoc in range(Ndocs):
                bow_dict = X[kdoc, 0]
                tmp_dict = {}
                for key in bow_dict:
                    if key in vocab:
                        tmp_dict[key] = 1

                df_dict = dict(Counter(df_dict) + Counter(tmp_dict))

            # Send back the ACK
            action = 'ACK_GET_DF'
            data = {'Ndocs': Ndocs, 'df_dict': df_dict}
            packet = {'action': action, 'data':data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Get feature count
        if packet['action'] == 'GET_FEAT_COUNT':
            self.display(self.name + ' %s: Getting feature count' %self.worker_address)
            X = self.Xtr_b
            count = np.array(np.sum((X > 0).astype(float), axis=0)).ravel()
            NP = X.shape[0]

            # Send back the ACK
            action = 'ACK_GET_FEAT_COUNT'
            data = {'NP': NP, 'count': count}
            packet = {'action': action, 'data': data}
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Check data
        if packet['action'] == 'CHECK_DATA':
            self.display(self.name + ' %s: Checking data' %self.worker_address)
            input_data_description = packet['data']['input_data_description']
            target_data_description = packet['data']['target_data_description']
                    
            error = ''
            NI = self.Xtr_b.shape[1]
            NT = self.ytr.reshape(-1, 1).shape[1]

            if NI != input_data_description['NI']:
                error += 'Incorrect number of features; '
            elif NT != target_data_description['NT']:
                error += 'Incorrect number of targets; '
            elif self.Xtr_b.shape[0] != self.ytr.shape[0]:
                error += 'Different number of inputs and targets; '
            else:
                try:
                    # Checking inputs
                    for k in range(NI):
                        x = self.Xtr_b[:, k]
                        xtype = input_data_description['input_types'][k]['type']
                        if xtype == 'num':
                            try:
                                x = x.astype(float)
                            except:
                                error += 'Input feature No. %d is not numeric; ' % k
                        if xtype == 'cat':
                            try:
                                x = set(x.astype(str))
                                x = list(x - set(input_data_description['input_types'][k]['values']))
                                if len(x) > 0:
                                    error += 'Input feature No. %d has an unrecognized categorical value; ' % k
                            except:
                                error += 'Input feature No. %d is not categorical; ' % k
                        if xtype == 'bin':
                            try:
                                x = x.astype(float)
                                x = list(set(x) - set([0.0, 1.0]))
                                if len(x) > 0:
                                    error += 'Input feature No. %d is not binary; ' % k
                            except:
                                error += 'Input feature No. %d is not binary; ' % k

                    # Checking targets
                    for k in range(NT):
                        x = self.ytr[:, k]
                        xtype = target_data_description['output_types'][k]['type']
                        if xtype == 'num':
                            try:
                                x = x.astype(float)
                            except:
                                error += 'Target No. %d is not numeric; ' % k
                        if xtype == 'cat':
                            try:
                                x = set(x.astype(str))
                                x = list(x - set(input_data_description['input_types'][k]['values']))
                                if len(x) > 0:
                                    error += 'Target No. %d has an unrecognized categorical value; ' % k
                            except:
                                error += 'Target No. %d is not categorical; ' % k
                        if xtype == 'bin':
                            try:
                                x = x.astype(float)
                                x = list(set(x) - set([0.0, 1.0]))
                                if len(x) > 0:
                                    error += 'Target No. %d is not binary; ' % k
                            except:
                                error += 'Target No. %d is not binary; ' % k
                except:
                    error += 'Unexpected error when processing features; '

            # Send back the ACK
            action = 'ACK_CHECK_DATA'
            packet = {'action': action, 'error': error}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Get hash ids
        if packet['action'] == 'DATA2NUM_V':
            self.display(self.name + ' %s: Transforming data to numerical in vertical partition' %self.worker_address)
            error = None
            try:
                model = self.get_data2num_model(self.input_data_description)
                Xtr = np.copy(self.Xtr_b)
                Xtr_transf = model.transform(Xtr)
                self.Xtr_b = np.copy(Xtr_transf)
                self.display(self.name + ' %s: Transformed data to numeric, with shape %dx%d' %(str(self.worker_address), self.Xtr_b.shape[0], self.Xtr_b.shape[1]))
                self.input_data_description = model.new_input_data_description
            except Exception as err:
                error = str(err)

            # Send back the ACK
            action = 'ACK_DATA2NUM_V'
            data = {'error': error, 'input_data_description': self.input_data_description, 'target_data_description': self.target_data_description}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Get hash ids
        if packet['action'] == 'GET_HASH_IDS':
            self.display(self.name + ' %s: Getting hash ids' %self.worker_address)
            X = self.Xtr_b
            ids = list(X[:,0])
            self.hashids = [self.hash_sha256(id) for id in ids]

            # Send back the ACK
            action = 'ACK_GET_HASH_IDS'
            data = {'hashids': self.hashids}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))


        # Get record linkage
        if packet['action'] == 'GET_RECORD_LINKAGE':
            self.display(self.name + ' %s: Getting hash ids' %self.worker_address)

            # Retrieve the hash ids
            hashids = packet['data']['hashids']
            Nhashids = len(hashids)

            X = self.Xtr_b
            y = self.ytr
            NF = X.shape[1] - 1  # id is no longer stored

            newX = np.empty((Nhashids, NF), dtype=np.dtype('U100'))
            # warning, binary
            newy = np.empty((Nhashids, 1), dtype=np.dtype('U100'))

            for k, hashid in enumerate(hashids):
                try:
                    pos = self.hashids.index(hashid)
                    newX[k, :] = X[pos, :][1:]
                    newy[k, :] = y[pos, :]
                except:
                    pass

            self.Xtr_b = newX
            self.ytr = newy
            self.display(self.name + ' %s: Completed record linkage, input data has shape %dx%d' %(str(self.worker_address), newX.shape[0], newX.shape[1]))

            # Send back the ACK
            action = 'ACK_GET_RECORD_LINKAGE'
            data = {'input_data_description': self.input_data_description, 'target_data_description': self.target_data_description}
            packet = {'action': action, 'data': data}            
            self.comms.send(packet, self.master_address)
            self.display(self.name + ' %s: Sent %s to master' %(self.worker_address, action))



