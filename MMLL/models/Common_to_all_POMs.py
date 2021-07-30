# -*- coding: utf-8 -*-
'''
Collection of methods common to all POMs. To be inherited by the ML classes'''

__author__ = "Angel Navia-Vázquez"
__date__ = "Mar 2020"


#import requests
import json
#import base64
import numpy as np
#import dill
from MMLL.Common_to_all_objects import Common_to_all_objects
import pickle
from pympler import asizeof #asizeof.asizeof(my_object)
import dill
import time

class Common_to_all_POMs(Common_to_all_objects):
    """
    This class implements basic methods and protocols common to all POMs.
    To be inherited by the specific ML models. Not every method is used by every POM.
    """

    def __init__(self):
        """
        Create a :class:`Common_to_all_POMs` instance.

        Parameters
        ----------
        None

        """
        self.message_counter = 1    # used to number the messages
        return

    def dill_it(self, x):
        return dill.dumps(x) 

    def undill_it(self, x):
        return dill.loads(x) 

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

    def cross_entropy(self, o, y, epsilon):
        """
        Computes the Cross-Entropy cost function

        Parameters
        ----------
        o: ndarray float
            estimated values

        y: ndarray float
            target values

        epsilon: float
            very small value to avoid numerical errors

        Returns
        -------
        cross_entropy: ndarray float

        """
        o = o * (1.0 - 2.0 * epsilon) + epsilon
        ce = - np.multiply(y.ravel(), np.log(o)) - np.multiply((1 - y.ravel()), np.log(1.0 - o))
        return ce

    def run_Master(self):
        """
        This is the Finite State Machine loop to be run at Masternode, it runs the following actions until the stop condition is met:
            - Process the received packets
            - Perform actions according to the state
            - Update the execution state

        Parameters
        ----------
            None
        """
        while self.FSMmaster.state != 'waiting_order':
            packet = None
            sender = None
            packet, sender = self.CheckNewPacket_master()
            #self.Update_State_Master()
            #self.ProcessReceivedPacket_Master(packet, sender)

    def serialize(self, x):
        """
        Serialize a given object.
        
        Parameters
        ----------
        x: arbitrary (typically a dictionary)
            Object to be serialized

        Returns
        -------
        Serialized object: string

        """
        return base64.b64encode(pickle.dumps(x)).decode('utf-8')

    def unserialize(self, x):
        """
        Unserialize a serialized object.
 
         Parameters
        ----------
        x: arbitrary (string)
            Object to be unserialized

        Returns
        -------
        Unserialized object: typically a dictionary
 
        """  
        return pickle.loads(base64.b64decode(x.encode()))


    def run_crypto(self):
        """
        This is the Finite State Machine loop to be run at Cryptonode, it runs the following actions until the stop condition is met:
            - Process the received packets
            - Perform actions according to the state
            - Update the execution state

        Parameters
        ----------
            None
        """

        self.display('Common Crypto is READY and waiting instructions.')
        self.terminate = False
        while not self.terminate:
            packet, sender = self.CheckNewPacket_crypto()

    def chekAllStates(self, condition):
        """
        Checks if all states are equal to the condition

         Parameters
        ----------
        condition: string (state name)
            State to be checked

        """
        try:
            all_active = True
            for waddr in self.workers_addresses:
                if self.state_dict[waddr] != condition:
                    all_active = False

            # Resetting states after checking
            if all_active:
                for waddr in self.workers_addresses:
                    self.state_dict[waddr] = ''
        except Exception as err:
            #print('STOP AT chekAllStates CommontoallPOMS OK')
            #import code
            #code.interact(local=locals())
            raise

        return all_active


    def CheckNewPacket_master(self):
        """
        Checks if there is a new message in the Master queue. If so, process it. This operation is highly dependant on the communications library.

        Parameters
        ----------
            None
        """
        packet = None
        active_sender = None

        if self.comms.name == 'localflask':
            addresses2check = self.workers_addresses.copy()
            try:
                if self.cryptonode_address is not None:
                    addresses2check += [self.cryptonode_address]
            except:
                pass

            try:
                addresses2check = list(set( addresses2check + self.broadcast_addresses))
            except:
                #print('STOP AT addresses2check' )
                #import code
                #code.interact(local=locals())
                #pass
                raise

            #print("Listening: ", addresses2check)

            for sender in addresses2check:
                try:
                    packet = self.comms.receive(sender, timeout=0.01)
                    self.display(self.name + ' %s received %s from %s' % (self.master_address, packet['action'], sender))
                    active_sender = sender
                    self.ProcessReceivedPacket_Master(packet, sender)
                    self.Update_State_Master()
                    packet = None
                except Exception as err:
                    if 'Timeout when receiving' not in str(err):
                        if 'NewConnectionError' in str(err):
                            self.display('\n*************************** ERROR ***************************')
                            self.display('The Flask Server is not running!')
                            self.display("Execute 'python local_flask_server.py' in a separate console")
                            self.display('************************************************************')
                            exit()
                        else:
                            #self.display('ERROR AT CheckNewPacket_master: %s' % str(err))
                            #import code
                            #code.interact(local=locals())
                            raise

        if self.comms.name == 'pycloudmessenger':

            try:
                self.comms.pycloudmessenger_timeout_POMs456 > 0
            except:
                self.comms.pycloudmessenger_timeout_POMs456 = 0.1

            try:
                packet = None
                packet = self.comms.receive(self.comms.pycloudmessenger_timeout_POMs456)
                if packet is not None:
                    sender = self.receive_from[packet['sender']]
                    active_sender = sender
                    self.ProcessReceivedPacket_Master(packet, sender)
                    self.Update_State_Master()
                    #print(packet)
                    # ANV Check this:
                    #packet = None
            except Exception as err:
                if 'Operation timeout reached' not in str(err):
                    if 'Operation timed out' not in str(err):
                        #self.display('Unhandled ERROR AT CheckNewPacket_master: %s' % str(err))
                        raise
                        #print('*** ERROR AT CheckNewPacket_master --------------')
                        #import code
                        #code.interact(local=locals())
                pass

        return packet, active_sender

    def CheckNewPacket_worker(self):
        """
        Checks if there is a new message in the Worker queue. If so, process it. This operation is highly dependant on the communications library.

        Parameters
        ----------
            None
        """
        packet = None
        sender = None

        if self.comms.name == 'localflask':
            try:
                # The worker can only know the address of the master
                #addresses2check = self.workers_addresses + [self.master_address]
                # Warning, roundrobin with the local flask is deactivated...

                addresses2check = []
                try:
                    addresses2check.append(self.cryptonode_address)
                except:
                    pass
                try:
                    addresses2check.append(self.master_address)
                except:
                    pass

                for sender in addresses2check:
                    try:
                        packet = self.comms.receive(sender, timeout=0.01)
                        self.display(self.name + ' %s received %s from %s' % (self.worker_address, packet['action'], sender))
                        #if str(self.worker_address) == '1':
                        #    print('STOP AT CheckNewPacket_worker_Common')
                        #    import code
                        #    code.interact(local=locals())
                        ## WARNING ProcessReceivedPacket_Worker is executed elsewhere
                        #self.ProcessReceivedPacket_Worker(packet, sender)
                    except Exception as err:
                        if 'Timeout when receiving' not in str(err):
                            if 'NewConnectionError' in str(err):
                                self.display('\n*************************** ERROR ***************************')
                                self.display('The Flask Server is not running!')
                                self.display("Execute 'python local_flask_server.py' in a separate console")
                                self.display('************************************************************')
                                exit()
                            else:
                                self.display('ERROR AT CheckNewPacket_worker: %s' % str(err))
                                #import code
                                #code.interact(local=locals())
            except Exception as err2:
                #self.display('ERROR AT CheckNewPacket_worker: %s' % str(err2))
                #import code
                #code.interact(local=locals())
                raise

        if self.comms.name == 'pycloudmessenger':
            try:
                self.comms.pycloudmessenger_timeout_POMs456 > 0
            except:
                self.comms.pycloudmessenger_timeout_POMs456 = 0.1

            try:
                packet = self.comms.receive(self.comms.pycloudmessenger_timeout_POMs456)
                if packet is not None:
                    sender = packet['sender']
                    self.display(self.name + ' %s received %s from %s through pycloudmessenger' % (self.worker_address, packet['action'], sender))
            except Exception as err:
                if 'Operation timeout reached' not in str(err):
                    if 'Operation timed out' not in str(err):
                        #print('ERROR AT CheckNewPacket_worker')
                        #self.display('Unhandled ERROR AT CheckNewPacket_worker: %s' % str(err))
                        #import code
                        #code.interact(local=locals())
                        raise
                pass

        return packet, sender

    def CheckNewPacket_crypto(self):
        """
        Checks if there is a new message in the Crypto queue. If so, process it. This operation is highly dependant on the communications library.

        Parameters
        ----------
            None
        """
        packet = None
        sender = None

        if self.comms.name == 'localflask':
            try:
                addresses2check = [self.master_address]
                for sender in addresses2check:
                    try:
                        packet = self.comms.receive(sender, timeout=0.01)
                        self.display(self.name + ' %s received %s from %s' % (self.cryptonode_address, packet['action'], sender))
                        #if str(self.worker_address) == '1':
                        #    print('STOP AT CheckNewPacket_worker_Common')
                        #    import code
                        #    code.interact(local=locals())

                        self.ProcessReceivedPacket_Crypto(packet, sender)
                    except Exception as err:
                        if 'Timeout when receiving' not in str(err):
                            #self.display('ERROR AT CheckNewPacket_crypto: %s' % str(err))
                            #import code
                            #code.interact(local=locals())
                            raise
            except Exception as err2:
                #self.display('ERROR AT CheckNewPacket_crypto: %s' % str(err2))
                #import code
                #code.interact(local=locals())
                raise
        if self.comms.name == 'pycloudmessenger':
            try:
                self.comms.pycloudmessenger_timeout_POMs456 > 0
            except:
                self.comms.pycloudmessenger_timeout_POMs456 = 0.1

            packet = None
            sender = None
            try:
                packet = self.comms.receive(self.comms.pycloudmessenger_timeout_POMs456)
                if packet is not None: 
                    sender = packet['sender']
                    self.display(self.name + ' %s received %s from %s through pycloudmessenger' % (self.cryptonode_address, packet['action'], sender))
                    self.ProcessReceivedPacket_Crypto(packet, sender)
            except Exception as err:
                if 'Operation timeout reached' not in str(err):
                    if 'Operation timed out' not in str(err):
                        #print('ERROR AT CheckNewPacket_crypto')
                        #self.display('Unhandled ERROR AT CheckNewPacket_crypto: %s' % str(err))
                        #import code
                        #code.interact(local=locals())
                        raise
                pass

        return packet, sender


    def terminate_workers_(self, workers_addresses_terminate=None):
        """
        Send order to terminate Workers

        Parameters
        ----------
        workers_addresses_terminate: list of strings
            addresses of the workers to be terminated

        """
        action = 'STOP'
        packet = {'action': action, 'to': 'CommonML', 'sender': self.master_address}

        message_id = self.master_address+'_'+str(self.message_counter)
        packet.update({'message_id': message_id})
        self.message_counter += 1
        size_bytes = asizeof.asizeof(dill.dumps(packet))
        #self.display('COMMS_MASTER_BROADCAST %s, id = %s, bytes=%s' % (action, message_id, str(size_bytes)), verbose=False)

        if workers_addresses_terminate is None:  # We terminate all of them
            workers_addresses_terminate = self.workers_addresses
            self.display(self.name + ' sent STOP to all Workers')
            self.comms.broadcast(packet)
        else:
            self.display(self.name + ' sent STOP to %d Workers' % len(workers_addresses_terminate))
            for worker in workers_addresses_terminate:
                self.comms.send(packet, self.send_to[worker])

        # Updating the list of active users
        self.workers_addresses = list(set(self.workers_addresses) - set(workers_addresses_terminate))
        self.Nworkers = len(self.workers_addresses)

        self.FSMmaster.go_Exit(self)
        self.FSMmaster.go_waiting_order(self)

    def stop_workers_(self):
        """
        Send order to stop Workers

        Parameters
        ----------
        None
        """
        message_id = self.master_address + str(self.message_counter)
        self.message_counter += 1            
        packet = {'action': 'STOP_NOT_CLOSE_CONNECTION', 'to': 'CommonML', 'sender': self.master_address, 'message_id': message_id}
        self.display(self.name + ' sent STOP_NOT_CLOSE_CONNECTION to all Workers')
        self.comms.broadcast(packet)

        self.FSMmaster.go_Exit(self)
        self.FSMmaster.go_waiting_order(self)



    def send_check(self, input_data_description, target_data_description):
        """
        Parameters
        ----------
            None
        """
        self.worker_errors_dict = {}
        self.FSMmaster.go_sending_check_data(self, input_data_description, target_data_description)
        self.display(self.name + ' : Sending check data')
        self.run_Master()
        self.display(self.name + ' : Checking data at workers is done')
        return self.worker_errors_dict

    def get_X_minus_mean_squared(self, mean_values=None, input_data_description=None, which_variables='num'):
        """
        Gets from workers their sum of input data multiplied by the targets

        Parameters
        ----------
        None
        """

        self.display(self.name + ': Asking workers their sum X-mean squared')

        if self.aggregation_type == 'direct':
            self.sumXminusmeansquared_dict = {}
            self.FSMmaster.go_getting_X_minus_mean_squared(self, mean_values, input_data_description, which_variables)
            self.run_Master()

            self.total_X_minus_mean_squared = np.zeros((1, self.NI))
            for waddr in self.workers_addresses:
                self.total_X_minus_mean_squared += self.sumXminusmeansquared_dict[waddr]

        if self.aggregation_type == 'roundrobin':
            self.display(self.name + ': Getting sum X-mean squared with roundrobin')

            # pending generate random numbers here...
            #self.x2_ini = 999999999 + np.zeros((1, self.NI))
            self.x2_ini = np.random.uniform(-9e5, 9e5, (1, self.NI))

            self.FSMmaster.go_getting_X_minus_mean_squared_roundrobin(self, mean_values, input_data_description, self.x2_ini, which_variables)
            self.run_Master()

            self.total_X_minus_mean_squared = self.sumX2_roundrobin - self.x2_ini

        self.display(self.name + ': getting X_minus_mean_squared is done')
        return self.total_X_minus_mean_squared

    def get_min_max_X(self, input_data_description):
        """
        Gets from workers their min and max of input data

        Parameters
        ----------
        None
        """
        self.input_data_description = input_data_description
        self.NI = self.input_data_description['NI']

        self.minX_dict = {}
        self.maxX_dict = {}

        # roundrobin not usable here
        if self.aggregation_type == 'direct' or self.aggregation_type == 'roundrobin':
            self.display(self.name + ': Asking workers their minX')
            self.FSMmaster.go_getting_minX(self, input_data_description)
            self.run_Master()

            self.total_minX = 9e20 * np.ones((1, self.NI))
            self.total_maxX = -9e20 * np.ones((1, self.NI))
            self.total_NP = 0
            for waddr in self.workers_addresses:
                self.total_minX = np.vstack((self.total_minX, self.minX_dict[waddr]))
                self.total_maxX = np.vstack((self.total_maxX, self.maxX_dict[waddr]))
                self.total_NP += self.NP_dict[waddr]

            self.total_minX = np.min(self.total_minX, axis=0)
            self.total_maxX = np.max(self.total_maxX, axis=0)

        '''
        if self.aggregation_type == 'roundrobin':
            self.display(self.name + ': Getting sumX with roundrobin')
            # pending generate random numbers here...
            #self.x_ini = 999999999 + np.zeros((1, self.NI))
            #self.NP_ini = 999999999
            self.FSMmaster.go_getting_sumX_roundrobin(self, input_data_description, self.x_ini, self.NP_ini)
            self.run_Master()
            self.total_sumX = self.sumX_roundrobin - self.x_ini
            self.total_NP =  self.NP_roundrobin - self.NP_ini   
        '''
        self.display(self.name + ': getting minX, maxX is done')

    def get_Rxyb_rxyb_direct(self):
        """
        Obtaining get_Rxyb_rxyb from workers, direct method

        Parameters
        ----------
            None
        """

        self.Rxyb_dict = {}
        self.rxyb_dict = {}

        self.display(self.name + ': Asking workers to compute Rxyb_rxyb, direct transmission')
        self.FSMmaster.go_getting_Rxyb_rxyb_direct(self)
        self.run_Master()

        workers = list(self.Rxyb_dict.keys())
        Rxy_b = self.Rxyb_dict[workers[0]]
        rxy_b = self.rxyb_dict[workers[0]]
        for worker in workers[1:]:
            Rxy_b += self.Rxyb_dict[worker]
            rxy_b += self.rxyb_dict[worker]

        self.display(self.name + ': compute Rxyb_rxyb is done')
        return Rxy_b, rxy_b

    def get_Rxyb_rxyb_roundrobin(self, Rxyb_ini, rxyb_ini):
        """
        Obtaining get_Rxyb_rxyb from workers, roundrobin method

        Parameters
        ----------
            None
        """
        self.display(self.name + ': Asking workers to compute Rxyb_rxyb, roundrobin aggregation')
        self.FSMmaster.go_getting_Rxyb_rxyb_roundrobin(self, Rxyb_ini, rxyb_ini)
        self.run_Master()

        Rxy_b = self.Rxyb_roundrobin - Rxyb_ini
        rxy_b = self.rxyb_roundrobin - rxyb_ini
        self.display(self.name + ': compute Rxyb_rxyb, roundrobin is done')
        return Rxy_b, rxy_b

    def get_Npc(self):
        """
        Obtain the number of patterns per class, to balance uneven pattern distribution among classes

        Parameters
        ----------
        None
        """
        self.display(self.name + ': Asking workers their Npc')
        self.FSMmaster.go_getting_Npc(self)
        self.run_Master()

        self.aggregated_Npc_dict = {} # Number of aggregated patterns per class
        for cla in self.classes:
            count = 0
            for wa in self.workers_addresses:
                count += self.Npc_dict[wa][cla]
            self.aggregated_Npc_dict.update({cla: count})   

        self.display(self.name + ': getting Npc is done')

    def get_sumXy(self):
        """
        Gets from workers their sum of input data multiplied by the targets

        Parameters
        ----------
        None
        """
        self.sumy_dict = {}
        self.sumX_dict = {}
        self.NP_dict = {}

        self.display(self.name + ': Asking workers their sumX sumy')
        self.FSMmaster.go_getting_sumXy(self)
        self.run_Master()

        self.total_sumX = np.zeros((1, self.NI))
        self.total_sumy = 0
        self.total_NP = 0
        for waddr in self.workers_addresses:
            self.total_sumX += self.sumX_dict[waddr]
            self.total_sumy += self.sumy_dict[waddr]
            self.total_NP += self.NP_dict[waddr]

        self.display(self.name + ': getting sumXy is done')

    def get_sumX(self, input_data_description, which_variables='num'):
        """
        Gets from workers their sum of input data

        Parameters
        ----------
        input_data_description
        which_variables : when to compute the mean, 'num' = only numerical, 'all' = numerical + binary
        None
        """
        self.input_data_description = input_data_description
        self.NI = self.input_data_description['NI']
        self.sumX_dict = {}

        if self.aggregation_type == 'direct':
            self.display(self.name + ': Asking workers their sumX')
            self.FSMmaster.go_getting_sumX(self, input_data_description, which_variables)
            self.run_Master()

            self.total_sumX = np.zeros((1, self.NI))
            self.total_NP = 0
            for waddr in self.workers_addresses:
                self.total_sumX += self.sumX_dict[waddr]
                self.total_NP += self.NP_dict[waddr]

        if self.aggregation_type == 'roundrobin':
            self.display(self.name + ': Getting sumX with roundrobin')
            # pending generate random numbers here...
            self.x_ini = np.random.uniform(-9e5, 9e5, (1, self.NI))
            self.NP_ini = np.random.uniform(-9e5, 9e5)
            self.FSMmaster.go_getting_sumX_roundrobin(self, input_data_description, self.x_ini, self.NP_ini, which_variables)
            self.run_Master()
            self.total_sumX = self.sumX_roundrobin - self.x_ini
            self.total_NP =  self.NP_roundrobin - self.NP_ini   

        self.display(self.name + ': getting sumX is done')

    def send_preprocess(self, prep_model):
        """
        This is the local preprocessing loop, it runs the following actions:
            - It sends the preprocessing object to the workers 
            - It sends instruction to the workers to preprocess the data

        Parameters
        ----------
            None
        """
        self.worker_errors = {}
        self.FSMmaster.go_sending_prep_object(self, prep_model)
        self.display(self.name + ' : Sending Preprocessing object')
        self.run_Master()
        self.display(self.name + ' : Local Preprocessing is done')
        return self.worker_errors

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
        self.FSMmaster.go_sending_prep_object_V(self, prep_model)
        self.display(self.name + ' : Sending Preprocessing object V' )
        self.run_Master()
        self.display(self.name + ' : Local Preprocessing V is done ')
        return self.worker_errors

    '''
    def send_check(self, input_data_description, target_data_description):
        """
        Parameters
        ----------
            None
        """
        self.worker_errors_dict = {}
        self.FSMmaster.go_sending_check_data(self, input_data_description, target_data_description)
        self.display(self.name + ' : Sending check data')
        self.run_Master()
        self.display(self.name + ' : Checking data at workers is done')
        return self.worker_errors_dict
    '''

    def local_prep_Master(self, prep_object):
        """
        This is the local preprocessing loop, it runs the following actions:
            - It sends the preprocessing object to the workers 
            - It sends instruction to the workers to preprocess the data

        Parameters
        ----------
            None
        """
        self.prep = prep_object
        self.FSMmaster.go_sending_prep_object(self)
        self.display(self.name + ' : Sending Preprocessing object')
        self.run_Master()
        self.display(self.name + ' : Local Preprocessing is done')

    def get_stats(self, stats_list):
        """
        Gets from workers their stats

        Parameters
        ----------
        stats_list: list of stats to be computed
        """
        self.stats_dict = {}
        self.display(self.name + ': Asking workers their stats')
        self.FSMmaster.go_getting_stats(self, stats_list)
        self.run_Master()

        self.display(self.name + ': getting stats is done')


    def crypto_mult_X(self, B=None):
        """
        Multiplies any matrix/vector encrypted with the previously stored at crypto X encr. This is only to be executed by some POM4 algorithms.

        Parameters
        ----------
        B: matrix, vector, or dictionary of matrix/vector of encrypted numbers
            matrix/vector to be multiplied by X. When B is None, X squared is returned

        Returns
        -------
        Multiplication result: matrix, vector, or dictionary of matrix/vector

        """
        try:
            # Checking if empty

            # Checking if dictionary
            is_empty = B is None

            if is_empty:
                B_bl_send = None
            else:
                is_dictionary = type(B) is dict
                if not is_dictionary:

                    # Checking size
                    try:
                        scalar_value = False
                        M, N = B.shape
                    except:  # scalar value
                        scalar_value = True


                    # Blinding, keep Rbl_B for later deblinding...
                    if not scalar_value:
                        # wT X
                        Rbl_B = np.random.normal(0, 2, (M, N))
                        try:
                            B_bl = B + Rbl_B
                        except:
                            print('ERROR AT crypto_mult_X: overflow???')
                            #import code
                            #code.interact(local=locals())                       
                            #pass
                    else:
                        # PENDING
                        print('ERROR AT crypto_mult_X: escalar value pending')
                        #import code
                        #code.interact(local=locals())

                    B_bl_send = B_bl

                if is_dictionary:
                    B_bl_dict = {}
                    Rbl_B_dict = {}
                    keys = list(B.keys())

                    for key in keys:
                        B_ = B[key]
                        # Checking size
                        try:
                            scalar_value = False
                            M, N = B_.shape
                        except:  # scalar value
                            scalar_value = True

                        # Blinding, keep Rbl_B for later deblinding...
                        if not scalar_value:
                            Rbl_B = np.random.normal(0, 2, (M, N))
                            try: ### overflow for small key_size...                                        
                                B_bl = B_ + Rbl_B
                            except:
                                print('STOP AT common xxx, overflow???')
                                #import code
                                #code.interact(local=locals())
                                #pass

                            '''
                                #aux = self.encrypter.encrypt(B_bl)
                                aux = self.decrypter.decrypt(B_bl)
                                print('Not overflow')
                            '''
                            B_bl_dict.update({key: B_bl})
                            Rbl_B_dict.update({key: Rbl_B})
                        else:
                            # PENDING
                            print('ERROR AT crypto_mult_X, dictionary: escalar value pending')
                            #import code
                            #code.interact(local=locals())
                    B_bl_send = B_bl_dict

            self.FSMmaster.go_mult_XB(self, B_bl_send)
            self.run_Master()
            # returns self.XB_bl_encr_dict

            # deblinding results, data:
            # self.XB_bl_encr_dict  -> X * B , blinded
            # Rbl_B_dict --> blinding values for B
            # self.BX_dict --> blinding values for X
            # self.X_encr_dict --> encrypted values for X
            # B  --> encrypted values for B

            XB_encr_dict = {} # stores the multiplication without blinding

            # Warning, the operation can be done on a selection of workers
            #for waddr in self.workers_addresses:
            keys = list(self.XB_bl_encr_dict.keys())
            for waddr in keys:
                if is_empty:

                    X_encr = self.X_encr_dict[waddr]
                    B_encr = X_encr

                    Rx = self.BX_dict[waddr]
                    Rb = Rx

                    aux = self.XB_bl_encr_dict[waddr]
                    aux = aux - X_encr * Rb
                    aux = aux - Rx * B_encr
                    aux = aux - Rx * Rb
                    XB_encr_dict.update({waddr: aux})

                else:
                    if not is_dictionary: #check
                        X_encr = self.X_encr_dict[waddr]
                        MQ, NQ = Rbl_B.shape
                        B_encr = B
                        Rx = self.BX_dict[waddr]
                        Rb = Rbl_B
                        aux = self.XB_bl_encr_dict[waddr]
                        aux = aux - Rb * X_encr
                        aux = aux - B_encr * Rx 
                        aux = aux - Rb * Rx
                        XB_encr_dict.update({waddr: aux})

                    if is_dictionary:  # the values in B change                  
                        X_encr = self.X_encr_dict[waddr]
                        B_encr = B[waddr]
                        Rx = self.BX_dict[waddr]
                        Rb = Rbl_B_dict[waddr]
                        aux = self.XB_bl_encr_dict[waddr]
                        aux = aux - X_encr * Rb
                        aux = aux - Rx * B_encr
                        aux = aux - Rx * Rb
                        XB_encr_dict.update({waddr: aux})
        except:
            #print('ERROR AT crypto_mult_X: general')
            #import code
            #code.interact(local=locals())
            raise

        return XB_encr_dict



    def crypto_mult_XM(self, B=None):
        """
        Multiplies a dictionary of vectors with the previously stored at crypto X encr. This is only to be executed by some POM4 algorithms.

        Parameters
        ----------
        B: matrix, vector, or dictionary of matrix/vector of encrypted numbers
            matrix/vector to be multiplied by X. When B is None, X squared is returned

        Returns
        -------
        Multiplication result: matrix, vector, or dictionary of matrix/vector

        """
        try:
            # B must be a double dictionary

            self.B_bl_dict = {}  # All Blinded values
            self.Rbl_B_dict = {}  # All Blinding values
            
            for waddr in self.workers_addresses:
                B_bl_dict = {}  # Blinded values
                Rbl_B_dict = {}  # Blinding values
                for cla in self.classes:
                    M, N = B[waddr][cla].shape
                    Rbl_B = np.random.normal(0, 2, (M, N))
                    B_bl = B[waddr][cla] + Rbl_B
                    B_bl_dict.update({cla: B_bl})
                    Rbl_B_dict.update({cla: Rbl_B})

                self.B_bl_dict.update({waddr: B_bl_dict})
                self.Rbl_B_dict.update({waddr: Rbl_B_dict})

            self.FSMmaster.go_mult_XBM(self, self.B_bl_dict)
            self.run_Master()
            # returns self.XB_bl_encr_dict

            # deblinding results, data:
            # self.XB_bl_encr_dict  -> X * B , blinded
            # self.Rbl_B_dict --> blinding values for B
            # self.BX_dict --> blinding values for X
            # self.X_encr_dict --> encrypted values for X
            # B  --> encrypted values for B

            self.XB_encr_dict = {} # stores the multiplication without blinding
            for waddr in self.workers_addresses:
                XB_encr_dict = {} # stores the multiplication without blinding
                X_encr = self.X_encr_dict[waddr]
                Rx = self.BX_dict[waddr]
                for cla in self.classes:
                    B_encr = B[waddr][cla]
                    Rb = self.Rbl_B_dict[waddr][cla]

                    aux = self.XB_bl_encr_dict[waddr][cla]
                    aux = aux - X_encr * Rb
                    aux = aux - Rx * B_encr
                    aux = aux - Rx * Rb
                    XB_encr_dict.update({cla: aux})

                self.XB_encr_dict.update({waddr: XB_encr_dict})
        except:
            print('ERROR AT Common all crypto_mult_XM: general')
            #import code
            #code.interact(local=locals())
            raise

        return self.XB_encr_dict


    def decrypt_model(self, model_encr):
        self.FSMmaster.go_decrypt_model(self, model_encr)
        self.run_Master()

        #self.model_decr_bl
        # deblinding

        #print('STOP AT deblinding')
        #import code
        #code.interact(local=locals())

        #print('COMMON decrypt_model')

        try:
            self.model_decr = {}
            for key in list(self.model_decr_bl.keys()):
                x_decr_bl = self.model_decr_bl[key]
                #print('COMMON decrypt_model BL:')
                #print(self.bl[key])
                #print(x_decr_bl)
                x = x_decr_bl - self.bl[key]
                #print(x)
                self.model_decr.update({key: x})
        except:
            #print('ERROR AT COMMON decrypt_model')
            #import code
            #code.interact(local=locals())
            #pass
            raise

        return self.model_decr

    def decrypt_modelM(self, model_encr_dict):
        self.FSMmaster.go_decrypt_modelM(self, model_encr_dict)
        self.run_Master()

        #self.model_decr_bl_dict
        # deblinding

        try:
            self.model_decr_dict = {}
            for key in list(self.model_decr_bl_dict.keys()):
                if key=='wM':
                    cla_tmp = {}
                    for cla in list(self.model_decr_bl_dict[key].keys()):
                        x_decr_bl = self.model_decr_bl_dict[key][cla]
                        x = x_decr_bl - self.bl_dict[key][cla]
                        cla_tmp.update({cla: x})   

                self.model_decr_dict.update({key: cla_tmp})

        except:
            #print('ERROR AT COMMON decrypt_model')
            #import code
            #code.interact(local=locals())
            #pass
            raise
            
        return self.model_decr_dict

