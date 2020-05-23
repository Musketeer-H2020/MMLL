# -*- coding: utf-8 -*-
'''
Task managing utilities
@author:  Angel Navia Vázquez
'''
__author__ = "Angel Navia Vázquez, UC3M."

import random, string
import time
try:
    import pycloudmessenger.ffl.abstractions as ffl
    import pycloudmessenger.ffl.fflapi as fflapi
    import pycloudmessenger.serializer as serializer
except:
    print("pycloudmessenger is not installed, use:")
    print("pip install https://github.com/IBM/pycloudmessenger/archive/v0.3.0.tar.gz")

class Task_Manager:
    """
    """

class Task_Manager:
    """
    """

    def __init__(self, credentials_filename):
        """
        """
        self.credentials_filename = credentials_filename

    def create_master_random_taskname(self, pom, Nworkers, user_name=None, user_password='Tester', user_org='Musketeer', task_name='Test', random_taskname=True):
        self.pom = pom
        self.Nworkers = Nworkers
        config = 'cloud'
        if random_taskname:
            rword = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
            version = '_' + rword
        else:
            version = '_v2'
        task_name += version
        self.task_name = task_name
        user_password  += version
        print('===========================================')
        print('Creating Master')
        
        if user_name is None:
            user_name = 'ma' + version
        
        ffl.Factory.register(config, fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
        context_master = ffl.Factory.context(config, self.credentials_filename)

        ffl_user_master = ffl.Factory.user(context_master)
        with ffl_user_master:
            try:
                ffl_user_master.create_user(user_name, user_password, user_org)
            except Exception as err:
                print(str(err).split(':')[1])

        context_master = ffl.Factory.context(config, self.credentials_filename, user_name, user_password, encoder = serializer.Base64Serializer)
        ffl_user_master = ffl.Factory.user(context_master)

        # Create task
        task_definition = {"task_name": task_name,
                            "owner": user_name, 
                           "quorum": self.Nworkers, 
                           "POM": self.pom,
                           "model_type": "None", 
                          }
        with ffl_user_master:
            try:
                result = ffl_user_master.create_task(task_name, ffl.Topology.star, task_definition)
            except Exception as err:
                print(str(err).split(':')[1])

        # We write to disk the name of the task, to be read by the workers. In the real system, 
        # the task_name must be communicated by other means.
        with open('current_taskname.txt', 'w') as f:
            f.write(task_name)

        return context_master, task_name

    def get_current_task_name(self):
        task_available = False
        while not task_available:
            try:
                with open('current_taskname.txt', 'r') as f:
                    self.task_name = f.read()
                task_available = True
            except:
                print('No available task yet...')
                time.sleep(1)
                pass
        return self.task_name

    def create_worker_join_task(self, id, user_password='Tester', user_org='Musketeer'):
        created = False
        while not created:
            try:
                self.task_name = self.get_current_task_name()
                print(self.task_name)
                config = 'cloud'

                version = self.task_name.split('_')[1]
                worker_name = 'worker_' + str(id) + '_' + version
                user_password  += version

                ffl.Factory.register(config, fflapi.Context, fflapi.User, fflapi.Aggregator, fflapi.Participant)
                context_w = ffl.Factory.context(config, self.credentials_filename)
                ffl_user_worker = ffl.Factory.user(context_w)
                with ffl_user_worker:
                    try:
                        ffl_user_worker.create_user(worker_name, user_password, user_org)
                    except Exception as err:
                        print(str(err).split(':')[1])
                context_w = ffl.Factory.context('cloud', self.credentials_filename, worker_name, user_password, encoder = serializer.Base64Serializer)
                user_worker = ffl.Factory.user(context_w)
                with user_worker:
                    try:
                        result = user_worker.join_task(self.task_name)
                        print('Worker %s has joined task %s' % (worker_name, self.task_name))
                        created = True
                    except Exception as err:
                        print(str(err).split(':')[1])
            except:
                print('waiting for Master...')
                time.sleep(1)
                pass

        return context_w, self.task_name

    def wait_for_workers(self, comms):

        stop = False
        workers = comms.commsffl.get_participants()

        while not stop: 
            try:
                with comms.commsffl:
                    resp = comms.commsffl.receive(5)
                participant = resp.notification['participant']
                workers.append(participant)
                print('Participant %s has joined' % participant)
            except Exception as err:
                print("Joined %d participants of %d" % (len(workers), self.Nworkers))
                #print(err)
                #print('Check here: error')
                #import code
                #code.interact(local=locals())
                pass

            if len(workers) == self.Nworkers:
                stop = True

        workers = comms.commsffl.get_participants()
        return list(workers.keys())
