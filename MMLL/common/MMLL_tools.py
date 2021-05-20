# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
May. 2020

'''
import numpy as np

def display(message, logger, verbose, uselog=True):
    if verbose:
        print(message)
    if uselog:
        try:
            logger.info(message)
        except:
            pass

def estimate_centroids(NCini, NI, NCcandidates, minvalue, maxvalue, verbose=False):
    C = np.random.uniform(minvalue, maxvalue, (NCcandidates, NI))
    stop = False
    try:
        while not stop: 
            # Computing Kcc
            C2 = np.sum(C ** 2, axis=1).reshape((-1, 1))
            Dcc = np.sqrt(np.abs(C2 -2 * np.dot(C, C.T) + C2.T))
            
            Dcc_ = ((Dcc + 1000 * np.eye(Dcc.shape[0])) * 1000000.0).astype(int)

            min_pos = np.where(Dcc_ == np.amin(Dcc_))[0]
            c1 = C[min_pos[0], :]
            c2 = C[min_pos[1], :]

            cnew = (c1 + c2) / 2
            C = np.delete(C, [min_pos[0], min_pos[1]], 0)
            C = np.vstack((C, cnew))
            
            if C.shape[0] == NCini:
                stop = True
            if verbose: 
                print('Merging %d and %d with distance %f, NC = %d' % (min_pos[0], min_pos[1], Dcc[min_pos[0], min_pos[1]], NCcandidates))
        return C
    except:
        print('ERROR AT ')
        import code
        code.interact(local=locals())
