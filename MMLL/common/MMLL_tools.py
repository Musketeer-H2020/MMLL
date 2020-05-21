# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
May. 2020

'''

def display(message, logger, verbose, uselog=True):
    if verbose:
        print(message)
    if uselog:
        try:
            logger.info(message)
        except:
            pass