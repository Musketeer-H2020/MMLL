# -*- coding: utf-8 -*-
'''
Collection of methods common to all models, to be inherited by models inside each POM.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "April 2021"


import pickle



class Common_to_models():
    """
    This class implements some basic functionalities common to all models.
    """

    def __init__(self):
        """
        Create a :class:`Common_to_models` instance.

        Parameters
        ----------
        None
        """



    def save(self, filename=None):
        """
        Save the trained model to local disk.

        Parameters
        ----------
        filename: string
            Full path and filename in which to store the model.      
        """
        if not self.is_trained:
            self.display('Model Save Error: Model not trained yet, nothing to save.')
        else:
            try:
                if self.name == 'NN':
                    self.keras_model.save(filename)
                else:
                    with open(filename, 'wb') as f:
                        pickle.dump(self, f)
                self.display('Model saved at %s' %filename)
            except:
                self.display('Model Save Error: Model cannot be saved, check the provided filename.')
                raise



    def display(self, message):
        """
        Use a logger to display messages on the console and/or file.

        Parameters
        ----------
        message: string
            Message to be displayed.      
        """
        try:
            self.logger.info(message)
        except:
            print(message)
