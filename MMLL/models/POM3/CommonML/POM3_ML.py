# -*- coding: utf-8 -*-

__author__ = "Marcos Fernández Díaz"
__date__ = "Oct 2019"


class POM3ML():
    """
    This class implements basic methods and protocols common to all machine learning methods
    under POM3. It is almost empty by now, but we will migrate here some common methods in 
    the future.
    To be inherited by the specific ML models.
    """

    def __init__(self):
        """
        Create a :class:`POM3ML` instance.
        """
        return

    def display(self, message):
        """
        Print message to log file and display on screen if verbose=True

        :param message: string message to be shown/logged
        :type message: str
        """

        if self.verbose:
            print(message)
        self.logger.info(message)
