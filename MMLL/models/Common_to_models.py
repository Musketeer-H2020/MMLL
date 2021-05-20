# -*- coding: utf-8 -*-
'''
Collection of methods common to all models, to be inherited by models inside each POM.
'''

__author__ = "Marcos Fernández Díaz"
__date__ = "May 2021"


import pickle
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline

# Ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



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
        elif filename is None:
            self.display('Model Save Error: A valid filename must be provided.')
        else:
            try:
                extension = filename.split('.')[-1].lower()
                if self.name == 'NN':
                    if extension == 'onnx':
                        import logging, tf2onnx
                        # Disable warnings and logs from tf2onnx and tensorflow
                        tf2onnx_logger = logging.getLogger('tf2onnx')
                        tf2onnx_logger.setLevel(logging.ERROR)
                        tf_logger = logging.getLogger('tensorflow')
                        tf_logger.setLevel(logging.ERROR)
                        onnx_model, _ = tf2onnx.convert.from_keras(self.keras_model, output_path=filename)

                    else:
                        self.keras_model.save(filename)

                elif self.name == 'SVM':
                    if extension == 'pkl':
                        with open(filename, 'wb') as f:
                            pickle.dump(self, f)

                    elif extension=='onnx' or extension=='pmml':
                        # Create an equivalent model using Scikit Learn
                        NC = self.centroids.shape[0]
                        NI = self.centroids.shape[1]
                        gamma = 1 / (self.sigma**2)
                        export_model = SVC(C=1.0, gamma=gamma, decision_function_shape='ovo')
                        X = np.random.normal(0, 1, (100, NI))
                        w = np.random.normal(0, 1, (NI, 1))
                        y = np.sign(np.dot(X, w)).ravel()
                        export_model.fit(X, y)

                        export_model.support_vectors_ = self.centroids
                        export_model._dual_coef_ = self.weights.T * (-1)
                        export_model.dual_coef_ = self.weights.T
                        export_model._intercept_ = np.zeros((1,))
                        export_model.intercept_ = np.zeros((1,))
                        export_model.n_support_[0] = NC
                        export_model.n_support_[1] = 0
                        export_model.support_ = np.array(range(NC)).astype(np.int32)

                        if extension=='onnx':
                            input_type = [('float_input', FloatTensorType([None, NI]))]
                            onnx_model = convert_sklearn(export_model, initial_types=input_type)
                            with open(filename, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            
                        elif extension == 'pmml':                            
                            pipeline = PMMLPipeline([("estimator", export_model)])
                            sklearn2pmml(pipeline, filename, with_repr=True)
                            
                    else:
                        self.display('Model Save Error: Unsupported format. The valid extensions are: %s' %self.supported_formats)
          
                elif self.name == 'DSVM' or self.name == 'FBSVM':
                    if extension == 'pkl':
                        with open(filename, 'wb') as f:
                            pickle.dump(self, f)

                    elif extension=='onnx' or extension=='pmml':
                        NC = self.centroids.shape[0]
                        NI = self.centroids.shape[1]
                        gamma = 1 / 2 / self.sigma**2                     
                        export_model = SVR(C=1.0, gamma=gamma)
                        X = np.random.normal(0, 1, (100, NI))
                        w = np.random.normal(0, 1, (NI, 1))
                        y = np.sign(np.dot(X, w)).ravel()
                        export_model.fit(X, y)

                        export_model.support_vectors_ = self.centroids
                        export_model._dual_coef_ = self.weights[1:, :].T
                        export_model.dual_coef_ = self.weights[1:, :].T
                        export_model._intercept_ = self.weights[0, :]
                        export_model.intercept_ = self.weights[0, :]
                        export_model.n_support_[0] = NC
                        export_model.support_ = np.array(range(NC))

                        if extension == 'onnx':
                            # Convert into ONNX format
                            input_type = [('float_input', FloatTensorType([None, NI]))]
                            onnx_model = convert_sklearn(export_model, initial_types=input_type)
                            with open(filename, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            
                        elif extension == 'pmml':                            
                            pipeline = PMMLPipeline([("estimator", export_model)])
                            sklearn2pmml(pipeline, filename, with_repr=True)
                            
                    else:
                        self.display('Model Save Error: Unsupported format. The valid extensions are: %s' %self.supported_formats)

                elif self.name == 'Kmeans':
                    if extension == 'pkl':
                        with open(filename, 'wb') as f:
                            pickle.dump(self, f)
                    elif extension == 'onnx' or extension == 'pmml':
                        NC = self.centroids.shape[0]
                        NI = self.centroids.shape[1]
                        export_model = KMeans(n_clusters=NC)
                        X = np.random.normal(0, 1, (100, NI))
                        export_model.fit(X)
                        export_model.cluster_centers_ = self.centroids

                        if extension == 'onnx':
                            # Convert into ONNX format
                            input_type = [('float_input', FloatTensorType([None, NI]))]
                            onnx_model = convert_sklearn(export_model, initial_types=input_type)
                            with open(filename, "wb") as f:
                                f.write(onnx_model.SerializeToString())
                            
                        elif extension == 'pmml':                            
                            pipeline = PMMLPipeline([("estimator", export_model)])
                            sklearn2pmml(pipeline, filename, with_repr=True)

                    else:
                        self.display('Model Save Error: Unsupported format. The valid extensions are: %s' %self.supported_formats)
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
