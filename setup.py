from setuptools import setup, find_packages

setup(
    name='MMLL',
    version='2.2.0',
    description='Package for interacting with MMLL',
    author='Angel Navia Vázquez, Marcos Fernandez Diaz, Roberto Diaz Morales',
    author_email='angel.navia@uc3m.es, marcos.fernandez@treelogic.com, roberto.diaz@treelogic.com',
    license='GPLv3',
    packages=find_packages('.'),
    python_requires='>=3.6',
    install_requires=[
        'six==1.15.0',
        'transitions==0.6.9',
        'pyparsing==2.3',
        'pygraphviz==1.5',
        'numpy==1.19.2',
        'sklearn==0.0',
        'scikit-learn',
        'matplotlib',
        'tensorflow==2.4.1',
        'phe==1.4.0',
        'dill==0.3.2',
        'tqdm==4.61.0',
        'pympler==0.8',
        'torchvision==0.8.1',
        'pillow==7.2.0',
        'skl2onnx==1.8.0',
        'sklearn2pmml==0.71.1',
        'tf2onnx==1.8.5'   
        ],
    url='https://github.com/Musketeer-H2020/MMLL'
)
