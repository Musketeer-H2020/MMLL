from setuptools import setup, find_packages

setup(
    name='MMLL',
    version='0.5.0',
    description='Package for interacting with MMLL',
    author='angelnaviavazquez, Marcos Fernandez Diaz',
    author_email='navia@ing.uc3m.es, marcos.fernandez@treelogic.com',
    license='GPLv3',
    packages=find_packages('.'),
    python_requires='>=3.6',
    install_requires=[
        'transitions==0.6.9',
        'pygraphviz==1.5',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'numpy==1.16.4', 
        'Keras==2.2.4',
        'tensorflow==1.14.0',
        'phe==1.4.0',
        'dill==0.3.2',
        'tqdm==4.50.2',
        'pympler==0.8',
        'torchvision==0.8.1',
        'pillow==7.2.0',
    ],
    url='https://github.com/Musketeer-H2020/MMLL'
)
