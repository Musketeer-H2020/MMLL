from setuptools import setup, find_packages

setup(
    name='MMLL',
    version='0.1.0',
    description='Package for interacting with MMLL',
    author='angelnaviavazquez',
    author_email='navia@ing.uc3m.es',
    license='Apache 2.0',
    packages=find_packages('.'),
    python_requires='>=3.6',
    install_requires=[
        'pika==0.13.0'
    ],
    url='https://github.com/Musketeer-H2020/MMLL'
)
