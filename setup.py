#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from setuptools import setup, find_packages

filepath = 'README.md'

setup(
    name='lapras',
    version='0.0.20',
    packages = find_packages(),

    description='scorecard,model',
    long_description=open(filepath).read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yhangang/lapras',
    author='Hayden Yang',
    author_email='yhangang@gmail.com',
    license='MIT',
    python_requires='>=3.5',
    install_requires=[
        # 'numpy >= 1.18.4',
        # 'pandas >= 0.25.1, <=0.25.3',
        # 'scipy >= 1.3.2',
        # 'scikit-learn =0.22.2',
        # 'seaborn >= 0.10.1',
        # 'statsmodels >= 0.13.1',
        # 'tensorflow >= 2.2.0, <=2.5.0',
        # 'hyperopt >= 0.2.7',
        # 'pickle >= 4.0',
    ],
    classifiers=[
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        # '': ['*.txt', '*.rst', '*.csv'],
        # include any *.msg files found in the 'test' package, too:
        # 'data': ['*.csv'],
    },
    # data_files=[('DealConfig.py'),('performance.py')],

    keywords=['scorecard', 'deep learning', 'wide&deep'],
    entry_points={

    },
    data_files=[filepath]
)

