# -*- coding: utf-8 -*-
"""
@author: kernke
"""
from setuptools import setup
from setuptools import find_packages


setup(name='image_processing',
      version='0.1',
      description='image analysis',
      url='https://github.com/kernke/image_processing',
      author='Kernke',
      author_email='r.s.kernke@gmail.com',
      license='AGPL-3.0',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
            'numpy',
            'opencv-python',
            'scipy',
            'numba'
               ])
