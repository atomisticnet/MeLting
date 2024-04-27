"""
Copyright (c) 2024 atomistic.net

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import setup, find_packages
from codecs import open
import os

__author__ = "Vahe Gharakhanyan"
__email__ = "vg2471@columbia.edu"

here = os.path.abspath(os.path.dirname(__file__))
package_name = 'MeLting'
package_description = 'Machine-learning models for melting-temperature prediction'

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as fp:
    long_description = fp.read()

# Get version number from the VERSION file
with open(os.path.join(here, package_name, 'VERSION')) as fp:
    version = fp.read().strip()
    
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]

setup(
    name=package_name,
    version=version,
    description=package_description,
    long_description=long_description,
    url='https://github.com/atomisticnet/MeLting',
    author=__author__,
    author_email=__email__,
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords=['materials science', 'thermodynamics', 'machine learning'],
    packages=find_packages(exclude=['tests']),
    install_requires=REQUIREMENTS,
    include_package_data=True,
)