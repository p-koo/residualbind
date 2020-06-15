"""Setup file for installing neuralbinder
PA created this file.
See:
https://github.com/p-koo/neuralbinder
"""

# Using setuptools instead of distutils
from setuptools import setup, find_packages


# To use a consistent encoding
from codecs import open
from os import path

setup(
    name='neuralbinder',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.1',

    description='NeuralBinder consists of a bunch of routines to build and train neural network models, primarily using deepomics, which is a high-level tensorflow API for biological applications.',
    long_description=open('README.rst').read(),

    # The project's main homepage.
    url='https://github.com/p-anand/neuralbinder',

    # Author details
    author='Peter Koo, Tim Dunn, Praveen Anand',
    author_email='peter_koo@harvard.edu',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 2 - Pre-Alpha',

        #Environment and Framework used
        'Environment :: Console',
        'Framework :: Jupyter',
        'Framework :: IPython',

	# Language used 
        'Natural Language :: English',

	# Operating systems that should work??
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        
        # Indicate who your project is intended for
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research'
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Intended Audience :: System Administrators',

        #Topics of the package
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        'License :: Free For Educational Use',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # Scripts that will be included in the bin directory
    scripts = ['neuralbinder/bin/affinitybinder_rnacompete_train'],

    # What does your project relate to?
    keywords='Deeplearning RNAprotein Motifs Saliency Bioinformatics',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['matplotlib',
                      'tensorflow',
                      'numpy',
                      'h5py',
                      'sklearn',
                      'scipy',
                      'pandas'
    ],

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # https://stackoverflow.com/questions/5897666/how-do-i-use-data-in-package-data-from-source-code
    #package_data={
    #    'neuralbinder': ['datasets/*.h5'],
    #},
)
