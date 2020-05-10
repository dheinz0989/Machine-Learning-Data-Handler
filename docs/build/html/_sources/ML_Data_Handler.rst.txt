.. Machine Learning Data Handler documentation master file, created by
   sphinx-quickstart on Sun May 10 20:36:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Machine Learning Data Handler's documentation!
=========================================================

.. toctree::
   :maxdepth: 2

Welcome to the documentation site of the Machine Learning Data Handler. The following site includes all necessary info about the current status.

Last Update
**************
Last update of the documentation is from |today|. This document will be split upon others while including a document for each module used within this project. 

Context
****************
The main goal of the code in this repository is to provide a Python classes to quickly solve tasks which are common amongst Machine Learning Project with respect to data. It aims at providing code
that can be used in other Machine Learning projects to ease the execute of the very time-consuming step of data handling. It Therefore includes three major classes:

  1. *Data Explorer*: is a class that aims at quickly generating insights about the underlying data. It includes methods for different types of data, provides basic statistic, memory usage, missing data calculator
  plotting functions. It furthermore has some functionality to analyze data by time, mixed types and GIS (Geo Information System) in terms of latitude and longitude columns

  2. *FeatureBuilder*: is a class which aims at quickly generating features. It is intended to be used for generating feature of special data. Different feature generating steps can be add in a composite class 
  and are stored in a pipeline. The pipeline can then be wrapped and executed to generate new fetaures. The special data mentioned in the beginning is at the moment (|today|):

    - *time*: several feature related to date time columns are provided:
      - converting strings to time
      - converting Unix time Stamps to date time
      - time difference between two date time columns 
      - basic time features
    - *GIS*: several features related to Geo Information Systems are provided as:
      - basic gis features applying radians or rounding the latitude/longitude
      - Haversine distance between two points
      - Clustering algorithms for geo data

  3. *DataPreProcessor*: allows to quickly generate a pre-processed dataframe for Machine Learning purposes. It aims at quickly building data pipelines and generate test and train data. It gives the user the control
  to modify the transformation steps but includes using properties to quickly apply the same steps to named or typed based feature lists.


The respetive tables are written from a parquet file format

Project Directory
*****************
The project directory is as shown in the following. The respective directories are:
  - *src/:* contains all modules for the code 
  - *docs/:* contains all files for documentation
  - *ims/:* contains images and data vizualisation
  - *test/:* contains a script for testing purposes

Furthermore, different files are found directly in the repository
  - *requirements.txt*: required Python packages within the code
  - *README.md*: information about this module


::

  ML_Data_Prep_backup/
  ├── __init__.py
  ├── docs/
  │   ├── build/
  │   │   ├── doctrees/
  │   │   │   ├── base.doctree
  │   │   │   ├── Basic_Attributes.doctree
  │   │   │   ├── Data_Selector.doctree
  │   │   │   ├── environment.pickle
  │   │   │   └── ML_Data_Handler.doctree
  │   │   └── html/
  │   │       ├── .buildinfo
  │   │       ├── _modules/
  │   │       │   ├── baseattrs.html
  │   │       │   ├── dataframeselector.html
  │   │       │   ├── index.html
  │   │       │   ├── numpy.html
  │   │       │   └── sklearn/
  │   │       │       └── base.html
  │   │       ├── _sources/
  │   │       │   ├── base.rst.txt
  │   │       │   ├── Basic_Attributes.rst.txt
  │   │       │   ├── Data_Selector.rst.txt
  │   │       │   └── ML_Data_Handler.rst.txt
  │   │       ├── _static/
  │   │       │   ├── basic.css
  │   │       │   ├── doctools.js
  │   │       │   ├── documentation_options.js
  │   │       │   ├── file.png
  │   │       │   ├── graphviz.css
  │   │       │   ├── jquery-3.4.1.js
  │   │       │   ├── jquery.js
  │   │       │   ├── language_data.js
  │   │       │   ├── minus.png
  │   │       │   ├── nature.css
  │   │       │   ├── plus.png
  │   │       │   ├── pygments.css
  │   │       │   ├── searchtools.js
  │   │       │   ├── underscore-1.3.1.js
  │   │       │   └── underscore.js
  │   │       ├── base.html
  │   │       ├── Basic_Attributes.html
  │   │       ├── Data_Selector.html
  │   │       ├── genindex.html
  │   │       ├── ML_Data_Handler.html
  │   │       ├── objects.inv
  │   │       ├── py-modindex.html
  │   │       ├── search.html
  │   │       └── searchindex.js
  │   ├── make.bat
  │   ├── Makefile
  │   └── source/
  │       ├── _static/
  │       ├── _templates/
  │       ├── conf.py
  │       └── ML_Data_Handler.rst
  ├── logs/
  │   └── log_src.utilities_2020-05-10.log
  ├── ml_prepaper.py
  ├── README.md
  ├── requirements.txt
  ├── src/
  │   ├── __init__.py
  │   ├── baseattrs.py
  │   ├── dataframeselector.py
  │   ├── date_and_time.py
  │   ├── explorer.py
  │   ├── feature_builder.py
  │   ├── gis.py
  │   ├── preprocess.py
  │   ├── textfeatures.py
  │   └── utilities.py
  └── test/
      ├── config.yaml
      ├── Quick_usage_demo.ipynb
      └── sunspots.csv


Usage
*****
Before running the Code, ensure that you have all requirements installed. You can the import them in your costum Machine Learning Projects and use them.
A usage jupyter notebook is shown in the test directory.

Source Code
===========


1. ``baseattrs.py``: Contains basic classes which are inherited most other modules.

2. ``dataframeselector.py``: Contains methods to select subsetf of DataFrames based on names or type.

3. ``date_and_time``: Provides methods to generate date and time features and methods to analyze them. 

4. ``explorer:`` Contains the source code for the DataExplorer.

5. ``feature_builder:`` Contains the source code for the  FeatureBuilder.

6. ``gis:`` Provides methods to generate Geoinformation System features and methods to analyze them. 

7. ``preprocess:`` Contains the source code for the Data Pre DataPreProcessor.

8. ``utilities``: Contains some uti functions.

9. ``textfeatures``: not yet implemented 

Basic Attributes
**********************
.. automodule:: baseattrs
   :members:
   :undoc-members:
   :private-members:


Data Selector
**********************
.. automodule:: dataframeselector
   :members:
   :undoc-members:
   :private-members:


Data and Time
**********************
.. automodule:: date_and_time
   :members:
   :undoc-members:
   :private-members:


Geoinformation System
**********************
.. automodule:: gis
   :members:
   :undoc-members:
   :private-members:


Explorer
**********************
.. automodule:: explorer
   :members:
   :undoc-members:
   :private-members:


Feature Builder
**********************
.. automodule:: feature_builder
   :members:
   :undoc-members:
   :private-members:


Data Pre Processor
**********************
.. automodule:: preprocess
   :members:
   :undoc-members:
   :private-members:


utilities
**********************
.. automodule:: utilities
   :members:
   :undoc-members:
   :private-members:
   :inherited-members:
   :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
