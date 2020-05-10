# Machine Learning Data Handler
This module contains [Python](https://www.python.org/) objects which are intended to faciliate Machine Learning projects by providing methods to quickly handle data. 
The idea is to have a three different classes for handling common Machine Learning related steps with regards to processing data. 
A major step of current Machine Learning models is to handle raw data, explore features, generate new features and pre-process them 
to feed them into a Machine Learning pipeline. Therefore, the three following classes defined and can be used for other Machine Learning projects.
Hint: these classes wrap up methods that already exists, (for instance, sklearn methods). Therefore, this module is considered to be rather 
an ease for Machine Learning project than implementing a completely new algorithm or whatsoever

   - **FeatureBuilder**: is a class used to generate features out of raw data. Its purpose is especially to handle special formats of data. At the current stage, it allows handling of the following two data source (others are to be added):
       - *Date and Time*: provdes methods for generating date and time related features
       - *GIS*: Geo Information System data. It is based on latitude and longitude columns and can create features out of them
   - **DataExplorer**: is used to generate quickly insights. It provides methods to handle different types of data, provides stats about columns and quickly generate plots to analyze relationsships
       - *Numerical*: provides methods to analyze numerical features (plots, density plots, correlations, etc.).
       - *Categorical*: provides methods to analyze categorical features (frequencies, plots, etc.)
       - *Mixed Type*: provides methods to analyze mixed types (basic stats, plots, etc.)
       - *Data by Time*: provides methods to analyze a feature by time, i.e. Time Series
       - *Geo Information*: provides methods for plotting and generating images with gis Data Information
   - **PreProcessor**: is a class that can be used to quickly generated data pre-processing pipelines. They provide methods for numerical and categorical features. They provide control for initializing Pre Processing classes. 
       - *Scaler*: includes Scaler from the scikit-learn to scale numerical data
       - *Imputer*: includes an Imputer for missing values from the scikit-learn to fill missing values
       - *Encoding*: incudes methods for encoding label data

The src file can be imported and used for other projects. 

# Prerequisits
The source code is written in [Python 3.8](https://www.python.org/). It use some of the standard libraries which are included in the most Python environments.
It furthermore uses some more specialized packages for Machine Learning and Geo Data visualization. You can pip install all requirements by

    pip install requirements

# Installation
You can clone this repository by running:
	
	git clone https://github.com/dheinz0989/Machine-Learning-Data-Handler

# Example usage
An example usage can be found in the [test](https://github.com/dheinz0989/Machine-Learning-Data-Handler/blob/master/test/Quick_usage_demo.ipynb). It imports most of the modules features and includes them in a test script. 

# Documentation
More details with regards to the function and for which use case they are intended to be used can be found in the [docs](https://github.com/dheinz0989/Machine-Learning-Data-Handler/blob/master/docs/build/html/ML_Data_Handler.html). 

# To Do
This repository has a lot of things which are not implemented yet. Amongs others, the following implementation are planned:
    - add further various method
    - enhance documentation
    - testing 