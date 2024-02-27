# automea


*automea* (Automated analysis of MEA datasets) is a open-source Python package for the analysis of Micro-Electrode Array (MEA) datasets. 

Bursts are detected using the *Max Interval Method*. Users can manually set *Max Interval Parameters*, or can use a machine learning model that dynamically predicts optimal parameters for specific recording times. Several models are distributed with *automea*, and users are free to fine-tune the existing models for their specific needs, or upload new models completely.  

The machine-learning-based burst detection routine is explained in the paper accompanying the package. 
