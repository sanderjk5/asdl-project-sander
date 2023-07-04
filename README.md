# ASDL Project Jurek Sander
Project for the lecture Analyzing Software using Deep Learning.

## Creating the random bug dataset
The folder '/dockerfiles' contains the fixed dockerfile which allows to create the random bug dataset of Buglab without Nvidia Cuda. The new rewriting scout in '/rewriting/rewritescouts.py' extends the bug patterns of Buglab. The bug patterns are the following: Variable Misuse, Argument Swapping, Wrong Operator (Assignment, Boolean, Binary or Comparison Operator), Wrong Literal and Wrong Loop Statement (new added by me).  
The folder '/data' contains already generated datasets that can be used for training and evaluating the model. Every graph of the dataset represents one original python function and offers additional information regarding the potential localizations where the bugs could be inserted. It states also whether a bug was inserted and if so, at which part of the code. Datasets with different sizes and bugtypes are offered. Each dataset has a subfolders that is called '/split' which contains the splits of the data set into training, validation and test data.

## Supervised model for bug localization
The file '/gnn-model/buglocalization-gnn.py' contains the implementation of a gnn model to localize the bug in a buggy python function. It can be trained and evaluated with the following command:

    $ python -m gnnmodels.buglocalization-gnn folder_training_data folder_validation_data folder_testing_data

The input folders must contain files with graphs in the format defined by Buglab. The results will be stored in the textfile 'evaluation.txt'. Additional graphics show the accuracy and the loss over all training epochs.
