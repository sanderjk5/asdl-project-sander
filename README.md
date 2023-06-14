# Asdl Project Jurek Sander
Project for the lecture Analyzing Software using Deep Learning.

## Creating the random bug dataset
The folder '/dockerfiles' contains the fixed dockerfile which allows to create the random bug dataset of Buglab without Nvidia Cuda. The new rewriting scout in '/rewriting/rewritescouts.py' extends the bug patterns of Buglab. The bug patterns are the following: Variable Misuse, Argument Swapping, Wrong Operator (Assignment, Boolean, Binary or Comparison Operator), Wrong Literal and Wrong Loop Statement (new added by me).
The folder '/data' contains already generated datasets that can be used for training and evaluating the model. Every graph of the dataset represents one original python function and offers additional information regarding the potential localizations where the bugs could be inserted. It states also whether a bug was inserted and if so, at which part of the code.

## Supervised Model for bug localization
The file '/gnn-model/buglocalization-gnn.py' contains the implementation of a gnn model to localize the bug in a buggy python function. It can be trained and evaluated with the following command:

    $ python -m gnn-model.geometric-gnn path_training_data path_validation_data path_testing_data

The input folders must contain files with graphs in the format defined by Buglab. The results will be stored in the textfile 'evaluation.txt'.

## Supervised Model for bug classification
The file '/gnn-model/bugclassification-gnn.py' contains the implementation of a gnn model to classify python functions as buggy or not. It can be trained and evaluated with the following command:

    $ python -m gnn-model.geometric-gnn folder_path

The input folder must contain files with graphs in the format defined by Buglab. But with the hardware I used, it was not possible to train the model such that it learns to differentiate between buggy and correct methods.