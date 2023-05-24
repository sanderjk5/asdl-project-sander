# Asdl Project Jurek Sander
Project for the lecture Analyzing Software using Deep Learning.

## Creating the random bug dataset
The folder '/dockerfiles' contains the fixed dockerfile which allows to create the random bug dataset of Buglab without Nvidia Cuda. The new rewriting scout in '/rewriting/rewritescouts.py' extends the bug patterns of Buglab.

## Supervised Model for bug detection
The file '/gnn-model/geometric-gnn.py' contains the implementation of a gnn model to classify graphs as buggy or not. It can be trained and evaluated with the following command:

    $ python -m gnn-model.geometric-gnn folder_path

The input folder must contain files with graphs in the format defined by Buglab. Example files are in the folder '/target'.
