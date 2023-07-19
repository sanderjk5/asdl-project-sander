# ASDL Project Jurek Sander
Project for the lecture Analyzing Software using Deep Learning.

## Creating the random bug dataset
The folder '/dockerfiles' contains the fixed dockerfile which allows to create the random bug dataset of Buglab without Nvidia Cuda. The new rewriting scout in '/rewriting/rewritescouts.py' extends the bug patterns of Buglab. The bug patterns are the following: Variable Misuse, Argument Swapping, Wrong Operator (Assignment, Boolean, Binary or Comparison Operator), Wrong Literal and Wrong Loop Statement (newly added with the scout).  
The folder '/data' contains already generated datasets that can be used for training and evaluating the model. Every graph of the dataset represents one original python function and offers additional information regarding the potential localizations where the bugs could be inserted ("reference nodes"). It also states at which node the bug was inserted. The subfolder '/split' contains already a split into training, evaluation and test data.

## Requirements
To train and evaluate the model, install first the following packages: torch, torchvision, torchaudio, wheel and torch_scatter using the following commands:

    $ pip install torch torchvision torchaudio
    $ pip install --no-cache-dir --upgrade wheel 
    $ pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html 

Install the remaining packages using the following command:

    $ pip install -r .\requirements.txt 


## Supervised model for bug localization
The file '/gnn-model/buglocalization-gnn.py' contains the implementation of a gnn model to localize the bug in a buggy python function. It can be trained and evaluated with the following command:

    $ python -m gnnmodels.buglocalization-gnn [options] TRAIN_DATA_PATH VALID_DATA_PATH TEST_DATA_PATH

It is possible to configure which defined scenario described in my report should be executed with the option '--scenario=<scenario>'. The following scenarios exists:
* 1: All bug patterns; unrestricted number of reference nodes
* 2: All bug patterns; maximal 25 reference nodes
* 3: Without the bug patterns "Variable Misuses", "Argument Swaps", "Wrong Literals"; unrestricted number of reference nodes
* 4: Without the bug patterns "Variable Misuses", "Argument Swaps", "Wrong Literals"; maximal 25 reference nodes
* 5: Only the bug pattern "Wrong Loop Statements"; unrestricted number of reference nodes
* 6: Only the bug pattern "Wrong Loop Statements"; maximal 25 reference nodes

The input folders must contain files with graphs in the format defined by Buglab. The results will be stored in the textfile 'evaluation.txt'. Additional plots show the accuracy and the loss over all training epochs.
