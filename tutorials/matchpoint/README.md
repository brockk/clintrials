# EffTox in Matchpoint tutorials #

## README ##


The three tutorials 

DTPs.pynb

Utility.ipynb

Ambivalence.ipynb

are provided to complement the publication _Implementing the EffTox Dose-Finding Design in the Matchpoint Trial_ (Brock et al.,in submission).
Please consult the paper for the clinical background, the methodology details, and full explanation of the terminology.

These notebooks can be viewed online at https://github.com/brockk/clintrials/tree/master/tutorials/matchpoint
but to run them, you will need Python and Jupyter.
We recommended you install Anaconda because it greatly simplifies the process of installing Python and the common add-ons like jupyter, numpy, scipy, pandas, etc.
Install it from https://www.continuum.io/downloads.

The notebooks with plots use ggplot. To get ggplot, run:

`pip install ggplot`

at the command line.

Clone this repository by navigating to a directory where the code will live and running

`git clone https://github.com/brockk/clintrials.git`

`cd clintrials`

You need to put clintrials on your path. 
An easy way to do this is to edit the PYTHONPATH environment variable.
To do this in Mac or Linux, run 
 
`export PYTHONPATH=$PYTHONPATH:$(pwd)`
 
Or, in Windows run
 
`set PYTHONPATH=%PYTHONPATH%;%CD%`

Then, load a jupyter notebook session for the tutorials using:

`jupyter notebook --notebook-dir=tutorials/matchpoint`

A browser window should appear and you should see the tutorials.
"Test ggplot.ipynb" is a notebook to test whether ggplot is correctly installed. 

## Plan B
If adding clintrials to your path by editing environment variables is not an option for you (e.g. lack of admin rights), an alternative is to copy the notebooks you want to use to the root directory that contains the folders named `docs` and `tests` and the `README.md` file.
Then navigate to that directory in console and run 

`jupyter notebook`

clintrials should automatically be on your path because it resides in the executing directory.






