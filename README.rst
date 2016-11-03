clintrials
==========

README
------

clintrials is a library of clinical trial designs and methods in Python.
This library is intended to facilitate research.
It is provided "as-is" and the author accepts absolutely no responsibility whatsoever for the correctness or integrity of the calculations.



What does clintrials do?
----

* This library implements some designs used in clinical trials.
* It has implementations of O'Quigley's CRM design, Thall & Cook's EffTox design, and Wages & Tait's efficacy+toxicity design.
* There is also an implementation of my very own BEBOP trial design for the simultaneous study of bivariate binary outcomes (like efficacy and toxicity) in the presence of predictive variables, both continuous and binary.
* There is a bias towards phase I and II trial designs because that is my research area.
* I expect to add more designs in the future.
* It is written in pure Python, intentionally. This library would be quicker if it was written in C++ or Java but it would not be as portable or readable.
* Some of the code is fairly mature but the repo itself is young and in flux.
* I use 64 bit Python 3.5 but endeavour to maintain 2.7 compatibility.


Why Python?
----
No biostatisticians use Python, they use R / Stata / SAS, so why is this in Python?
Well, Python is used in lots of other sciences because it is rich and pleasant to work with.
Python is object-orientated, which is important when you are writing a bunch of classes that do a similar job in fundamentally different ways, like clinical trial designs, say.
It is nice to program in Python.
I think it is sadly underused in clinical trials.
Python also offers lots of extras and the parallel capabilities of IPython are having a positive impact on my work.

If you have never used Python, I recommend you install Anaconda, a distribution of Python aimed at academics and researchers that includes the tools we need, switch to the tutorial directory of clintrials and then fire up jupyter notebook.

Dependencies
----

* numpy, scipy, pandas & statsmodels - all of these are installed by Anaconda so I highly recommend that
* Some features also require matplotlib and ggplot. matplotlib also comes with Anaconda but ggplot will require a separate install.
If you need ggplot, be nice to yourself and use pip:
 `pip install ggplot`


How do I get set up?
----

There are two ways.
The first method uses pip and the Python package index.
The extras like the tutorials are not provided.
The second clones this repo using git.
Tutorials are provided.


Using pip to get just the clintrials code
----
To get the latest milestone release, use pip.
Open up a terminal or DOS session and fire off a:

`pip install clintrials`

The disadvantage of this method is that you don't get the nice tutorial workbooks that illustrate the methods. If you want those, use...


Using git to clone this repo, including tutorial notebooks
----

Navigate in terminal or DOS to a directory where you want the code and run

`git clone https://github.com/brockk/clintrials.git`

`cd clintrials`

You need to put clintrials on your path.
An easy way to do this is to edit the PYTHONPATH environment variable.
To do this in Mac or Linux, run

`export PYTHONPATH=$PYTHONPATH:$(pwd)`

Or, in Windows run

`set PYTHONPATH=%PYTHONPATH%;%CD%`

Then, load a jupyter notebook session for the tutorials using:

`jupyter notebook --notebook-dir=tutorials`

A browser window should appear and you should see the tutorials.
Tutorials related to the _Implementing the EffTox Dose-Finding Design in the Matchpoint Trial_ publication
are in the `matchpoint` directory.

Documentation
----

Documentation will eventually appear at

<http://brockk.github.io/clintrials/>

Contact
----

The repo owner is Kristian Brock, @brockk.
Feel free to get in contact through GitHub.