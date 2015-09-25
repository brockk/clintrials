clintrials
==========

README
------

clintrials is a library of clinical trial designs and methods in Python.


What does clintrials do?
----

* This library performs the calculations necessary to run clinical trials.
* It has working implementations of O'Quigley et al's CRM design, Thall & Cook's EffTox design, and Wages & Tait's efficacy+toxicity design.
* There is also an implementation of my very own BeBoP trial design for the simultaneous study of bivariate binary outcomes (like efficacy and toxicity) in the presence of predictive variables, both continuous and binary.
* There is a bias towards phase I and II trial designs because that is my PhD research area.
* I expect to add more phase II and III designs in the future.
* It is written in pure Python, intentionally. This library would be quicker if it was written in C++ or Java but it would not be anywhere near as user friendly or accessible.
* Some of the code is fairly mature but the repo itself is young and in flux.

Why Python?
----
No biostatisticians use Python, they use R / Stata / SAS, so why is this in Python?
Well, Python is used **extensively** in most other sciences and I think it is pitifully underused in clinical trials.
This sad state of affairs arises because i) universities teach statistics with R / Stata / SAS and ii) humans are lazy lumps with a propensity for 'one size fits all' thinking. Once they learn enough R / Stata / SAS, they give up learning new tricks.

Python is object-orientated, which is important when you are writing a bunch of classes that do a similar job in fundamentally different ways, like clinical trial designs, say.

If you have never used Python, I recommend you install Anaconda, a distribution of Python aimed at academics and researchers that includes the tools we need, switch to the tutorial directory of clintrials and then fire up ipython notebook.
You will soon be hooked by how interactive and intuitive Python is.

Dependencies
----

* numpy, scipy, pandas & statsmodels - all of these are installed by Anaconda so I highly recommend that
* Some features also require matplotlib and ggplot. matplotlib also comes with Anaconda but ggplot will require a separate install. If you need ggplot, be nice to yourself and use pip:
 `pip install ggplot`


How do I get set up?
----

There are two ways. One is easier than the other.

The easy way
----
To get the latest milestone release, use pip.
Open up a terminal or DOS session and fire off a:

`pip install clintrials`

or, if your UNIX-based OS requires superuser authorisation to install software, perhaps use:

`sudo pip install clintrials`

The disadvantage of this method is that you don't get the nice tutorial workbooks that illustrate the methods. If you want those, use...


The ever-so-slightly-more-fiddly way:
----

Get the bleedy edge content (including tutorials) by cloning this git repo:

`mkdir clintrials`

`cd clintrials`

`git clone https://github.com/brockk/clintrials.git`

and then checkout the dev branch:

`git checkout dev`

Fire up a ipython notebook session for the tutorials using:

`ipython notebook --notebook-dir=tutorials`

A browser window should leap into life and you should see the tutorial workbooks.

Got any docs?
----

They will eventually appear at

<http://brockk.github.io/clintrials/>

Contribution guidelines and contact
----

I do not have any collaborators yet but input and help is welcome! The repo owner is Kristian Brock, @brockk. Please, feel free to get in contact through GitHub.