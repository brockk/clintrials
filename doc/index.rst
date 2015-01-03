.. clintrials documentation master file, created by
   sphinx-quickstart on Sat Dec 13 19:54:47 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to clintrials's documentation!
======================================

Contents: BLAH

.. toctree::
   :maxdepth: 2



Common, helpful stuff
======================

General functions
__________________

.. automodule:: clintrials.common
    :members: inverse_logit

dfcrm-style link functions and their inverses
______________________________________________

See http://cran.r-project.org/web/packages/dfcrm/dfcrm.pdf

.. automodule:: clintrials.common
    :members: empiric, inverse_empiric, logistic, inverse_logistic, hyperbolic_tan, inverse_hyperbolic_tan



Coll
____

.. automodule:: clintrials.coll
    :members:


Recruitment
____________
.. automodule:: clintrials.recruitment
    :members: RecruitmentStream, ConstantRecruitmentStream, QuadrilateralRecruitmentStream

Util
_____
.. automodule:: clintrials.util
    :members:



Phase I Trial Designs
======================
Dose-finding based on toxicity
_______________________________

These designs are used to find the maximum tolerable dose (MTD) for cytotoxic agents.


Dose-finding based on efficacy and toxicity
____________________________________________

These designs are used to find the optimum biological dose (OBD) for cytotoxic and cytostatic agents.



Phase II Trial Designs
=======================

.. automodule:: clintrials.phase2
    :members:

Time-to-Event Designs
______________________

Time-to-event outcomes are not typical in phase II clinical trials, but they do exist.

.. automodule:: clintrials.tte
    :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

