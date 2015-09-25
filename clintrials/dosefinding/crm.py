__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'

import logging
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, trapz
from scipy.optimize import minimize

from clintrials.dosefinding import DoseFindingTrial
from clintrials.common import empiric, logistic, inverse_empiric, inverse_logistic


def toxicity_likelihood(link_func, a0, beta, dose, tox, log=False):
    """ Calculate likelihood of toxicity outcome in CRM given link_func plus parameters and a dose & tox pair.

    Note: this method can be Memoized to save iterations at the expense of memory.
    Also, this method allows vectorisation (in beta, importantly) if your link_func allows vectorisation.

    Params:
    link_func, link function like logistic or empiric, taking params (dose_label, intercept, slope), returning a probability
    a0, the second parameter to link_func, the intercept
    beta, the third parameter to link_func, the slope
    dose, the first parameter to link_func, the dose label (often derived by backwards substitution!)
    tox, use 1 if toxicity observed, 0 if not.
    log, True to return log likelihood

    Returns a probability (or the log of a probability)

    """

    p = link_func(dose, a0, beta)
    if log:
        return tox*np.log(p) + (1-tox)*np.log(1-p)
    else:
        return p**tox * (1-p)**(1-tox)


def compound_toxicity_likelihood(link_func, a0, beta, doses, toxs, log=False):
    """ Calculate the compound likelihood of many toxicity outcomes in CRM given many dose & tox pairs.

    Params:
    link_func, link function like logistic or empiric, taking params (dose_label, intercept, slope), returning a probability
    a0, the second parameter to link_func, the intercept
    beta, the third parameter to link_func, the slope
    doses, a list of the first parameters to link_func, the dose labels (often derived by backwards substitution!)
    toxs, a list of toxicity markers. Use 1 if toxicity observed, 0 if not. Should be same length as doses.
    log, True to return log likelihood

    Returns a probability (or the log of a probability)

    """

    if len(doses) != len(toxs):
        ValueError('doses and toxs should be same length.')

    if log:
        l = 0
        for dose, tox in zip(doses, toxs):
            l += toxicity_likelihood(link_func, a0, beta, dose, tox, log=True)
        return l
    else:
        l = 1
        for dose, tox in zip(doses, toxs):
            l *= toxicity_likelihood(link_func, a0, beta, dose, tox, log=False)
        return l


def get_beta_hat_bayes(F, first_dose, codified_doses, toxs, beta_pdf, use_quick_integration=False, estimate_var=False):
    """ Get posterior estimate of beta parameter (and optionally its variance) in Bayesian CRM.

    Params:
    F, link function like logistic or empiric, taking params (dose_label, intercept, slope), returning a probability
    first_dose, the second parameter to F, the intercept
    codified_doses, a list of the first parameters to F, the dose labels (often derived by backwards substitution!)
    toxs, a list of toxicity markers. Use 1 if toxicity observed, 0 if not. Should be same length as codified_doses.
    beta_pdf, PDF of beta's prior distribution
    use_quick_integration, True to use a faster but slightly less accurate estimate of the pertinent
                            integrals, False to use a slower but more accurate method.
    estimate_var, True to get a posterior estimate of beta variance. Routine is quicker when False, obsv.

    Returns:
    a 2-tuple, (posterior mean, posterior variance)

    """

    if use_quick_integration:
        # This method uses simple trapezium quadrature. It is quite accurate and very fast.
        a, b = -10, 10 # TODO: these are sensible only for a dist centred on zero.
        n = 100 * max(np.log(len(codified_doses)+1)/2, 1)
        z, dz = np.linspace(a, b, num=n, retstep=1)
        num_y = z * compound_toxicity_likelihood(F, first_dose, z, codified_doses, toxs) * beta_pdf(z)
        denom_y = compound_toxicity_likelihood(F, first_dose, z, codified_doses, toxs) * beta_pdf(z)
        num = trapz(num_y, z, dz)
        denom = trapz(denom_y, z, dz)
        beta_hat = num / denom
        if estimate_var:
            num2_y = z**2 * compound_toxicity_likelihood(F, first_dose, z, codified_doses, toxs) * beta_pdf(z)
            num2 = trapz(num2_y, z, dz)
            exp_x2 = num2 / denom
            var = exp_x2 - beta_hat**2
            return beta_hat, var
        else:
            return num / denom, None
    else:
        # This method uses numpy's adaptive quadrature method. It is very accurate but quite slow
        num = quad(lambda t: t * compound_toxicity_likelihood(F, first_dose, t, codified_doses, toxs) * beta_pdf(t),
                   -np.inf, np.inf)
        denom = quad(lambda t: compound_toxicity_likelihood(F, first_dose, t, codified_doses, toxs) * beta_pdf(t),
                     -np.inf, np.inf)
        beta_hat = num[0] / denom[0]
        if estimate_var:
            num2 = quad(lambda t: t**2 * compound_toxicity_likelihood(F, first_dose, t, codified_doses, toxs) * beta_pdf(t),
                        -np.inf, np.inf)
            exp_x2 = num2[0] / denom[0]
            var = exp_x2 - beta_hat**2
            return beta_hat, var
        else:
            return beta_hat, None


def get_beta_hat_mle(F, first_dose, codified_doses, toxs, estimate_var=False):
    """ Get maximum likelihood estimate of beta parameter (and optionally its variance) in MLE CRM.

    TODO: how is estimating variance possible?

    Params:
    F, link function like logistic or empiric, taking params (dose_label, intercept, slope), returning a probability
    first_dose, the second parameter to F, the intercept
    codified_doses, a list of the first parameters to F, the dose labels (often derived by backwards substitution!)
    toxs, a list of toxicity markers. Use 1 if toxicity observed, 0 if not. Should be same length as codified_doses.
    beta_pdf, PDF of beta's prior distribution
    use_quick_integration, True to use a faster but slightly less accurate estimate of the pertinent
                            integrals, False to use a slower but more accurate method.
    estimate_var, True to get a posterior estimate of beta variance. Routine is quicker when False, obsv.

    Returns:
    a 2-tuple, (maximum likelihood estimate of beta, and variance)

    """
    if sum(np.array(toxs) == 1) == 0 or sum(np.array(toxs) == 0) == 0:
        logging.warn('Need heterogeneity in toxic events (i.e. toxic and non-toxic outcomes must be observed) for MLE to exist. \
        See Cheung p.23.')
        return np.nan, None

    f = lambda beta: -1 * compound_toxicity_likelihood(F, first_dose, beta, codified_doses, toxs, log=True)
    res = minimize(f, x0=0, method='nelder-mead')
    if estimate_var:
        logging.warn('Variance estimation in MLE mode is not implemented because I am not convinced it makes sense. \
        In dfcrm, Ken Cheung uses a method that I do not understand. It yields comparatively huge variances.')
    return res.x[0], None


def crm(prior, target, toxicities, dose_levels, first_dose=3, F_func=logistic, inverse_F=inverse_logistic,
        beta_dist=norm(loc=0, scale=np.sqrt(1.34)), method="bayes", use_quick_integration=False,
        estimate_var=False):
    """
    Run CRM calculation on observed dosages and toxicities.

    This method is similar to Ken Cheung's method in the R-package dfcrm. Take a look at that
    or his book for more information.

    Params:
    prior, list of prior probabilities of toxicity, from least toxic to most.
    target, target toxicity rate
    toxicities, list of bools denoting whether toxicity event occurred. 1=Yes, 0=No. Same length as dose_levels
    dose_levels, list of given 1-based dose levels, NOT the actual doses. See Cheung. Same length as toxicities
    first_dose, starting dose level, 1-based. I.e. first_dose=3 means the middle dose of 5.
    F_func and inverse_F, the link function and inverse for CRM method, e.g. logistic and inverse_logistic
    beta_dist, prior distibution for beta parameter
    method, one of "bayes" or "mle"
    use_quick_integration, numerical integration is slow. Set this to False to use the most accurate (slowest)
                            method; False to use a quick but approximate method.
                            In simulations, fast and approximate often suffices.
                            In trial scenarios, use slow and accurate!
    estimate_var, True to estimate the posterior variance of beta

    I omitted Ken's parameters:
    n=length(level), dosename=NULL, include=1:n, pid=1:n, conf.level=0.9, model.detail=TRUE, patient.detail=TRUE

    """

    if len(dose_levels) != len(toxicities):
        ValueError('toxicities and dose_levels should be same length.')
    if first_dose <= 0 or first_dose > len(prior):
        ValueError('Nonsense starting dose level.')

    beta0 = beta_dist.mean()
    dose_labels = [inverse_F(p, a0=first_dose, beta=beta0) for p in prior]
    codified_doses = [inverse_F(prior[dl-1], a0=first_dose, beta=beta0) for dl in dose_levels]
    if method == 'bayes':
        beta_hat, var = get_beta_hat_bayes(F_func, first_dose, codified_doses, toxicities, beta_dist.pdf, use_quick_integration,
                                           estimate_var)
        # dose_index = [abs(F_func(x, a0=first_dose, beta=beta_hat) - target) for x in dose_labels]
        # dose = np.argmin(dose_index) + 1
        # return dose, beta_hat, var
    elif method == 'mle':
        beta_hat, var = get_beta_hat_mle(F_func, first_dose, codified_doses, toxicities, estimate_var)
    else:
        print 'Only Bayes and MLE methods are implemented.'
        return None, None, None
    dose_index = [abs(F_func(x, a0=first_dose, beta=beta_hat) - target) for x in dose_labels]
    dose = np.argmin(dose_index) + 1
    return dose, beta_hat, var


class CRM(DoseFindingTrial):
    """ This is an object-oriented attempt at the CRM method.

    e.g. general usage

    >>> prior_tox_probs = [0.025, 0.05, 0.1, 0.25]
    >>> tox_target = 0.35
    >>> first_dose = 3
    >>> trial_size = 30
    >>> trial = CRM(prior_tox_probs, tox_target, first_dose, trial_size)
    >>> trial.next_dose()
    3
    >>> trial.update([(3,0), (3,0), (3,0)])
    4
    >>> trial.size(), trial.max_size()
    (3, 30)
    >>> trial.update([(4,0), (4,1), (4,1)])
    4
    >>> trial.update([(4,0), (4,1), (4,1)])
    3
    >>> trial.has_more()
    True

    """

    def __init__(self, prior, target, first_dose, max_size, F_func=empiric, inverse_F=inverse_empiric,
                 beta_prior=norm(0, np.sqrt(1.34)), method="bayes", use_quick_integration=False, estimate_var=True,
                 avoid_skipping_untried_escalation=False, avoid_skipping_untried_deescalation=False,
                 lowest_dose_too_toxic_hurdle=0.0, lowest_dose_too_toxic_certainty=0.0, termination_func=None):
        """

        Params:
        prior, list of prior probabilities of toxicity, from least toxic to most.
        target, target toxicity rate
        first_dose, starting dose level, 1-based. I.e. first_dose=3 means the middle dose of 5.
        F_func and inverse_F, the link function and inverse for CRM method, e.g. logistic and inverse_logistic
        beta_prior, prior distibution for beta parameter
        max_size, maximum number of patients to use in trial
        method, one of "bayes" or "mle"
        use_quick_integration, numerical integration is slow. Set this to False to use the most accurate (slowest)
                                method; False to use a quick but approximate method.
                                In simulations, fast and approximate often suffices.
                                In trial scenarios, use slow and accurate!
        estimate_var, True to estimate the posterior variance of beta
        avoid_skipping_untried_escalation, True to avoid skipping untried doses in escalation
        avoid_skipping_untried_deescalation, True to avoid skipping untried doses in de-escalation
        lowest_dose_too_toxic_hurdle,
        lowest_dose_too_toxic_certainty,
        termination_func, a function that takes this trial instance as a sole parameter and returns True if trial should
         terminate, else False. The function is invoked when trial is asked whether it has more.
         This function gives trials a general facility to terminate early if certain specified conditions are met.

        Note: this class makes no attempt (yet) to tackle the problem Ken Cheung describes of 'incoherent
        escalation', where the design escalates after observing a toxicity. When the cohort size is 1,
        i.e. patients are treated singly and dose is continually recalculated, incoherent escalation is clearly
        defined. However, in CRCTU, we typically run cohorts of three patients in dose-finding trials. The problem
        of incoherent escalation is much less clearly defined when the cohort size is greater than 1. Perhaps
        I will implement a coherent-escalation-only constraint iff cohort size is equal to 1.

        """

        DoseFindingTrial.__init__(self, first_dose=first_dose, num_doses=len(prior), max_size=max_size)

        self.prior = prior
        self.target = target
        self.F_func = F_func
        self.inverse_F = inverse_F
        self.beta_prior = beta_prior
        self.method = method
        self.use_quick_integration = use_quick_integration
        self.estimate_var = estimate_var
        self.avoid_skipping_untried_escalation = avoid_skipping_untried_escalation
        self.avoid_skipping_untried_deescalation = avoid_skipping_untried_deescalation
        self.lowest_dose_too_toxic_hurdle = lowest_dose_too_toxic_hurdle
        self.lowest_dose_too_toxic_certainty = lowest_dose_too_toxic_certainty
        self.termination_func = termination_func
        if lowest_dose_too_toxic_hurdle and lowest_dose_too_toxic_certainty:
            if not self.estimate_var:
                logging.warn('To monitor toxicity at lowest dose, I had to enable beta variance estimation.')
            self.estimate_var = True
        # Reset
        self.beta_hat, self.beta_var = None, None

    def _DoseFindingTrial__reset(self):
        self.beta_hat, self.beta_var = None, None

    def _DoseFindingTrial__process_cases(self, cases):
        return

    def _DoseFindingTrial__calculate_next_dose(self):
        proposed_dose, beta_hat, beta_var = crm(prior=self.prior, target=self.target, toxicities=self._toxicities,
                                                dose_levels=self._doses, first_dose=self._first_dose,
                                                F_func=self.F_func, inverse_F=self.inverse_F,
                                                beta_dist=self.beta_prior, method=self.method,
                                                use_quick_integration=self.use_quick_integration,
                                                estimate_var=self.estimate_var)
        self.beta_hat = beta_hat
        self.beta_var = beta_var

        max_dose_given = self.maximum_dose_given()
        min_dose_given = self.minimum_dose_given()
        if self.avoid_skipping_untried_escalation and max_dose_given and proposed_dose - max_dose_given > 1:
            # Avoid skipping untried doses in escalation by setting proposed dose to max_dose_given + 1
            proposed_dose = max_dose_given + 1
        elif self.avoid_skipping_untried_deescalation and min_dose_given and min_dose_given - proposed_dose > 1:
            # Avoid skipping untried doses in de-escalation by setting proposed dose to min_dose_given - 1
            proposed_dose = min_dose_given - 1
        # Note: other methods of limiting dose escalation and de-escalation are possible.

        # Excess toxicity at lowest dose?
        if self.lowest_dose_too_toxic_hurdle and self.lowest_dose_too_toxic_certainty:
            labels = [self.inverse_F(p, a0=self.first_dose(), beta=self.beta_prior.mean()) for p in self.prior]
            beta_sample = norm(loc=beta_hat, scale=np.sqrt(beta_var)).rvs(1000000)  # N.b. normal sample a la prior
            p0_sample = self.F_func(labels[0], a0=self.first_dose(), beta=beta_sample)
            p0_tox = np.mean(p0_sample > self.lowest_dose_too_toxic_hurdle)

            if p0_tox > self.lowest_dose_too_toxic_certainty:
                proposed_dose = 0
                self._status = -1

        return proposed_dose

    # def prob_tox(self, n=10**6):
    #     if self.estimate_var:
    #         # Estimate probability of toxicity using plug-in mean and variance for beta, and randomly
    #         # sampling values from normal. Why normal? Because the prior for is normal and the posterior
    #         # is asymptotically normal. For low n, non-normality may lead to bias.
    #         # TODO: research replacing this with a proper posterior integral when in bayes mode.
    #         labels = [self.inverse_F(p, a0=self.first_dose(), beta=self.beta_prior.mean()) for p in self.prior]
    #         beta_sample = norm(loc=self.beta_hat, scale=np.sqrt(self.beta_var)).rvs(n)
    #         p0_sample = [self.F_func(label, a0=self.first_dose(), beta=beta_sample) for label in labels]
    #         return np.array([np.mean(x) for x in p0_sample])
    #     else:
    #         raise Exception('CRM can only estimate posterior probabilities when estimate_var=True')

    def prob_tox(self):
        labels = [self.inverse_F(p, a0=self.first_dose(), beta=self.beta_prior.mean()) for p in self.prior]
        return [self.F_func(x, a0=self.first_dose(), beta=self.beta_hat) for x in labels]

    def prob_tox_exceeds(self, tox_cutoff, n=10**6):
        if self.estimate_var:
            # Estimate probability of toxicity exceeds tox_cutoff using plug-in mean and variance for beta, and randomly
            # sampling values from normal. Why normal? Because the prior for is normal and the posterior
            # is asymptotically normal. For low n, non-normality may lead to bias.
            # TODO: research replacing this with a proper posterior integral when in bayes mode.
            labels = [self.inverse_F(p, a0=self.first_dose(), beta=self.beta_prior.mean()) for p in self.prior]
            beta_sample = norm(loc=self.beta_hat, scale=np.sqrt(self.beta_var)).rvs(n)
            p0_sample = [self.F_func(label, a0=self.first_dose(), beta=beta_sample) for label in labels]
            return np.array([np.mean(x>tox_cutoff) for x in p0_sample])
        else:
            raise Exception('CRM can only estimate posterior probabilities when estimate_var=True')

    def has_more(self):
        """ Is the trial ongoing? """
        if not DoseFindingTrial.has_more(self):
            return False
        if self.termination_func:
            return not self.termination_func(self)
        else:
            return True

    def optimal_decision(self, prob_tox):
        """ Get the optimal dose choice for a given dose-toxicity curve.

        .. note:: Ken Cheung (2014) presented the idea that the optimal behaviour of a dose-finding
        design can be calculated for a given set of patients with their own specific tolerances by
        invoking the dose decicion on the complete (and unknowable) toxicity curve.

        :param prob_tox: collection of toxicity probabilities
        :type prob_tox: list
        :return: the optimal (1-based) dose decision
        :rtype: int

        """

        return np.argmin(np.abs(prob_tox - self.target)) + 1

    def get_tox_prob_quantile(self, p):
        """ Get the quantiles of the probabilities of toxicity at each dose using normal approximation.
        :param p: probability, i.e. 0.05 means 5th quantile, i.e. 95% of values are greater
        :type p: float
        :return: the quantiles of the probabilities of toxicity at each dose
        :rtype: list
        """
        norm_crit = norm.ppf(p)
        beta_est = self.beta_hat - norm_crit*np.sqrt(self.beta_var)
        labels = [self.inverse_F(p, a0=self.first_dose(), beta=self.beta_prior.mean()) for p in self.prior]
        p = [self.F_func(x, a0=self.first_dose(), beta=beta_est) for x in labels]
        return p

    def plot_toxicity_probabilities(self, chart_title=None, use_ggplot=False):
        """ Plot prior and posterior dose-toxicity curves.

        :param chart_title: optional chart title. Default is fairly verbose
        :type chart_title: str
        :param use_ggplot: True to use ggplot, else matplotlib
        :type use_ggplot: bool
        :return: plot of toxicity curves

        """

        if not chart_title:
            chart_title="Prior (dashed) and posterior (solid) dose-toxicity curves"
            chart_title = chart_title + "\n"

        if use_ggplot:
            from ggplot import (ggplot, ggtitle, geom_line, geom_hline, aes, ylim)
            import numpy as np
            import pandas as pd
            data = pd.DataFrame({'Dose level': self.dose_levels(),
                         'Prior': self.prior,
                         'Posterior': self.prob_tox(),
    #                      'Lower': crm.get_tox_prob_quantile(0.05),
    #                      'Upper': crm.get_tox_prob_quantile(0.95)
                        })
            var_name = 'Type'
            value_name = 'Probability of toxicity'
            melted_data = pd.melt(data, id_vars='Dose level', var_name=var_name, value_name=value_name)
            #melted_data['LineType'] =  np.where(melted_data.Type=='Posterior', '--', np.where(melted_data.Type=='Prior', '-', '..'))
            # melted_data['LineType'] =  np.where(melted_data.Type=='Posterior', '--', np.where(melted_data.Type=='Prior', '-', '..'))
            # melted_data['Col'] =  np.where(melted_data.Type=='Posterior', 'green', np.where(melted_data.Type=='Prior', 'blue', 'yellow'))
            # np.where(melted_data.Type=='Posterior', '--', '-')

            p = ggplot(melted_data, aes(x='Dose level', y=value_name, linetype=var_name)) + geom_line() \
                + ggtitle(chart_title) + ylim(0, 1) + geom_hline(yintercept=self.target, color='black')
            # Can add confidence intervals once I work out linetype=??? in ggplot

            return p
        else:
            import matplotlib.pyplot as plt
            import numpy as np
            dl = self.dose_levels()
            prior_tox = self.prior
            post_tox = self.prob_tox()
            post_tox_lower = self.get_tox_prob_quantile(0.05)
            post_tox_upper = self.get_tox_prob_quantile(0.95)
            plt.plot(dl, prior_tox, '--', c='black')
            plt.plot(dl, post_tox, '-', c='black')
            plt.plot(dl, post_tox_lower, '-.', c='black')
            plt.plot(dl, post_tox_upper, '-.', c='black')
            plt.scatter(dl, prior_tox, marker='x', s=300, facecolors='none', edgecolors='k')
            plt.scatter(dl, post_tox, marker='o', s=300, facecolors='none', edgecolors='k')
            plt.axhline(self.target)
            plt.ylim(0,1)
            plt.xlim(np.min(dl), np.max(dl))
            plt.xticks(dl)
            plt.ylabel('Probability of toxicity')
            plt.xlabel('Dose level')
            plt.title(chart_title)

            p = plt.gcf()
            phi = (np.sqrt(5)+1)/2.
            p.set_size_inches(12, 12/phi)
            # return p
