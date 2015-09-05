__author__ = 'Kristian Brock'
__contact__ = 'kristian.brock@gmail.com'


""" Classes and functions for the PePs2 trial """

from collections import OrderedDict
from datetime import datetime
import glob
import json
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm, binom, uniform

from clintrials.phase2 import Phase2EffToxBase
from clintrials.stats import chi_squ_test, or_test, ProbabilityDensitySample
from clintrials.util import correlated_binary_outcomes, atomic_to_json, iterable_to_json



def pi_t(disease_status, mutation_status, alpha0=0, alpha1=0, alpha2=0):
    z = alpha0 + alpha1*disease_status + alpha2*mutation_status
    response = 1/(1+np.exp(-z))
    return response


def pi_e(disease_status, mutation_status, beta0=0, beta1=0, beta2=0):
    z = beta0 + beta1*disease_status + beta2*mutation_status
    return 1/(1+np.exp(-z))


def pi_ab(disease_status, mutation_status, eff, tox, alpha0, beta0, beta1, beta2, psi):
    a=tox
    b=eff
    p1 = pi_t(disease_status, mutation_status, alpha0)
    p2 = pi_e(disease_status, mutation_status, beta0, beta1, beta2)
    response = p1**a * (1-p1)**(1-a) * p2**b * (1-p2)**(1-b)
    response = response + (-1)**(a+b) * p1 * (1-p1) * p2 * (1-p2) * (np.exp(psi) - 1) / (np.exp(psi) + 1)
    return response


def l_n(D, alpha0, beta0, beta1, beta2, psi) :
    if len(D) > 0:
        disease_status, mutation_status, eff, tox = zip(*D)
        response = np.ones_like(alpha0)
        for i in range(len(tox)):
            p = pi_ab(disease_status[i], mutation_status[i], eff[i], tox[i], alpha0, beta0, beta1, beta2, psi)
            response = response * p
        return response
    else:
        return np.ones_like(alpha0)


def get_posterior_probs(D, priors, tox_cutoffs, eff_cutoffs, n=10**5):
    """ Get the posterior probabilities after having observed cumulative data D in a TODO trial.

    Note: This function evaluates the posterior integrals using Monte Carlo integration. Thall & Cook
    use the method of Monahan & Genz. I imagine that is quicker and more accurate but it is also
    more difficult to program, so I have skipped it. It remains a medium-term aim, however, because
    this method is slow.

    Params:
    D, list of 3-tuples, (disease_status, mutation_status, toxicity, efficacy), where:
                            disease_status = 0 for stage 1; 1 for stage 2,
                            mutation_status = 0 for mutation negative; 1 for mutation positive,
                            toxicity = 1 for toxic event, 0 for tolerance event,
                            and efficacy = 1 for efficacious outcome, 0 for alternative.
    priors, list of prior distributions corresponding to alpha_0, beta_0, beta_1, beta_2, psi respectively
            Each prior object should support obj.ppf(x) and obj.pdf(x)
    tox_cutoffs, list, the desired maximum toxicity cutoffs for the four groups
    eff_cutoffs, list, the desired minimum efficacy cutoffs for the four groups
    n, number of random points to use in Monte Carlo integration.

    Returns:
    nested lists of posterior probabilities, [ Prob(Toxicity, Prob(Efficacy), Prob(Toxicity less than cutoff),
                Prob(Efficacy greater than cutoff)], for each patient cohort,
            running from (disease_status, mutation_status) = (0,0), (0,1), (1,0), (1,1)
            i.e. returned obj is of length 4 and each interior list of length 4.

    """

    if not isinstance(tox_cutoffs, list):
        tox_cutoffs = [tox_cutoffs] * 4
    if not isinstance(eff_cutoffs, list):
        eff_cutoffs = [eff_cutoffs] * 4

    epsilon = 0.00001
    limits = [(dist.ppf(epsilon), dist.ppf(1-epsilon)) for dist in priors]
    samp = np.column_stack([np.random.uniform(*limit_pair, size=n) for limit_pair in limits])

    lik_integrand = lambda x: l_n(D, x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]) \
                              * priors[0].pdf(x[:, 0]) * priors[1].pdf(x[:, 1]) * priors[2].pdf(x[:, 2]) \
                              * priors[3].pdf(x[:, 3]) * priors[4].pdf(x[:, 4])
    pds = ProbabilityDensitySample(samp, lik_integrand)

    probs = []
    patient_cohorts = [(0,0), (0,1), (1,0), (1,1)]
    for (disease_status, mutation_status), tox_cutoff, eff_cutoff in zip(patient_cohorts, tox_cutoffs, eff_cutoffs):

        tox_probs = pi_t(disease_status, mutation_status, alpha0=samp[:,0])
        eff_probs = pi_e(disease_status, mutation_status, beta0=samp[:,1], beta1=samp[:,2], beta2=samp[:,3])

        # print np.mean(samp[:,1]), np.mean(samp[:,2]), np.mean(samp[:,3]), np.mean(eff_probs)

        probs.append((pds.expectation(tox_probs), pds.expectation(eff_probs),
                      pds.expectation((tox_probs < tox_cutoff)), pds.expectation((eff_probs > eff_cutoff))
                      ))

    return probs, pds


class PePs2BeBoP(Phase2EffToxBase):

    def __init__(self, theta_priors, tox_cutoffs, eff_cutoffs, tox_certainty, eff_certainty):
        """

        Params:
        theta_priors, list of prior distributions corresponding to alpha0, beta0, beta1, beta2, psi
                        respectively. Each prior object should support obj.ppf(x) and obj.pdf(x)
        tox_cutoff, scalar or list, the maximum acceptable probabilities of toxicity per group
        eff_cutoff, scalar or list, the minimium acceptable probabilities of efficacy per group
        tox_certainty, scalar or list, the posterior certainty required that toxicity is less than cutoff
        eff_certainty, scalar or list, the posterior certainty required that efficacy is greater than than cutoff

        tox_cuttoffs, eff_cutoffs, etc take the order:
        (0,0) - Not pre-treated, biomarker negative
        (0,1) - Not pre-treated, biomarker positive
        (1,0) - Pre-treated, biomarker negative
        (1,1) - Pre-treated, biomarker positive

        """

        self.priors = theta_priors
        self.tox_cutoffs = tox_cutoffs
        self.eff_cutoffs = eff_cutoffs
        self.tox_certainty = tox_certainty
        self.eff_certainty = eff_certainty

        # Initialise model
        self.reset()

    def reset(self):
        self.cases = []
        self.prob_tox = []
        self.prob_eff = []
        self.prob_acc_tox = []
        self.prob_acc_eff = []

    def update(self, cases, n=10**6, **kwargs):
        for disease_status, mutation_status, eff, tox in cases:
            self.cases.append((disease_status, mutation_status, eff, tox))

        # Update probabilities a-posteriori given observed cases
        post_probs, pds = get_posterior_probs(self.cases, self.priors, self.tox_cutoffs, self.eff_cutoffs, n)
        self._update(post_probs)
        self._pds = pds
        return 0

    def decision(self, i):
        """ Get the trial-outcome decision in group i.

        True means approve treatment.

        """

        eff_prob_hurdle = self.eff_certainty[i] if isinstance(self.eff_certainty, list) else self.eff_certainty
        tox_prob_hurdle = self.tox_certainty[i] if isinstance(self.tox_certainty, list) else self.tox_certainty
        return self.prob_acc_eff[i] >= eff_prob_hurdle and self.prob_acc_tox[i] >= tox_prob_hurdle

    def efficacy_effect(self, j, alpha=0.05):
        """ Get confidence interval and mean estimate of the effect on efficacy, expressed as odds-ratios.

        Use:
        - j=0, to get treatment effect of the intercept variable
        - j=1, to get treatment effect of the pre-treated status variable
        - j=2, to get treatment effect of the mutation status variable

        """

        if j==0:
            expected_log_or = self._pds.expectation(self._pds._samp[:,1])
            return np.exp([self._pds.quantile(1, alpha/2), expected_log_or, self._pds.quantile(1, 1-alpha/2)])
        elif j==1:
            expected_log_or = self._pds.expectation(self._pds._samp[:,2])
            return np.exp([self._pds.quantile(2, alpha/2), expected_log_or, self._pds.quantile(2, 1-alpha/2)])
        elif j==2:
            expected_log_or = self._pds.expectation(self._pds._samp[:,3])
            return np.exp([self._pds.quantile(3, alpha/2), expected_log_or, self._pds.quantile(3, 1-alpha/2)])
        else:
            return (0,0,0)

    def toxicity_effect(self, j=0, alpha=0.05):
        """ Get confidence interval and mean estimate of the effect on toxicity, expressed as odds-ratios.

        Use:
        - j=0, to get effect on toxicity of the intercept variable

        """

        if j==0:
            expected_log_or = self._pds.expectation(self._pds._samp[:,0])
            return np.exp([self._pds.quantile(0, alpha/2), expected_log_or, self._pds.quantile(0, 1-alpha/2)])
        else:
            return (0,0,0)

    def correlation_effect(self, alpha=0.05):
        """ Get confidence interval and mean estimate of the correlation between efficacy and toxicity. """
        expected_psi = self._pds.expectation(self._pds._samp[:,4])
        psi_levels = np.array([self._pds.quantile(4, alpha/2), expected_psi, self._pds.quantile(4, 1-alpha/2)])
        return (np.exp(psi_levels) - 1) / (np.exp(psi_levels) + 1)

    def _update(self, post_probs):
        """ Private method to update the model.

        Params:
        post_probs, numpy array of probabilities output from get_posterior_probs

        Returns: 0

        """

        prob_tox, prob_eff, prob_acc_tox, prob_acc_eff = zip(*post_probs)
        self.prob_tox = prob_tox
        self.prob_eff = prob_eff
        self.prob_acc_tox = prob_acc_tox
        self.prob_acc_eff = prob_acc_eff
        return 0


class PePs2BryantAndDayStage(Phase2EffToxBase):

    def __init__(self, min_efficacy_events, max_tox_events, effects_mode='ChiSqu'):

        """ A stage in a Bryant & Day trial accepts a treatment by comparing number of efficacy and tox
        events to some critical (optimal) thresholds.

        I have bolted on ways of testing treatment effects. The method of testing treatment effects is
        governed by effects_mode:
        - 'ChiSqu' to measure association by chi-squared tests;
        - 'Logit' to measure effects using joint logit GLM

        """

        self.min_efficacy_events = min_efficacy_events
        self.max_tox_events = max_tox_events
        self.effects_mode = effects_mode
        self.reset()

    def reset(self):
        self.cases = []
        self.num_eff = 0
        self.num_tox = 0

    def update(self, cases):
        for disease_status, mutation_status, tox, eff in cases:
            self.cases.append((disease_status, mutation_status, tox, eff))

        self.num_eff = sum(self.efficacies())
        self.num_tox = sum(self.toxicities())


    def decision(self, i):
        """ Get the trial-outcome decision in group i.

        True means approve treatment.

        """

        return self.num_eff >= self.min_efficacy_events and self.num_tox <= self.max_tox_events

    def efficacy_effect(self, j, alpha=0.05):
        """ Get confidence interval and mean estimate of the effect on efficacy, expressed as odds-ratios.

        Use:
        - j=0, to get treatment effect of the intercept variable
        - j=1, to get treatment effect of the pre-treated status variable
        - j=2, to get treatment effect of the mutation status variable

        """

        if self.effects_mode == 'ChiSqu':
            return self._response_effect_by_chisqu(j, alpha)
        elif self.effects_mode == 'Logit':
            return self._response_effect_by_logit(j, alpha)
        else:
            return [0,0,0]

    def toxicity_effect(self, j=0, alpha=0.05):
        """ Get confidence interval and mean estimate of the effect on toxicity, expressed as odds-ratios.

        Use:
        - j=0, to get effect on toxicity of the intercept variable

        """

        # These could be added by running the appropriate tests. But they're not required yet.
        return (0,0,0)

    def correlation_effect(self, alpha=0.05):
        """ Get confidence interval and mean estimate of the correlation between efficacy and toxicity. """
        # TODO: Confidence interval for correlation is described here
        # http://ncss.wpengine.netdna-cdn.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Interval_for_Pearsons_Correlation.pdf
        return [-1, np.corrcoef(self.toxicities(), self.efficacies())[0,1], 1]

    def _response_effect_by_chisqu(self, j, alpha):
        if j==0:
            return [0,0,0]
        elif j==1:
            test = chi_squ_test(self.pretreated_statuses(), self.efficacies(), ci_alpha=alpha)
            ci = test['Odds']['OR CI']
            return [ci[0], test['Odds']['OR'], ci[1]]
        elif j==2:
            test = chi_squ_test(self.mutation_statuses(), self.efficacies(), ci_alpha=alpha)
            ci = test['Odds']['OR CI']
            return [ci[0], test['Odds']['OR'], ci[1]]
        else:
            return (0,0,0)

    def _response_effect_by_logit(self, j, alpha):
        import statsmodels.api as sm
        if j==0:
            return [0,0,0]
        elif j==1:
            eff_logit_model = sm.Logit(self.efficacies(), pd.DataFrame({'PreTreated': self.pretreated_statuses(),
                                                                        'Mutation': self.mutation_statuses()}))
            eff_logit_model_result = eff_logit_model.fit(disp=0)
            ci = eff_logit_model_result.conf_int(alpha=alpha).loc['PreTreated'].values
            return np.exp([ci[0], eff_logit_model_result.params['PreTreated'], ci[1]])
        elif j==2:
            eff_logit_model = sm.Logit(self.efficacies(), pd.DataFrame({'PreTreated': self.pretreated_statuses(),
                                                                        'Mutation': self.mutation_statuses()}))
            eff_logit_model_result = eff_logit_model.fit(disp=0)
            ci = eff_logit_model_result.conf_int(alpha=alpha).loc['Mutation'].values
            return np.exp([ci[0], eff_logit_model_result.params['Mutation'], ci[1]])
        else:
            return [0,0,0]


def simulate_associated_eff_tox_outcomes(n, prob_eff, prob_tox, psi):
    return correlated_binary_outcomes(n, (prob_eff, prob_tox), psi)


def simulate_peps2_patients(num_patients, prob_pretreated, prob_mutated, things):
    """ Returns np.array with cols PreTreated, Mutated, Eff, Tox """
    subsample_sizes = np.random.multinomial(num_patients, [(1-prob_pretreated)*(1-prob_mutated), (1-prob_pretreated)*prob_mutated,
                                                           prob_pretreated*(1-prob_mutated), prob_pretreated*prob_mutated],
                                            size=1)[0]
    n00, n01, n10, n11 = subsample_sizes


    statuses =  n00 * [[0,0]] + n01 * [[0,1]] + n10 * [[1,0]]  + n11 * [[1, 1]]
    statuses = np.array(statuses)

    outcomes = [simulate_associated_eff_tox_outcomes(n, prob_eff, prob_tox, psi) for (n, (prob_eff, prob_tox, psi)) in
                zip(subsample_sizes, things)]
    outcomes = np.vstack(outcomes)

    to_return = np.hstack([statuses, outcomes])
    np.random.shuffle(to_return)
    # return to_return, (n11, n01, n10, n00)
    return to_return, (n00, n01, n10, n11)


def get_corr(x):
    return np.corrcoef(x[:,0], x[:,1])[0,1]


def simulate_trial(model_map, num_patients, prob_pretreated, prob_biomarker, prob_effes,
                   prob_toxes, efftox_ors, num_sims=5, alpha=0.05, log_every=10):

    decisions, pretreated_confints, mutated_confints = {}, {}, {}
    group_sizes, eff_events, tox_events = np.empty((num_sims, 4)), np.empty((num_sims, 4)), np.empty((num_sims, 4))

    for k, v in model_map.iteritems():
        decisions[k] = np.empty((num_sims, 4))
        pretreated_confints[k] = np.empty((num_sims, 3))
        mutated_confints[k] = np.empty((num_sims, 3))

    for i in range(num_sims):

        if i % log_every == 0:
            print 'Iteration', i, datetime.now()

        # Simulate patient statuses (x) and outcomes (y)
        x, n = simulate_peps2_patients(num_patients, prob_pretreated, prob_biomarker,  zip(prob_effes, prob_toxes, efftox_ors))
        df = pd.DataFrame(x, columns=['PreTreated', 'Mutated', 'Eff', 'Tox'])
        df['One'] = 1
        grouped = df.groupby(['PreTreated','Mutated'])
        # Group sizes
        sub_df = grouped['One'].agg(np.sum)

        # Eff events by group
        num_pats = [sub_df.get((0,0), default=0), sub_df.get((0,1), default=0), sub_df.get((1,0), default=0), sub_df.get((1,1), default=0)]
        group_sizes[i,:] = num_pats
        # Eff events by group
        sub_df = grouped['Eff'].agg(np.sum)
        num_effs = [sub_df.get((0,0), default=0), sub_df.get((0,1), default=0), sub_df.get((1,0), default=0), sub_df.get((1,1), default=0)]
        eff_events[i,:] = num_effs
        # Tox events by group
        sub_df = grouped['Tox'].agg(np.sum)
        num_toxs = [sub_df.get((0,0), default=0), sub_df.get((0,1), default=0), sub_df.get((1,0), default=0), sub_df.get((1,1), default=0)]
        tox_events[i,:] = num_toxs


        for k, v in model_map.iteritems():
            v.reset()
            v.update(x)
            decisions[k][i,:] = [v.decision(j) for j in range(4)]

            try:
                pretreated_confints[k][i,:] = v.efficacy_effect(1)
            except:
                pass
            try:
                mutated_confints[k][i,:] = v.efficacy_effect(2)
            except:
                pass

    # Make JSON-friendly report to return. Randomly sampled patient data plus design-specific statistical data.
    # Each trial design has a top-level map, with sub-keys for Decisions, ConfInts, Events, etc
    returnable = OrderedDict()
    # Parameters
    returnable['NumSims'] = atomic_to_json(num_sims)
    returnable['NumPatients'] = atomic_to_json(num_patients)
    returnable['Groups'] = iterable_to_json(['Not pretreated, biomarker negative', 'Not pretreated, biomarker positive',
                                             'Pretreated, biomarker negative', 'Pretreated, biomarker positive'])
    returnable['Prob(Pretreated)'] = atomic_to_json(prob_pretreated)
    returnable['Prob(Biomarker)'] = atomic_to_json(prob_biomarker)
    returnable['Prob(Efficacy)'] = iterable_to_json(prob_effes)
    returnable['Prob(Toxicity)'] = iterable_to_json(prob_toxes)
    returnable['Efficacy-Toxicity OR '] = iterable_to_json(efftox_ors)
    returnable['Alpha'] = atomic_to_json(alpha)
    # Derived parameters
    pretreated_response_prob = prob_effes[3] * prob_biomarker + prob_effes[2] * (1 - prob_biomarker)
    returnable['Prob(Response | Pretreated)'] = atomic_to_json(pretreated_response_prob)
    pretreated_response_odds = pretreated_response_prob / (1 - pretreated_response_prob)
    returnable['Odds(Response | Pretreated)'] = pretreated_response_odds
    not_pretreated_response_prob = prob_effes[1] * prob_biomarker + prob_effes[0] * (1 - prob_biomarker)
    returnable['Prob(Response | !Pretreated'] = not_pretreated_response_prob
    not_pretreated_response_odds = not_pretreated_response_prob / (1 - not_pretreated_response_prob)
    returnable['Odds(Response | !Pretreated)'] = not_pretreated_response_odds
    pretreated_response_or = pretreated_response_odds / not_pretreated_response_odds
    returnable['Response OR for Pretreated'] = atomic_to_json(pretreated_response_or)
    biomarker_pos_response_prob = prob_effes[3] * prob_pretreated + prob_effes[1] * (1 - prob_pretreated)
    returnable['Prob(Response | Biomarker)'] = atomic_to_json(biomarker_pos_response_prob)
    biomarker_pos_response_odds = biomarker_pos_response_prob / (1 - biomarker_pos_response_prob)
    returnable['Odds(Response | Biomarker)'] = atomic_to_json(biomarker_pos_response_odds)
    biomarker_neg_response_prob = prob_effes[2] * prob_pretreated + prob_effes[0] * (1 - prob_pretreated)
    returnable['Prob(Response | !Biomarker'] = atomic_to_json(biomarker_neg_response_prob)
    biomarker_neg_response_odds = biomarker_neg_response_prob / (1 - biomarker_neg_response_prob)
    returnable['Odds(Response | !Biomarker)'] = atomic_to_json(biomarker_neg_response_odds)
    biomarker_response_or = biomarker_pos_response_odds / biomarker_neg_response_odds
    returnable['Response OR for Biomarker'] = atomic_to_json(biomarker_response_or)
    # Non-method-specific data
    returnable['Group Sizes'] = [iterable_to_json(x) for x in group_sizes]
    returnable['Efficacy Events'] = [iterable_to_json(x) for x in eff_events]
    returnable['Toxicity Events'] = [iterable_to_json(x) for x in tox_events]
    # Method-specific data
    for k, v in model_map.iteritems():
        method_report = OrderedDict()
        method_report['Decisions'] = [iterable_to_json(x) for x in decisions[k]]
        method_report['Response OR for Pretreated'] = [iterable_to_json(np.round(x, 4)) for x in pretreated_confints[k]]
        method_report['Response OR for Biomarker Positive'] = [iterable_to_json(np.round(x, 4)) for x in mutated_confints[k]]
        returnable[k] = method_report

    return returnable


def tell_me_about_results(sims):
    pretreated_response_or = sims['Response OR for Pretreated']
    biomarker_response_or = sims['Response OR for Biomarker']
    n_pats = sims['NumPatients']

    print 'Events'
    print 'Efficacies:', sum(np.mean(sims['Efficacy Events'], axis=0))
    print 'Efficacy %:', sum(np.mean(sims['Efficacy Events'], axis=0)) / n_pats
    print 'Toxicities:', sum(np.mean(sims['Toxicity Events'], axis=0))
    print 'Toxicity %:', sum(np.mean(sims['Toxicity Events'], axis=0)) / n_pats
    print
    print 'By Group'
    print 'Size', sum(np.mean(sims['Group Sizes'], axis=0)), np.mean(sims['Group Sizes'], axis=0)
    print 'Efficacies:', np.mean(sims['Efficacy Events'], axis=0)
    print 'Efficacy %:', np.nanmean( np.array(sims['Efficacy Events']) / np.array(sims['Group Sizes']),  axis=0)
    print 'Toxicities:', np.mean(sims['Toxicity Events'], axis=0)
    print 'Toxicity %:', np.nanmean( np.array(sims['Toxicity Events']) / np.array(sims['Group Sizes']),  axis=0)
    print
    print 'Approve Treatment'
    print 'BD', np.mean(sims['B&D+ChiSqu']['Decisions'], axis=0)
    print 'BBB', np.mean(sims['BBB']['Decisions'], axis=0)
    print
    print 'Coverage'
    print 'True Pretreated & PD-L1 ORs:', pretreated_response_or, biomarker_response_or
    print 'BD Pretreated CI Coverage:', np.mean((pretreated_response_or > np.array(sims['B&D+ChiSqu']['Response OR for Pretreated'])[:,0]) &
            (pretreated_response_or < np.array(sims['B&D+ChiSqu']['Response OR for Pretreated'])[:,2]))
    print 'BBB Pretreated CI Coverage:', np.mean((pretreated_response_or > np.array(sims['BBB']['Response OR for Pretreated'])[:,0]) &
            (pretreated_response_or < np.array(sims['BBB']['Response OR for Pretreated'])[:,2]))
    print 'BD PD-PL+ CI Coverage:', np.mean((biomarker_response_or > np.array(sims['B&D+ChiSqu']['Response OR for Biomarker Positive'])[:,0]) &
            (biomarker_response_or < np.array(sims['B&D+ChiSqu']['Response OR for Biomarker Positive'])[:,2]))
    print 'BBB PD-PL+ CI Coverage:', np.mean((biomarker_response_or > np.array(sims['BBB']['Response OR for Biomarker Positive'])[:,0]) &
            (biomarker_response_or < np.array(sims['BBB']['Response OR for Biomarker Positive'])[:,2]))


def splice_sims(in_files_pattern, out_file=None):

    def _splice_sims(sims1, sims2):
        sims1['NumSims'] = atomic_to_json(sims1['NumSims'] + sims2['NumSims'])
        sims1['Group Sizes'] = [iterable_to_json(x) for x in np.vstack((sims1['Group Sizes'], sims2['Group Sizes']))]
        sims1['Efficacy Events'] = [iterable_to_json(x) for x in np.vstack((sims1['Efficacy Events'], sims2['Efficacy Events']))]
        sims1['Toxicity Events'] = [iterable_to_json(x) for x in np.vstack((sims1['Toxicity Events'], sims2['Toxicity Events']))]

        sims1['BBB']['Decisions'] = [iterable_to_json(x) for x in np.vstack((sims1['BBB']['Decisions'], sims2['BBB']['Decisions']))]
        sims1['BBB']['Response OR for Biomarker Positive'] = [iterable_to_json(x) for x in np.vstack((sims1['BBB']['Response OR for Biomarker Positive'],
                                                                                                      sims2['BBB']['Response OR for Biomarker Positive']))]
        sims1['BBB']['Response OR for Pretreated'] = [iterable_to_json(x) for x in np.vstack((sims1['BBB']['Response OR for Pretreated'],
                                                                                              sims2['BBB']['Response OR for Pretreated']))]

        sims1['B&D+ChiSqu']['Decisions'] = [iterable_to_json(x) for x in np.vstack((sims1['B&D+ChiSqu']['Decisions'], sims2['B&D+ChiSqu']['Decisions']))]
        sims1['B&D+ChiSqu']['Response OR for Biomarker Positive'] = [iterable_to_json(x) for x in np.vstack((sims1['B&D+ChiSqu']['Response OR for Biomarker Positive'],
                                                                                                             sims2['B&D+ChiSqu']['Response OR for Biomarker Positive']))]
        sims1['B&D+ChiSqu']['Response OR for Pretreated'] = [iterable_to_json(x) for x in np.vstack((sims1['B&D+ChiSqu']['Response OR for Pretreated'],
                                                                                                     sims2['B&D+ChiSqu']['Response OR for Pretreated']))]
        return sims1

    files = glob.glob(in_files_pattern)
    if len(files):

        sims = json.load(open(files[0], 'r'))
        print 'Fetched from', files[0]
        for f in files[1:]:
            sub_sims = json.load(open(f, 'r'))
            print 'Fetched from', f
            # Checks for homogeneity go here
            sims = _splice_sims(sims, sub_sims)

    if out_file:
        json.dump(sims, open(out_file, 'w'))
    else:
        return sims
