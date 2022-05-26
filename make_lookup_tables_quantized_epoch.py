from __future__ import division

import numpy as np
import os, glob, json
import matplotlib.pyplot as plt
import pickle
import scipy.linalg as sl
import scipy.integrate as spint
import healpy as hp
import multiprocessing as mp
import math

os.environ["TEMPO2"]='/home/nima/.local/share/tempo2/'

from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from enterprise.signals import utils
from enterprise.signals import gp_signals
from enterprise import constants as const
from enterprise.signals.signal_base import LogLikelihood
from enterprise_extensions import models as ee_models
from enterprise_extensions import model_utils as ee_model_utils
from enterprise_extensions import sampler as ee_sampler
from enterprise_extensions import blocks as ee_blocks
from enterprise_extensions import deterministic

from la_forge.core import Core, load_Core
from la_forge import rednoise
from la_forge.diagnostics import plot_chains

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

def make_lookup_table(psr, noisefile, outdir, sign, log10_rn_amps, log10_rn_amp_spacing,
                        log10_bwm_amps, log10_bwm_amp_spacing,
                        gammas, gamma_spacing, Ts, time_spacing, full_12p5_tspan_sec=407576851.48121357):

    if not os.path.exists(outdir + psr.name):
        os.mkdir(outdir + psr.name)

    #now we need to make a pta for this pulsar to look up likelihoods for each amplitude we calculate
    #################
    ####   PTA   ####
    #################

    # we want to introduce a fixed-index common process in addition to the usual ramp, so we'll make this by hand

    """
    This is now outdated since we want a common process
    pta = ee_models.model_ramp([psr], LogLikelihood,
                          upper_limit=False, bayesephem=False,
                          Tmin_bwm=t0min, Tmax_bwm=t0max, logmin=min(log10_amps), logmax=max(log10_amps))
    """

    #expect Ts to be passed in units of seconds
    U,_ = utils.create_quantization_matrix(psr.toas)
    eps = 9  # clip first and last N observing epochs

    tmin_mjd = np.floor(max(U[:,eps] * psr.toas/24/3600))
    tmax_mjd = np.ceil(max(U[:,-eps] * psr.toas/24/3600))

    print("I was handed tmin: {} and tmax: {}".format(tmin_sec, tmax_sec))
    print("Expected these times to be in units of seconds")

    Tspan_sec = (tmax_mjd-tmin_mjd) * 24 * 3600
    Tspan_mjd = tmax_mjd - tmin_mjd

    print("Tspan_sec: {}".format(Tspan_sec))
    IRN_logmin = min(log10_rn_amps)
    IRN_logmax = max(log10_rn_amps)


    bwm_logmin = min(log10_bwm_amps)
    bwm_logmax = max(log10_bwm_amps)

    full_12p5_tspan_mjd = full_12p5_tspan_sec/3600/24

    #Intrinsic Red Noise
    s = ee_blocks.red_noise_block(psd='powerlaw', prior='log-uniform', components=30,
                                 logmin=IRN_logmin, logmax=IRN_logmax, Tspan=None)
    # Common Red Noise
    # Jury is still out on which Tspan to use
    # I think this needs to reflect what the Tspan will be at runtime
    # which ought to be the Tspan of the entire PTA, even if it overparameterizes
    # the individual PTA Tspan.
    #print("creating cRN with tspan = {}".format(full_12p5_tspan_sec))
    # s += ee_blocks.common_red_noise_block(psd='powerlaw', prior='log-uniform',
    #                                     Tspan=full_12p5_tspan_sec, components=30, gamma_val=13./3.,
    #                                     logmin=CRN_logmin, logmax=CRN_logmax)


    # Add a ramp to the data
    print("creating ramp block with tmax, tmin: {} , {}\n and amplitudes {} , {}".format(tmin_mjd, tmax_mjd, bwm_logmin, bwm_logmax))
    s += ee_blocks.bwm_sglpsr_block(Tmin=tmin_mjd, Tmax=tmax_mjd, amp_prior='log-uniform',
                                 logmin=bwm_logmin, logmax=bwm_logmax,
                                 fixed_sign=None)

    s += gp_signals.TimingModel()

    if 'NANOGrav' in psr.flags['pta']:
        s += ee_blocks.white_noise_block(vary=False, inc_ecorr=True)
    else:
        s += ee_blocks.white_noise_block(vary=False, inc_ecorr=False)

    models = []
    models.append(s(psr))

    pta = signal_base.PTA(models)

    print("Here are the parameters of the pta: {}".format(pta.params))
    with open(noisefile, 'rb') as nfile:
        setpars = json.load(nfile)

    pta.set_default_params(setpars)

    with open(outdir + "{}/{}_{}.txt".format(psr.name, psr.name, sign),'a+') as f:
        for t0 in Ts:
            #since t0 needs to be in mjd, we need to convert...
            t0_mjd = t0/3600/24
            for log10_strain in log10_bwm_amps:
                this_l10rn_gamma_chart = np.zeros((len(log10_rn_amps), len(gammas)))
                # set up the sky location of the pulsar
                # ramp_amp*=sign
                # print(psr.name + "would see an amplitude of: " + str(ramp_amp))

                # Now we need to add the A_red and gamma_red params so that we have in total:
                # A_red, Gamma_red, A_ramp, t_ramp
                # Since we want to marginalize over the individual RN, we'll only write the
                # likelihood after summing over all the RN parameters
                for ii, log10_rn_amp in enumerate(log10_rn_amps):
                    for jj, gamma in enumerate(gammas):
                        #now we have the four parameters, we need to ask the pta to calculate a likelihood
                        #the pta params are in the order:
                        this_l10rn_gamma_chart[ii,jj] = pta.get_lnlikelihood([gamma, log10_rn_amp, log10_strain, t0_mjd, sign])
                # now that we have the log likelihoods, we need to add them


                # just to make sure that some large numbers aren't breaking things, we're going to do this in 2 steps
                compressed_chart = np.zeros(len(log10_rn_amps))
                for ii, l10A in enumerate(log10_rn_amps):
                    normed_margin_like = 0
                    submtx = this_l10rn_gamma_chart[ii,:]
                    maxloglike = np.amax(submtx)
                    # used simpsons rule to marginalize over gamma
                    gam_post = np.exp(submtx - maxloglike)
                    compressed_chart[ii] = np.log(spint.simpson(gam_post))+ maxloglike

                # now we need to use simpsons to integrate the compressed charts
                # since this integral is over the log of the amplitudes, we need to put in that
                # prior-like term to make sure we integrate out over uniform amplitudes

                # Each term in the compresed_chart is now the likelihood marginalized over rn gamma
                compressed_max = np.amax(compressed_chart)
                corrected_amp_post = np.exp(compressed_chart-compressed_max)

                for ii, amp in enumerate(log10_rn_amps):
                    corrected_amp_post[ii] = 10**amp * corrected_amp_post[ii]


                lnlike = np.log(spint.simpson(corrected_amp_post)) + compressed_max



                if lnlike > 0:
                    f.write('{:.15e}\n'.format(float(lnlike)))
                else:
                    f.write('{:.14e}\n'.format(float(lnlike)))

pkl_path = '/home/nima/nanograv/11yr_factlike/NANOGrav_11yr_DE436.pickle'
lookup_outdir = '/home/nima/nanograv/11yr_factlike/lookup_tables_final'
noisefile = '/home/nima/nanograv/11yr_factlike/noisefiles/noisedict.json'



## Load the pulsar

with open(pkl_path, 'rb') as f:
    allpsrs=pickle.load(f)

psrlist = [p.name for p in allpsrs]

if not os.path.exists(lookup_outdir):
    os.mkdir(lookup_outdir)

for psr in allpsrs:
    psrname = psr.name
    if psrname in psrlist:
    #assert psrname[0:5] == "B1855"
    ## Build grid spacing
        iRN_amp_spacing = '-17,-11,60'
        iRN_log10_amps = np.linspace(-17, -11, 60, endpoint=True)

        bwm_amp_spacing = '-17,-10,70'
        bwm_log10_amps = np.linspace(-17, -10, 70, endpoint=True)

        gamma_spacing ='0,7,28'
        gammas = np.linspace(0, 7, 28, endpoint=True)

        #t0min_sec = psr.toas.min()
        #t0max_sec = psr.toas.max()


        #U,_ = utils.create_quantization_matrix(psr.toas)
        #eps = 9  # clip first and last N observing epochs

        # t0min_sec = np.floor(max(U[:,eps] * psr.toas))
        # t0max_sec = np.ceil(max(U[:,-eps] * psr.toas))

        t0min_sec = psr.toas.min() + 180*3600*24
        t0max_sec = psr.toas.max() - 180*3600*24

        t0min_mjd = t0min_sec/3600/24
        t0max_mjd = t0max_sec/3600/24



        # we're actually going to extend the t0s to a little before and after the pulsar's observing baseline
        # hopefully, this is enough to fix our problem with early and late times

        tspan_sec = t0max_sec - t0min_sec
        tspan_mjd = (t0max_mjd - t0min_mjd)
        tspan_months = tspan_mjd/30

        print("For PSR {} I got {} months of data".format(psr.name, tspan_months))

        epoch_steps = int(np.floor(tspan_months))

        #we're going to pass in the times as seconds, and the worker function will process the input
        Ts = np.linspace(t0min_sec, t0max_sec, num=epoch_steps, endpoint=True) # These probably need to reflect the change above in 192/193
        time_spacing = '{},{},{}'.format(t0min_mjd, t0max_mjd, epoch_steps)
        sign_spacing = '-1,1,2'


        ## Some bookkeeping

        if not os.path.exists(lookup_outdir + psr.name):
            os.mkdir(lookup_outdir + psr.name)


        with open(lookup_outdir+'{}/pars.txt'.format(psr.name), 'w+') as f:
            f.write('{};{}\n{};{}\n{};{}'.format( 'ramp_log10_A',bwm_amp_spacing, 'ramp_t0',time_spacing,'sign', sign_spacing))

        ## Let it rip! We're doing the signs in parallel to speed things up, and we'll just add them back up
        ## atoothe end.

        #psr, noisefile, outdir, sign, log10_rn_amps, log10_rn_amp_spacing,
        #                        log10_cRN_amps, log10_crn_amp_spacing, log10_bwm_amps, log10_bwm_amp_spacing,
        #                        gammas, gamma_spacing, Ts, time_spacing,

        params1=[psr, noisefile, lookup_outdir, 1, iRN_log10_amps, iRN_amp_spacing, bwm_log10_amps, bwm_amp_spacing, gammas, gamma_spacing, Ts, time_spacing]
        params2=[psr, noisefile, lookup_outdir, -1, iRN_log10_amps, iRN_amp_spacing, bwm_log10_amps, bwm_amp_spacing, gammas, gamma_spacing, Ts, time_spacing]

        pool = mp.Pool(2)
        pool.starmap(make_lookup_table, [params1, params2])
