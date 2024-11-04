import os
import h5py as h5
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
import json, yaml
import corner
import matplotlib.lines as mlines
from scipy.special import erf, erfinv
from copy import deepcopy

line_style = {
    'Truth':'dotted',
    'Truth (Pythia)':'dotted',
    'Truth (Herwig)':'dotted',
    'SBUnfold':'-',
    'cINN':'-',
    'OmniFold (step 1)':'-',
    'Reconstructed':'-',
    'FPCD': '-',
    
}

colors = {
    'Truth':'black',
    'Truth (Pythia)':'black',
    'Truth (Herwig)':'black',
    'SBUnfold':'#7b3294',
    'cINN':'#c2a5cf',
    'OmniFold (step 1)':'#a6dba0',
    #'OmniFold (step 1) 1k':'#ff8547',
    'Reconstructed':'#008837',
    'FPCD': '#abd9e9',
}


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs


def GetEMD(ref,array,weights_arr,binning,nboot = 100):
    from scipy.stats import wasserstein_distance
    ds = []
    hists = []
    for _ in range(nboot):
        #ref_boot = np.random.choice(ref,ref.shape[0])
        arr_idx = np.random.choice(range(array.shape[0]),array.shape[0])
        array_boot = array[arr_idx]
        w_boot = weights_arr[arr_idx]
        ds.append(wasserstein_distance(ref,array_boot,v_weights=w_boot))
        hists.append(np.histogram(array_boot,weights=w_boot,bins=binning,density=True)[0])

    unc = np.std(hists,0)
    return np.mean(ds), np.std(ds),unc
    # mse = np.square(ref-array)/ref
    # return np.sum(mse)


class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')

def get_triangle_distance(x,y,x_norm,y_norm,binning,ntrials = 100):
    dist = 0
    w = binning[1:] - binning[:-1]
    for ib in range(len(x)):
        dist+=0.5*w[ib]*(x[ib]*x_norm - y[ib]*y_norm)**2/(x[ib]*x_norm + y[ib]*y_norm) if x[ib]*x_norm + y[ib]*y_norm >0 else 0.0

    x_plus = x + np.sqrt(x)
    x_minus = x - np.sqrt(x)
    y_plus = y + np.sqrt(y)
    y_minus = y - np.sqrt(y)
    
    results = []
    for trial in range(ntrials):
        x_ = np.random.uniform(low=x_minus, high=x_plus)
        y_ = np.random.uniform(low=y_minus, high=y_plus)
        d_ = 0.0
        for ib in range(len(x)):
            d_+=0.5*w[ib]*(x_[ib]*x_norm - y_[ib]*y_norm)**2/(x_[ib]*x_norm + y_[ib]*y_norm) if x_[ib]*x_norm + y_[ib]*y_norm >0 else 0.0
            results.append(d_)
        
    return dist*1e3, np.std(results)*1e3

def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Truth',logy=False,binning=None,label_loc='best',plot_ratio=True,weights=None,uncertainty=None,triangle=True):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    ref_plot = {'histtype':'stepfilled','alpha':0.2}
    other_plots = {'histtype':'step','linewidth':2}
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),30)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    reference_hist_counts,_ = np.histogram(feed_dict[reference_name],bins=binning)

    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        plot_style = ref_plot if reference_name == plot else other_plots
        if weights is not None:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,weights=weights[plot],**plot_style)
            dist_counts,_ = np.histogram(feed_dict[plot],bins=binning,weights=weights[plot])
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,**plot_style)
            dist_counts,_ = np.histogram(feed_dict[plot],bins=binning)

        if triangle:
            if reference_name==plot: continue
            print(plot)
            d,err,uncertainty = GetEMD(feed_dict[reference_name],feed_dict[plot],weights[plot],binning)
            # print("EMD distance is: {}+-{}".format(10*d,10*err))
            dist_norm = np.abs(binning[1] - binning[0])*np.sum(dist_counts)
            ref_norm = np.abs(binning[1] - binning[0])*np.sum(reference_hist_counts)
            d,err = get_triangle_distance(dist_counts,reference_hist_counts,1.0/dist_norm,1.0/ref_norm,binning)
            print("Triangular distance is: {}+-{}".format(d, err))
        else:
            uncertainty = np.zeros(reference_hist.shape)
            
        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if reference_name!=plot:
                ratio = np.ma.divide(dist,reference_hist).filled(0)
                uncertainty = np.ma.divide(uncertainty,reference_hist).filled(0)
                #ax1.plot(xaxis,ratio,color=colors[plot],marker='+',ms=8,lw=0,markerfacecolor='none',markeredgewidth=3)
                ax1.errorbar(xaxis,ratio,yerr = uncertainty,color=colors[plot],
                            marker='+',ms=8,ls='none',markerfacecolor='none',markeredgewidth=3)
                
                # if uncertainty is not None:
                #     for ibin in range(len(binning)-1):
                #         xup = binning[ibin+1]
                #         xlow = binning[ibin]
                #         ax1.fill_between(np.array([xlow,xup]),
                #                          uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')
        ax0.set_ylim(1e-3,10*maxy)
    else:
        ax0.set_ylim(0,1.3*maxy)

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Truth')
        plt.axhline(y=1.0, color='black', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([0.0,2.0])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0







def HistRoutineEnsemble(feed_dict,xlabel='',ylabel='',reference_name='Truth',logy=False,binning=None,
                        label_loc='best',plot_ratio=True,weights=None,uncertainty=None,triangle=True,feed_dict_map=None):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    ref_plot = {'histtype':'stepfilled','alpha':0.2}
    other_plots = {'histtype':'step','linewidth':2}
    fig,gs = SetGrid(ratio=plot_ratio) 
    ax0 = plt.subplot(gs[0])

    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),30)
    #binning_center = (binning[:-1] + binning[1:]) / 2

        
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]

    maxy = 0    
    for ip,plot in enumerate(feed_dict.keys()):
        plot_style = ref_plot if reference_name == plot else other_plots
        
        if feed_dict[plot].shape[-1] > 1:

            hists = []
            
            for i_ens in range(feed_dict[plot].shape[-1]):
                if weights is not None:
                    hist, _ = np.histogram(feed_dict[plot][:,i_ens], bins=binning,weights=weights[plot])
                else:
                    hist, _ = np.histogram(feed_dict[plot][:,i_ens], bins=binning)
                hists.append(hist)
            hists = np.array(hists)
            print('hists.shape', hists.shape)
            mean = np.mean(hists, 0)
            std = np.std(hists, 0)
            
            if not feed_dict_map is None:
                if weights is not None:
                    hist, _ = np.histogram(feed_dict_map[plot][:,0], bins=binning,weights=weights[plot])
                else:
                    hist, _ = np.histogram(feed_dict_map[plot][:,0], bins=binning)
                mean = (hist)

            
            
            dist = mean
            dist_counts = mean
            uncertainty = std
            
            print('dist_counts', dist_counts)
            print('uncertainty', uncertainty)
            
            
            
            feed_dict_map
            
            
            
            mean_plot =  np.concatenate((mean, np.array([0])),0) # append 0 to allign x and y arguments of step 
            std_plot =  np.concatenate((std, np.array([0])),0) # append 0 to allign x and y arguments of step 
            
            
            if reference_name == plot: 
                ax0.step(binning, mean_plot,label=plot,linestyle=line_style[plot],
                      color=colors[plot], alpha=0.2)

                ax0.fill_between(x=binning, y1=mean_plot+std_plot, y2=0,
                                 color=colors[plot], step='pre', alpha=0.2)

            else:
                ax0.step(binning, mean_plot,label=plot,linestyle=line_style[plot],
                      color=colors[plot],linewidth=2)

                ax0.fill_between(x=binning, y1=mean_plot+std_plot, y2=mean_plot-std_plot, 
                                 color=colors[plot], step='pre', alpha=0.2)


        if feed_dict[plot].shape[-1] == 1:
            if weights is not None:
                dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],
                                  color=colors[plot],density=True,weights=weights[plot],**plot_style)
                dist_counts,_ = np.histogram(feed_dict[plot],bins=binning,weights=weights[plot])
            else:
                dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],
                                  color=colors[plot],density=True,**plot_style)
                dist_counts,_ = np.histogram(feed_dict[plot],bins=binning)
                
                
        if reference_name == plot: 
            reference_hist = dist
            reference_hist_counts = dist_counts

        if triangle:
            if reference_name==plot: continue
            print(plot)
            d,err,uncertainty = GetEMD(feed_dict[reference_name],feed_dict[plot],weights[plot],binning)
            # print("EMD distance is: {}+-{}".format(10*d,10*err))
            dist_norm = np.abs(binning[1] - binning[0])*np.sum(dist_counts)
            ref_norm = np.abs(binning[1] - binning[0])*np.sum(reference_hist_counts)
            d,err = get_triangle_distance(dist_counts,reference_hist_counts,1.0/dist_norm,1.0/ref_norm,binning)
            print("Triangular distance is: {}+-{}".format(d, err))
            
        if np.max(dist) > maxy:
            maxy = np.max(dist)
            
        if plot_ratio:
            if feed_dict[plot].shape[-1] > 1:
                
                if reference_name!=plot:
                    ratio = np.ma.divide(dist,reference_hist).filled(0)
                    uncertainty = np.ma.divide(uncertainty,reference_hist).filled(0)
                    #ax1.plot(xaxis,ratio,color=colors[plot],marker='+',ms=8,lw=0,markerfacecolor='none',markeredgewidth=3)
                    ax1.errorbar(xaxis,ratio,yerr = uncertainty,color=colors[plot],
                                marker='+',ms=8,ls='none',markerfacecolor='none',markeredgewidth=3)

                
            
            if feed_dict[plot].shape[-1] == 1:

                if reference_name!=plot:
                    ratio = np.ma.divide(dist,reference_hist).filled(0)
                    uncertainty = np.ma.divide(uncertainty,reference_hist).filled(0)
                    #ax1.plot(xaxis,ratio,color=colors[plot],marker='+',ms=8,lw=0,markerfacecolor='none',markeredgewidth=3)
                    ax1.errorbar(xaxis,ratio,yerr = uncertainty,color=colors[plot],
                                marker='+',ms=8,ls='none',markerfacecolor='none',markeredgewidth=3)

                    # if uncertainty is not None:
                    #     for ibin in range(len(binning)-1):
                    #         xup = binning[ibin+1]
                    #         xlow = binning[ibin]
                    #         ax1.fill_between(np.array([xlow,xup]),
                    #                          uncertainty[ibin],-uncertainty[ibin], alpha=0.3,color='k')    
    if logy:
        ax0.set_yscale('log')
        ax0.set_ylim(1e-3,10*maxy)
    else:
        ax0.set_ylim(0,1.3*maxy)

    ax0.legend(loc=label_loc,fontsize=16,ncol=2)
    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 
        plt.ylabel('Ratio to Truth')
        plt.axhline(y=1.0, color='black', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([0.0,2.0])
        plt.xlabel(xlabel)
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0) 
        
    return fig,ax0











def DataLoader(sample_name,
               N_t=1000000,N_v=600000,
               cache_dir="/global/cfs/cdirs/m3929/I2SB/",json_path='JSON'):
    import energyflow as ef
    datasets = {sample_name: ef.zjets_delphes.load(sample_name, num_data=N_t+N_v,
                                                   cache_dir=cache_dir,exclude_keys=['particles'])}
    feature_names = ['widths','mults','sdms','zgs','tau2s']
    gen_features = [datasets[sample_name]['gen_jets'][:,3]]
    sim_features = [datasets[sample_name]['sim_jets'][:,3]]

    for feature in feature_names:
        gen_features.append(datasets[sample_name]['gen_'+feature])
        sim_features.append(datasets[sample_name]['sim_'+feature])

    gen_features = np.stack(gen_features,-1)
    sim_features = np.stack(sim_features,-1)
    #ln rho
    gen_features[:,3] = 2*np.ma.log(np.ma.divide(gen_features[:,3],datasets[sample_name]['gen_jets'][:,0]).filled(0)).filled(0)
    sim_features[:,3] = 2*np.ma.log(np.ma.divide(sim_features[:,3],datasets[sample_name]['sim_jets'][:,0]).filled(0)).filled(0)
    #tau2
    gen_features[:,5] = gen_features[:,5]/(10**-50 + gen_features[:,1])
    sim_features[:,5] = sim_features[:,5]/(10**-50 + sim_features[:,1])

    #Standardize
    gen_features = ApplyPreprocessing(gen_features,'gen_features.json',json_path)
    sim_features = ApplyPreprocessing(sim_features,'sim_features.json',json_path)

    train_gen = gen_features[:N_t]
    train_sim = sim_features[:N_t]
    
    test_gen = gen_features[N_t:]
    test_sim = sim_features[N_t:]

    return train_gen, train_sim, test_gen,test_sim
    


def DataLoaderZ2Jet(
    sample_name, 
    N_t=1500000,N_v=400000,
    cache_dir="/global/cfs/cdirs/m3929/I2SB/",json_path='JSON',
    test_name=None,
    skip_preprocessing=False,
    noise_preprocessing=False,
    noise_muon=False,
    basic_preprocessing=False
):

    gen = np.load(cache_dir+"Z_2j_Gen.npy")
    rec = np.load(cache_dir+"Z_2j_Sim.npy")

    gen_train = np.array(gen[:N_t])
    rec_train = np.array(rec[:N_t])
    gen_test = np.array(gen[N_t:N_t+N_v])
    rec_test = np.array(rec[N_t:N_t+N_v])

    if basic_preprocessing:
        json_name='_Z_2Jet_basic'
    else:
        json_name='_Z_2Jet'

    CalcPreprocessingZ2Jet(
        gen_train,rec_train,
        'features' + json_name + '.json', 
        json_path,
        basic_preprocessing=basic_preprocessing
    )

    gen_train, rec_train = ApplyPreprocessingZ2Jet(
        gen_train, 
        rec_train,
        'features' + json_name + '.json', 
        json_path, 
        noise_preprocessing=noise_preprocessing,
        noise_muon=noise_muon,
        basic_preprocessing=basic_preprocessing
    )
    gen_test, rec_test = ApplyPreprocessingZ2Jet(
        gen_test, 
        rec_test, 
        'features' + json_name + '.json', 
        json_path,
        noise_preprocessing=False, 
        noise_muon=False,
        basic_preprocessing=basic_preprocessing
    )

    print('gen_train.shape', gen_train.shape)
    print('rec_train.shape', rec_train.shape)
    print('gen_test.shape', gen_test.shape)
    print('rec_test.shape', rec_test.shape)

    return gen_train, rec_train, gen_test, rec_test

    
def DataLoaderComparison(sample_name,
               N_t=1000000,N_v=600000,
               cache_dir="/global/cfs/cdirs/m3929/I2SB/",json_path='JSON', 
               test_name='OmniFold_test.h5',
               skip_preprocessing=False):
    
    if sample_name=='OmniFold_train_small.h5':
        json_name='_comparison_small'
        
        
    if sample_name=='OmniFold_train_large.h5':
        json_name='_comparison_large'

    #frame = pd.read_hdf(load_dir + file, mode='r')
    f_train = h5.File(cache_dir + sample_name, 'r')
    gen_features_train = f_train['hard'][:]
    sim_features_train = f_train['reco'][:]
    
    f_test = h5.File(cache_dir + test_name, 'r')
    gen_features_test = f_test['hard'][:]
    sim_features_test = f_test['reco'][:]
    
    # ln rho
    #gen_features[:,3] = 2*np.ma.log(gen_features[:,3]).filled(0)
    #sim_features[:,3] = 2*np.ma.log(sim_features[:,3]).filled(0)
    # tau2
    #gen_features[:,5] = gen_features[:,5]/(10**-50 + gen_features[:,1])
    #sim_features[:,5] = sim_features[:,5]/(10**-50 + sim_features[:,1])

    if skip_preprocessing:
        train_gen = gen_features_train
        train_sim = sim_features_train

        test_gen = gen_features_test
        test_sim = sim_features_test

        return train_gen, train_sim, test_gen, test_sim
    
    CalcPreprocessingComparison(gen_features_train,'gen_features1' + json_name + '.json',json_path)
    CalcPreprocessingComparison(sim_features_train,'sim_features1' + json_name + '.json',json_path)

    
    gen_features_train = ApplyPreprocessingComparison(gen_features_train,'gen_features1' + json_name + '.json',json_path)
    sim_features_train = ApplyPreprocessingComparison(sim_features_train,'sim_features1' + json_name + '.json',json_path)
    
    print(np.mean(gen_features_train==gen_features_train))
    print(np.mean(sim_features_train==sim_features_train))
    
    CalcPreprocessing(gen_features_train,'gen_features2' + json_name + '.json',json_path)
    CalcPreprocessing(sim_features_train,'sim_features2' + json_name + '.json',json_path)

    gen_features_train = ApplyPreprocessing(gen_features_train,'gen_features2' + json_name + '.json',json_path)
    sim_features_train = ApplyPreprocessing(sim_features_train,'sim_features2' + json_name + '.json',json_path)

    
    

    
    gen_features_test = ApplyPreprocessingComparison(gen_features_test,'gen_features1' + json_name + '.json',json_path)
    sim_features_test = ApplyPreprocessingComparison(sim_features_test,'sim_features1' + json_name + '.json',json_path)
    
    gen_features_test = ApplyPreprocessing(gen_features_test,'gen_features2' + json_name + '.json',json_path)
    sim_features_test = ApplyPreprocessing(sim_features_test,'sim_features2' + json_name + '.json',json_path)

    train_gen = gen_features_train
    train_sim = sim_features_train
    
    test_gen = gen_features_test
    test_sim = sim_features_test

    return train_gen, train_sim, test_gen, test_sim





def DataLoaderToy(sample_name,
               N_t=1,N_v=1,
               cache_dir="/global/cfs/cdirs/m3929/I2SB/",json_path='', 
               test_name=''):


    f_t = h5.File(cache_dir + sample_name, 'r')
    train_origin = f_t['train_origin'][:]
    train_target = f_t['train_target'][:]

    f_t = h5.File(cache_dir + sample_name, 'r')
    test_origin = f_t['test_origin'][:]
    test_target = f_t['test_target'][:]

    return train_origin, train_target, test_origin, test_target



def pT_eta_phi_m_2_E_px_py_pz(pT_eta_phi_m):
    pt = pT_eta_phi_m[..., 0]
    eta = pT_eta_phi_m[..., 1]
    phi = pT_eta_phi_m[..., 2]
    m = pT_eta_phi_m[..., 3]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    return np.stack([E, px, py, pz], axis=-1)

def E_px_py_pz_2_pT_eta_phi_m(E_px_py_pz):
    E = E_px_py_pz[..., 0]
    px = E_px_py_pz[..., 1]
    py = E_px_py_pz[..., 2]
    pz = E_px_py_pz[..., 3]

    pt = np.sqrt(px**2 + py**2)
    eta = np.arctanh(pz / np.sqrt(px ** 2 + py ** 2 + pz ** 2))
    phi = np.arctan2(py, px)
    m = np.sqrt(E ** 2 - px**2 - py**2 - pz**2)
    return np.stack([pt, eta, phi, m], axis=-1)

def invariant_mass(*particles):
    particles_Eppp = [pT_eta_phi_m_2_E_px_py_pz(p) for p in particles]
    particles_sum = sum(particles_Eppp)
    return E_px_py_pz_2_pT_eta_phi_m(particles_sum)[..., -1]

def breit_wigner_forward(events, peak_position, width):
    z1 = 1 / np.pi * np.arctan((events - peak_position) / width) + 0.5
    return np.sqrt(2) * erfinv(2 * z1 - 1)

def breit_wigner_reverse(events, peak_position, width):
    a = events/np.sqrt(2)
    a = erf(a)
    a = 0.5*(a+1)
    a = a -0.5
    a = a*np.pi
    a = np.tan(a)
    a = a*width + peak_position
    return a
     
def CalcPreprocessingZ2Jet(gen,rec,fname,base_folder,basic_preprocessing=False):
    '''Apply data preprocessing'''
    
    gen=deepcopy(gen)
    rec=deepcopy(rec)
    
    if not basic_preprocessing:

        muon_mass = np.zeros((len(gen), ))#.to(self.device)

        muon1_gen_pT_eta_phi_m = np.stack([
            deepcopy(gen[:, 0]),
            deepcopy(gen[:, 1]),
            deepcopy(gen[:, 2]),
            deepcopy(muon_mass)
        ], axis=-1)
        muon2_gen_pT_eta_phi_m = np.stack([
            deepcopy(gen[:, 3]),
            deepcopy(gen[:, 4]),
            deepcopy(gen[:, 5]),
            deepcopy(muon_mass)
        ], axis=-1)
        
        dimuon_mass_gen = invariant_mass(muon1_gen_pT_eta_phi_m, muon2_gen_pT_eta_phi_m)
        dimuon_mass_gen = breit_wigner_forward(dimuon_mass_gen, peak_position=91, width=1)
        
        gen[:, 0] = deepcopy(dimuon_mass_gen)
    
        muon1_rec_pT_eta_phi_m = np.stack([
            deepcopy(rec[:, 0]),
            deepcopy(rec[:, 1]),
            deepcopy(rec[:, 2]),
            deepcopy(muon_mass)
        ], axis=-1)
        muon2_rec_pT_eta_phi_m = np.stack([
            deepcopy(rec[:, 3]),
            deepcopy(rec[:, 4]),
            deepcopy(rec[:, 5]),
            deepcopy(muon_mass)
        ], axis=-1)
        
        dimuon_mass_rec = invariant_mass(muon1_rec_pT_eta_phi_m, muon2_rec_pT_eta_phi_m)
        dimuon_mass_rec = breit_wigner_forward(dimuon_mass_rec, peak_position=91, width=1)
        rec[:, 0] = deepcopy(dimuon_mass_rec)

    
    
    data_dict = {}
    mean_gen = np.average(gen,axis=0)
    std_gen = np.std(gen,axis=0)
    mean_rec = np.average(rec,axis=0)
    std_rec = np.std(rec,axis=0)
    data_dict['mean_gen']=mean_gen.tolist()
    data_dict['std_gen']=std_gen.tolist()
    data_dict['mean_rec']=mean_rec.tolist()
    data_dict['std_rec']=std_rec.tolist()
    SaveJson(fname,data_dict,base_folder)
    
    del gen
    del rec


def ApplyPreprocessingZ2Jet(gen,rec,fname,base_folder, noise_preprocessing=False, noise_muon=False, basic_preprocessing=False):
    gen=deepcopy(gen)
    rec=deepcopy(rec)
    data_dict = LoadJson(fname,base_folder)

    if not basic_preprocessing:

        muon_mass = np.zeros((len(gen), ))#.to(self.device)

        muon1_gen_pT_eta_phi_m = np.stack([
            deepcopy(gen[:, 0]),
            deepcopy(gen[:, 1]),
            deepcopy(gen[:, 2]),
            deepcopy(muon_mass)
        ], axis=-1)
        muon2_gen_pT_eta_phi_m = np.stack([
            deepcopy(gen[:, 3]),
            deepcopy(gen[:, 4]),
            deepcopy(gen[:, 5]),
            deepcopy(muon_mass)
        ], axis=-1)
        
        dimuon_mass_gen = invariant_mass(muon1_gen_pT_eta_phi_m, muon2_gen_pT_eta_phi_m)
        dimuon_mass_gen = breit_wigner_forward(dimuon_mass_gen, peak_position=91, width=1)
    
        gen[:, 0] = deepcopy(dimuon_mass_gen)    

        muon1_rec_pT_eta_phi_m = np.stack([
            deepcopy(rec[:, 0]),
            deepcopy(rec[:, 1]),
            deepcopy(rec[:, 2]),
            deepcopy(muon_mass)
        ], axis=-1)
        muon2_rec_pT_eta_phi_m = np.stack([
            deepcopy(rec[:, 3]),
            deepcopy(rec[:, 4]),
            deepcopy(rec[:, 5]),
            deepcopy(muon_mass)
        ], axis=-1)
        
        dimuon_mass_rec = invariant_mass(muon1_rec_pT_eta_phi_m, muon2_rec_pT_eta_phi_m)
        dimuon_mass_rec = breit_wigner_forward(dimuon_mass_rec, peak_position=91, width=1)
        rec[:, 0] = deepcopy(dimuon_mass_rec)

    # standardize events
    mean_rec = data_dict['mean_rec']
    std_rec = data_dict['std_rec']
    mean_gen = data_dict['mean_gen']
    std_gen = data_dict['std_gen']

    gen = (gen - mean_gen)/std_gen
    rec = (rec - mean_rec)/std_rec

    if noise_preprocessing:
        gen = gen+np.random.normal(0, 0.001, gen.shape)
        rec = rec+np.random.normal(0, 0.001, gen.shape)

    if noise_muon:
        gen[:, 1] = gen[:, 1]+np.random.normal(0, 0.05, gen[:, 1].shape)
        gen[:, 2] = gen[:, 2]+np.random.normal(0, 0.05, gen[:, 2].shape)

        gen[:, 4] = gen[:, 4]+np.random.normal(0, 0.05, gen[:, 4].shape)
        gen[:, 5] = gen[:, 5]+np.random.normal(0, 0.05, gen[:, 5].shape)

    return gen,rec



def ReversePreprocessingZ2Jet(gen,rec,fname,base_folder, noise_preprocessing=False, noise_muon=False, basic_preprocessing=False):

    gen=deepcopy(gen)
    rec=deepcopy(rec)

    data_dict = LoadJson(fname,base_folder)
    
    # standardize events
    mean_rec = data_dict['mean_rec']
    std_rec = data_dict['std_rec']
    mean_gen = data_dict['mean_gen']
    std_gen = data_dict['std_gen']

    gen = gen*std_gen + mean_gen
    rec = rec*std_rec + mean_rec

    if noise_muon:
        close_mask1 = np.abs(gen[:, 1] - rec[:, 1]) < 0.25
        gen[close_mask, 1] = rec[close_mask, 1]
        close_mask2 = np.abs(gen[:, 2] - rec[:, 2]) < 0.25
        gen[close_mask, 2] = rec[close_mask, 2]
        close_mask4 = np.abs(gen[:, 4] - rec[:, 4]) < 0.25
        gen[close_mask, 4] = rec[close_mask, 4]
        close_mask5 = np.abs(gen[:, 5] - rec[:, 5]) < 0.25
        gen[close_mask, 5] = rec[close_mask, 5]

    if noise_preprocessing:
        close_mask = np.abs(gen - rec) < 0.01
        gen[close_mask] = rec[close_mask]
    
    if not basic_preprocessing:
        dimuon_mass_gen = breit_wigner_reverse(deepcopy(gen[:, 0]), peak_position=91, width=1)
        muon1_eta_gen = deepcopy(gen[:, 1])
        muon1_phi_gen = deepcopy(gen[:, 2])
        muon2_pt_gen  = deepcopy(gen[:, 3])
        muon2_eta_gen = deepcopy(gen[:, 4])
        muon2_phi_gen = deepcopy(gen[:, 5])
        muon1_pt_gen = dimuon_mass_gen**2 / (
            2 * muon2_pt_gen * (np.cosh(muon1_eta_gen - muon2_eta_gen) - np.cos(
                muon1_phi_gen - muon2_phi_gen)))
        
        gen[:, 0] = deepcopy(muon1_pt_gen)

        dimuon_mass_rec = breit_wigner_reverse(deepcopy(rec[:, 0]), peak_position=91, width=1)
        muon1_eta_rec = deepcopy(rec[:, 1])
        muon1_phi_rec = deepcopy(rec[:, 2])
        muon2_pt_rec  = deepcopy(rec[:, 3])
        muon2_eta_rec = deepcopy(rec[:, 4])
        muon2_phi_rec = deepcopy(rec[:, 5])
        muon1_pt_rec = dimuon_mass_rec ** 2 / (
            2 * muon2_pt_rec * (np.cosh(muon1_eta_rec - muon2_eta_rec) - np.cos(
                muon1_phi_rec - muon2_phi_rec)))
        
        rec[:, 0] = deepcopy(muon1_pt_rec)

    return gen,rec

    
def CalcPreprocessing(data,fname,base_folder):
    '''Apply data preprocessing'''
    
    data_dict = {}
    mean = np.average(data,axis=0)
    std = np.std(data,axis=0)
    data_dict['mean']=mean.tolist()
    data_dict['std']=std.tolist()
    data_dict['min']=np.min(data,0).tolist()
    data_dict['max']=np.max(data,0).tolist()    
    SaveJson(fname,data_dict,base_folder)


def ApplyPreprocessing(data,fname,base_folder):
    #CalcPreprocessing(data,fname,base_folder)    
    data_dict = LoadJson(fname,base_folder)
    data = (np.ma.divide((data-data_dict['mean']),data_dict['std']).filled(0)).astype(np.float32)
    #data = (np.ma.divide((data-data_dict['min']),np.array(data_dict['max']) - data_dict['min']).filled(0)).astype(np.float32)
    return data


def ReversePreprocessing(data,fname,base_folder):
    data_dict = LoadJson(fname,base_folder)
    #data = (np.array(data_dict['max']) - data_dict['min']) * data + data_dict['min']
    data = data * data_dict['std'] + data_dict['mean']
    data[:,2] = np.round(data[:,2]) #particle multiplicity should be an integer
    return data


def CalcPreprocessingComparison(data,fname,base_folder):
    '''Apply data preprocessing'''
    
    data_dict = {}
    
    z = data.copy()
    z4 = z[:, 4]
    noise = np.random.rand(*z4.shape)/1000. * 3 + 0.097
    z4 = np.where(z4 < 0.1, noise, z4)
    z4 = np.log(z4)
    shift = (np.max(z4) + np.min(z4))/2.
    z4 = z4-shift
    factor = max(np.max(z4), -1 * np.min(z4))*1.001
    
    data_dict['shift']=shift.tolist()
    data_dict['factor']=factor.tolist()
    SaveJson(fname,data_dict,base_folder)

def ApplyPreprocessingComparison(data,fname,base_folder):
    channels = 2
    z = data.copy()
    noise = np.random.rand(*z[:, channels].shape)-0.5
    z[:, channels] = z[:,channels] + noise

    data_dict = LoadJson(fname,base_folder)

    z4 = z[:, 4]
    noise = np.random.rand(*z4.shape)/1000. * 3 + 0.097
    z4 = np.where(z4 < 0.1, noise, z4)
    z4 = np.log(z4)
    z4 = z4-data_dict['shift']
    z4 = z4/data_dict['factor']
    z4 = sp.special.erfinv(z4)
    z[:, 4] = z4
    
    return z


def ReversePreprocessingComparison(data,fname,base_folder):    
    channels = 2
    z = data.copy()
    z[:, channels] = np.round(z[:, channels])    
    
    data_dict = LoadJson(fname,base_folder)
    
    z4 = z[:, 4]
    z4 = sp.special.erf(z4)
    z4 = z4*data_dict['factor']
    z4 = z4+data_dict['shift']
    z4 = np.exp(z4)
    z4 = np.where(z4 < 0.1, 0, z4)
    z[:, 4] = z4
    
    return z


def SaveJson(save_file,data,base_folder='JSON'):
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    
    with open(os.path.join(base_folder,save_file),'w') as f:
        json.dump(data, f)

    
def LoadJson(file_name,base_folder='JSON'):
    import json,yaml
    JSONPATH = os.path.join(base_folder,file_name)
    return yaml.safe_load(open(JSONPATH))


def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)

def overlaid_corner(samples_list, sample_labels,name=''):
    """Plots multiple corners on top of each other"""

    CORNER_KWARGS = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=16),
        title_kwargs=dict(fontsize=16),
        #quantiles=[0.16, 0.84],
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
        plot_density=False,
        plot_datapoints=False,
        fill_contours=True,
        #show_titles=True,
        max_n_ticks=3
    )

    
    # get some constants
    n = len(samples_list)
    _, ndim = samples_list[0].shape
    max_len = max([len(s) for s in samples_list])
    cmap = plt.cm.get_cmap('gist_rainbow', n)
    colors = ["black","red","blue"]

    plot_range = []
    for dim in range(ndim):
        plot_range.append(
            [
                min([min(samples_list[i].T[dim]) for i in range(n)]),
                max([max(samples_list[i].T[dim]) for i in range(n)]),
            ]
        )
    plot_range = [[3,70],[0.,0.6],[1.0,70],[-13,-3],[0,0.5],[0.1,1.2]]

    CORNER_KWARGS.update(range=plot_range)

    fig = corner.corner(
        samples_list[0],
        color=colors[0],
        labels = ["Jet Mass [GeV]","Jet Width", "$n_{constituents}$",r"$ln\rho$","$z_g$",r"$\tau_{21}$"],
        **CORNER_KWARGS
    )

    for idx in range(1, n):
        fig = corner.corner(
            samples_list[idx],
            fig=fig,
            weights=get_normalisation_weight(len(samples_list[idx]), max_len),
            color=colors[idx],
            **CORNER_KWARGS
        )

    plt.legend(
        handles=[
            mlines.Line2D([], [], color=colors[i], label=sample_labels[i])
            for i in range(n)
        ],
        fontsize=24, frameon=False,
        bbox_to_anchor=(1, ndim), loc="upper right"
    )
    plt.savefig("plots/corner_{}.pdf".format(name))
    #plt.close()


        
