import matplotlib

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np


from typing import *
import pandas as pd
import seaborn as sns
import math

sns.set()


import yaml
from scipy.stats import norm



class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()



class CertifiedRate(Accuracy):
    def __init__(self, smoothed_fname,agg_weight=None,M=0,alpha= 0):
        cert_bound, cert_bound_exp, is_acc = certify(smoothed_fname,agg_weight=agg_weight,M=M,alpha= alpha)
        self.cert_bound = cert_bound
        self.cert_bound_exp = cert_bound_exp
        self.is_acc = is_acc

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        return np.array([self.at_radius(radius) for radius in radii])

    def at_radius(self, radius: float):
        return (self.cert_bound  >= radius).mean()

class CertifiedAcc(Accuracy):
    def __init__(self, smoothed_fname, agg_weight=None,M=0,alpha= 0):
        cert_bound, cert_bound_exp, is_acc = certify(smoothed_fname,agg_weight=agg_weight, M=M,alpha= alpha)
        self.cert_bound = cert_bound
        self.cert_bound_exp = cert_bound_exp
        self.is_acc = is_acc

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        return np.array([self.at_radius(radius) for radius in radii])

    def at_radius(self, radius: float):
        return (np.logical_and(self.cert_bound>=radius, self.is_acc)).mean()


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.001) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_rate(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.001) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified rate", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
   
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def cal_prob_bound(pa, pb, sigma_test,epoch, training_params,agg_weight=None):
    sigma_train = training_params['sigma_param']
    sigma_test= sigma_test
    eta = training_params['lr'] # lr 
    T = epoch # epoch 
    N =  training_params['no_models']  # number of clients
    R = len(training_params['adversary_list'])  # number of R
    q_B = training_params['poisoning_per_batch']  # poison per batch
    n_B = training_params['batch_size']  # poison per batch
    gamma= training_params['scale_factor'] #scale
    tau =  int(28628/n_B)
    rho_tadv= 2.15 # attack at round 6
    L_z =  math.sqrt(2+2*rho_tadv+rho_tadv**2)
    if agg_weight==None:
        agg_weight= []
        for i in range(0,R):
            agg_weight.append(float(1/N))
    weighted_avg =0   
    for i in range(0,R):
        weighted_avg+= agg_weight[i]**2
    t_adv= training_params['poison_epochs'][0]
    if pa==1.0:
        return 100000
    fraction= - math.log(1- (math.sqrt(pa)-math.sqrt(pb))**2) * sigma_train**2
    denominator= 2* R* tau**2 *L_z**2 * weighted_avg * gamma**2 * eta**2 * float(q_B**2 / n_B**2 ) 
    contract=1
    for _epoch in range(t_adv+1, T): # from round t_adv+1 to round T-1 
        rho_t = _epoch *0.025+2
        contract *= 2*norm.cdf(rho_t*1.0/sigma_train)-1 
    rho_T = T *0.025+2
    contract  *= (2*norm.cdf(rho_T*1.0/sigma_test)-1) # round T
    denominator= denominator * contract
    delta_pat  = math.sqrt(fraction/  denominator)

    return delta_pat 


def certify(smoothed_fname,agg_weight=None, M=0, alpha= 0):

    foldername= smoothed_fname.split('/')
    epoch = int(foldername[-1].split('_')[-1])

    foldername = os.path.join(foldername[0],foldername[1])

    training_param_fname= os.path.join(foldername,'params.yaml')
    with open(training_param_fname, 'r') as f:
        training_params = yaml.load(f)
    print(training_params)

    if M==0:
        M= params_loaded['N_m']
    if alpha==0:
        alpha = params_loaded['alpha']
    
    # data_file_path =  os.path.join(foldername, "pred_poison_Epoch%dM%dSigma%.4f.txt"%(epoch,params_loaded['N_m'], params_loaded['test_sigma']))
    data_file_path =  os.path.join(foldername, "pred_clean_Epoch%dM%dSigma%.4f.txt"%(epoch,M, params_loaded['test_sigma']))

    num_samples= 10000
    df = pd.read_csv(data_file_path, delimiter="\t")
    print(len(np.array(df["pa_exp"])))
    pa_exp = np.array(df["pa_exp"])[:num_samples]
    pb_exp = np.array(df["pb_exp"])[:num_samples]
    is_acc = np.array(df["is_acc"])[:num_samples]

    heof_factor = np.sqrt(np.log(1/alpha)/2/M)
    pa = np.maximum(1e-8, pa_exp - heof_factor) # [num_samples]
    pb = np.minimum(1-1e-8, pb_exp + heof_factor) # [num_samples]

    # Calculate the metrics
    cert_bound= np.zeros_like(pa)
    cert_bound_exp = np.zeros_like(pa)
    for i in range(len(pa)):
        cert_bound[i]  = cal_prob_bound(pa=pa[i], pb=pb[i],sigma_test=params_loaded['test_sigma'], epoch=epoch, training_params=training_params,agg_weight=agg_weight )
        cert_bound_exp[i]  = cal_prob_bound(pa=pa_exp[i], pb=pb_exp[i],sigma_test=params_loaded['test_sigma'], epoch=epoch, training_params =training_params,agg_weight=agg_weight )
    return cert_bound, cert_bound_exp, is_acc






if __name__ == "__main__":
    with open(f'./configs/loan_smooth_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
  

    #  ### vary N  
    # plot_certified_rate(
    #     "plots/loan/vary_N_tadv6_T100_cer_rate", "vary N ($t_{adv}=6$, T=100, R=1, $\gamma=10$)", 6, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.40.57/model_last.pt.tar.epoch_100"), "N = 10"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "N = 20"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.41.10/model_last.pt.tar.epoch_100"), "N = 30"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.41.18/model_last.pt.tar.epoch_100"), "N = 40"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.41.25/model_last.pt.tar.epoch_100"), "N = 50"),

    #     ])
    # plot_certified_accuracy(
    #      "plots/loan/vary_N_tadv6_T100_cer_acc", "vary N ($t_{adv}=6$, T=100, R=1, $\gamma=10$)", 6, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.40.57/model_last.pt.tar.epoch_100"), "N = 10"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "N = 20"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.41.10/model_last.pt.tar.epoch_100"), "N = 30"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.41.18/model_last.pt.tar.epoch_100"), "N = 40"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.41.25/model_last.pt.tar.epoch_100"), "N = 50"),
    #     ])

    # #  #### noise
    # plot_certified_rate(
    #     "plots/loan/vary_sigma_T100_cer_rate", "vary $\sigma$ ($t_{adv}=6$, T=100, $\gamma=10$, R=1)", 6, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.36/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.005"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.010"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.44/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.015"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.50/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.020"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.56/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.025"),
    #     ])
    # plot_certified_accuracy(
    #     "plots/loan/vary_sigma_T100_cer_acc", "vary $\sigma$ ($t_{adv}=6$, T=30, $\gamma=100$, R=1)", 6, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.36/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.005"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.010"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.44/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.015"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.50/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.020"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.56/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.025"),
    #     ])

    
    
    #### vary T 
    # plot_certified_rate(
    #     "plots/loan/vary_T_tadv6_cer_rate", "vary T ($t_{adv}=6$, R=1, $\gamma=10$)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_30"), "T = 30"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_50"), "T = 50"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_60"), "T = 60"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_80"), "T = 80"),
    #      
            
    #     ])
    # plot_certified_accuracy(
    #     "plots/loan/vary_T_tadv6_cer_acc", "vary T ($t_{adv}=6$, R=1, $\gamma=10$)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_30"), "T = 30"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_50"), "T = 50"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_60"), "T = 60"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_80"), "T = 80"),
    #   
    #     ])
    

    # #### vary R 
    # plot_certified_rate(
    #     "plots/loan/vary_R_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " R = 1"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.38.30/model_last.pt.tar.epoch_100"), " R = 2"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.38.40/model_last.pt.tar.epoch_100"), " R = 3"),
    #     ])

    # plot_certified_accuracy(
    #     "plots/loan/vary_R_tadv6_T100_cer_acc", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " R = 1"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.38.30/model_last.pt.tar.epoch_100"), " R = 2"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.38.40/model_last.pt.tar.epoch_100"), " R = 3"),
    #     ])

    # # robust RFA 
    # plot_certified_accuracy(
    # "plots/loan/vary_agg_tadv6_T100_cer_acc", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])
    # plot_certified_rate(
    # "plots/loan/vary_agg_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])

    # # gammma
    # plot_certified_accuracy(
    #     "plots/loan/vary_gamma_tadv6_T100_cer_acc", "vary $\gamma$ ($t_{adv}=6$, T=100, R=1)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "$\gamma$ = 10"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.43.48/model_last.pt.tar.epoch_100"), "$\gamma$ = 20"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.43.59/model_last.pt.tar.epoch_100"), "$\gamma$ = 30"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.44.06/model_last.pt.tar.epoch_100"), "$\gamma$ = 50"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.44.29/model_last.pt.tar.epoch_100"), "$\gamma$ = 100"),
    #     ])
    
    
    # plot_certified_rate(
    #     "plots/loan/vary_gamma_tadv6_T100_cer_rate", "vary $\gamma$ ($t_{adv}=6$, T=100, R=1)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "$\gamma$ = 10"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.43.48/model_last.pt.tar.epoch_100"), "$\gamma$ = 20"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.43.59/model_last.pt.tar.epoch_100"), "$\gamma$ = 30"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.44.06/model_last.pt.tar.epoch_100"), "$\gamma$ = 50"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.44.29/model_last.pt.tar.epoch_100"), "$\gamma$ = 100"),
    #     ])
 
    ##### t_adv 
    # plot_certified_rate(
    #     "plots/loan/vary_tadv_T100_cer_rate", "vary $t_{adv}$ ($\gamma=10$, T=100, R=1)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 6"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.18/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 20"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.26/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 40"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.34/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 60"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.42/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 80"),
    #     ])

    # plot_certified_accuracy(
    #      "plots/loan/vary_tadv_T100_cer_acc", "vary $t_{adv}$ ($\gamma=10$, T=100, R=1)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 6"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.18/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 20"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.26/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 40"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.34/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 60"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.42/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 80"),
    #     ])

    # plot_certified_accuracy(
    #     "plots/loan/vary_tadv_T50_cer_acc", "vary $t_{adv}$ ($\gamma=100$, T=50, R=2)", 0.15, [
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.36.31/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 20"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.37.23/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 40"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.37.38/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 43"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.38.48/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 45"),
    #     ])


    

    ##### poison_ratio 
    # plot_certified_rate(
    #     "plots/loan/vary_qn_T100_cer_rate", "vary $q_{B_i}/n_{B_i}$ ($\gamma=10$, $t_{adv}=6$, T=100, R=1)", 5.0, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.45.58/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 2.5%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 5%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.46.15/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 10%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.46.22/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 15%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.46.29/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 20%"),
    #     ])

    # plot_certified_accuracy(
    #      "plots/loan/vary_qn_T100_cer_acc", "vary $q_{B_i}/n_{B_i}$ ($\gamma=10$, $t_{adv}=6$, T=100, R=1)", 5.0, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.45.58/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 2.5%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 5%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.46.15/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 10%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.46.22/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 15%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.46.29/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 20%"),
    #     ])



    # ### vary M
    # plot_certified_accuracy(
    #     "plots/loan/vary_M_T45_cer_acc", "vary M ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=100), " M = 100"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=500), " M = 500"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=1000), " M = 1000"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=2000), " M = 2000"),
    #     ]
    #     )

    # plot_certified_rate(
    #     "plots/loan/vary_M_T45_cer_rate", "vary M ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=100), " M = 100"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=500), " M = 500"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=1000), " M = 1000"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=2000), " M = 2000"),
    #     ]
    # )



    # ### vary alpha
    # plot_certified_accuracy(
    #     "plots/loan/vary_alpha_T45_cer_acc", "vary $alpha$ ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.01), " 99% confidence"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.001), " 99.9% confidence"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.0001), " 99.99% confidence"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.00001), " 99.999% confidence"),
    #     ]
    #     )
    # plot_certified_rate(
    #     "plots/loan/vary_alpha_T45_cer_rate", "vary $alpha$ ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.01), " 99% confidence"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.001), " 99.9% confidence"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.0001), " 99.99% confidence"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.00001), " 99.999% confidence"),
    #     ]
    #     )
