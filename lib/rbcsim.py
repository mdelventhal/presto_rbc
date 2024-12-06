import warnings
import numpy as np
import pandas as pd
import altair as alt
from lib.dataprocess import make_detrended_rbc_sample

def newt_meth(f,x0,fprime=None,derivstep=1e-5,thresh=1e-15,maxits=1000,damper=1.,overshootretries=10):
    diff = float('inf')
    its = 0

    if fprime is None:
        fprime = lambda x: (f(x+derivstep) -f(x))/derivstep

    while (diff > thresh) and (its < maxits):
        curdamper = float(damper)
        its_retry = 0
        while its_retry < overshootretries:
            x = x0 - curdamper*f(x0)/fprime(x0)
            curdamper *= .5
            its_retry += 1
            if (x == x) and (f(x) == f(x)) and (abs(x) != float('inf')) and (abs(f(x)) != float('inf')):
                break
        diff = abs((x - x0)/x0)
        its += 1
        x0 = float(x)

    if (its >= maxits) and (diff > thresh):
        warnings.warn('Reached maximum iterations without converging.')
    #print(its)
    return x

def rbc_labsupply_objfunc(l,psi,alph,delt,A,k,kprime):
    term1 = l*(psi/(1-alph) + 1)
    term2 = (l**alph)*(psi/(1-alph))*(k*(1-delt) - kprime)/(A*k**alph)
    term3 = -1
    return term1 + term2 + term3

def rbc_labsupply_objfunc_fd(l,psi,alph,delt,A,k,kprime):
    term1 = (psi/(1-alph) + 1)
    term2 = alph*(l**(alph-1))*(psi/(1-alph))*(k*(1-delt) - kprime)/(A*k**alph)
    return term1 + term2

def rbc_labsupply(psi,alph,delt,A,k,kprime,x0=.5):
    return newt_meth(lambda x: rbc_labsupply_objfunc(x,psi,alph,delt,A,k,kprime),x0,
                     fprime=lambda x: rbc_labsupply_objfunc_fd(x,psi,alph,delt,A,k,kprime))


def kprime_find(kbar,k,A,Abar,aak,aaA):
    ktil = np.log(k/kbar)
    Atil = np.log(A/Abar)
    return kbar*np.exp(aak*ktil + aaA*Atil)

def rbc_labsupply_endogkprime(psi,alph,delt,A,k,
                            kbar,Abar,aak,aaA,
                            x0=.5):
    kprime = kprime_find(kbar,k,A,Abar,aak,aaA)
    return newt_meth(lambda x: rbc_labsupply_objfunc(x,psi,alph,delt,A,k,kprime),x0,
                     fprime=lambda x: rbc_labsupply_objfunc_fd(x,psi,alph,delt,A,k,kprime))

def rbc_psi_endogkprime(lbar,alph,delt,A,k,
                            kbar,Abar,aak,aaA,
                            x0=.5):
    kprime = kprime_find(kbar,k,A,Abar,aak,aaA)
    return newt_meth(lambda x: rbc_labsupply_objfunc(lbar,x,alph,delt,A,k,kprime),x0,
                     fprime=lambda x: rbc_labsupply_objfunc_fd(lbar,x,alph,delt,A,k,kprime))


def A_iter(A,shock,rho,sig):
    return np.exp(rho*np.log(A) + (sig**2)*shock)

def A_path(A0,periods,rho,sig,Abar=1,shocks=None,return_shocks=False):
    if shocks is None:
        shocks = np.random.randn(periods)
    A_out = [A0]
    for shock in shocks:
        A_out.append(Abar*A_iter(A_out[-1]/Abar,shock,rho,sig))
    if return_shocks:
        return A_out,shocks
    else:
        return A_out

def RBC_fluct_path(A0,k0,periods,rho,sig,
                   kbar,Abar,aak,aaA,
                   psi,alph,delt,shocks=None,return_shocks=False
                  ):
    if return_shocks:
        A_out,shocks = A_path(A0,periods,rho,sig,Abar=Abar,shocks=shocks,return_shocks=return_shocks)
    else:
        A_out = A_path(A0,periods,rho,sig,Abar=Abar,shocks=shocks,return_shocks=return_shocks)
    k_out = [k0]
    for A in A_out[1:]:
        k_out.append(kprime_find(kbar,k_out[-1],A,Abar,aak,aaA))

    l_out =[rbc_labsupply_endogkprime(psi,alph,delt,A,k,
                                       kbar,Abar,aak,aaA) for A,k in zip(A_out,k_out)]
    df_out = pd.DataFrame({'k':k_out,'A':A_out,'l':l_out})
    df_out['y'] = df_out['A']*(df_out['k']**alph)*(df_out['l']**(1-alph))
    df_out['c'] = df_out['y'] + df_out['k']*(1-delt) - df_out['k'].shift(-1)
    df_out['u'] = np.log(df_out['c']) + psi*np.log(1-df_out['l'])
    if return_shocks:
        return df_out,shocks
    else:
        return df_out

class RBC_model():
    INPUT_PARAM_LIST = ['delt','sbar','alph','aak','aaA','rho','sig']
    DF_IN_COLUMNS = ['date','y','k','A','l','k_stat','y_stat','A_stat']
    def __init__(self,
                 df_in,
                 params_dict,
                 periodicity=4
                ):

        self.df_raw = df_in
        self.periodicity = periodicity

        self.update_params(params_dict)

        self.last_sim_df = None

        self.raw_A_shocks = None


    def update_params(self,params_dict):
        self.params = {}
        for key in self.INPUT_PARAM_LIST:
            self.params[key] = params_dict[key]

        df_detrend,self.g_y = make_detrended_rbc_sample(self.df_raw,self.params['alph'],return_gy=True,periodicity=self.periodicity)
        self.df_in = df_detrend[self.DF_IN_COLUMNS].copy()

        self.kbar = self.df_in['k_stat'].mean()
        self.lbar = self.df_in['l'].mean()
        self.Abar = (self.params['delt'])/(self.params['sbar']*(self.kbar**(self.params['alph']-1))*((self.lbar)**(1-self.params['alph'])))

        self.psi = rbc_psi_endogkprime(self.lbar,
                                       self.params['alph'],
                                       self.params['delt'],
                                       self.Abar,
                                       self.kbar,
                                       self.kbar,self.Abar,
                                       self.params['aak'],self.params['aaA'])

    def simulate_model(self,new_shocks=False):
        if new_shocks:
            self.raw_A_shocks = None
        df_sim,self.raw_A_shocks = RBC_fluct_path(self.Abar,self.kbar,len(self.df_in)-1,
                                                  self.params['rho'],self.params['sig'],
                                                  self.kbar,self.Abar,self.params['aak'],self.params['aaA'],
                                                  self.psi,self.params['alph'],self.params['delt'],
                                                  shocks=self.raw_A_shocks,return_shocks=True
                                                 )
        df_sim['date'] = self.df_in['date']
        df_sim['y_stat'] = df_sim['y']/df_sim['y'].mean()
        df_sim['k_stat'] = df_sim['k']/df_sim['k'].mean()
        df_sim['A_stat'] = df_sim['A']/df_sim['A'].mean()

        #df_sim['log_y_stat'] = np.log(df_sim['y_stat'])
        #df_sim['log_k_stat'] = np.log(df_sim['k_stat'])
        #df_sim['log_A_stat'] = np.log(df_sim['A_stat'])
        #df_sim['log_l'] = np.log(df_sim['l'])
        #df_sim['log_l_stat'] = np.log(df_sim['l']/df_sim['l'].mean())

        self.last_df_sim = df_sim
        return df_sim

    def sim_deviations_chart(self,vars_to_include={'y_stat':'output/worker',
                                                   'k_stat':'capital/worker',
                                                   'l':'labor supply'},width=None,height=None):
        return self.deviations_chart(self.last_df_sim,vars_to_include=vars_to_include,width=width,height=height)

    def data_deviations_chart(self,vars_to_include={'y_stat':'output/worker',
                                                   'k_stat':'capital/worker',
                                                   'l':'labor supply'},width=None,height=None):
        return self.deviations_chart(self.df_in,vars_to_include=vars_to_include,width=width,height=height)

    def deviations_chart(self,df,vars_to_include={'y_stat':'output/worker',
                                                   'k_stat':'capital/worker',
                                                   'l':'labor supply'},width=None,height=None):
        df_sim_altair = df[['date'] + [x for x in vars_to_include]].copy()

        for var in vars_to_include:
            df_sim_altair[var] = np.log(df_sim_altair[var]/df_sim_altair[var].mean())

        chartkwargs = {}
        if width is not None:
            chartkwargs['width'] = width
        if height is not None:
            chartkwargs['height'] = height
        chart = alt.Chart(df_sim_altair.rename(columns=vars_to_include).melt(id_vars='date',
                                                                             value_vars=[vars_to_include[x] for x in vars_to_include],
                                                                             value_name='log deviations from BGP'),
                          **chartkwargs)

        return chart.mark_line().encode(x='date:T',y='log deviations from BGP:Q',color=alt.Color('variable:N',sort='descending'),)\
                    + chart.mark_rule(strokeDash=[4, 4]).encode(y=alt.datum(0))

    def get_sigma_data(self,var):
        return self.get_sigma(self.df_in,var)

    def get_sigma_sim(self,var):
        return self.get_sigma(self.last_df_sim,var)

    def get_sigma(self,df,var):
        return np.log(df[var]/df[var].mean()).std()

    def plot_data_variable(self,var,with_trend=False,label=None):
        return self.plot_variable(self.df_in,var,with_trend=with_trend,label=label)

    def plot_sim_variable(self,var,with_trend=False,label=None):
        return self.plot_variable(self.last_df_sim,var,with_trend=with_trend,label=label)

    def plot_variable(self,df,var,with_trend=False,label=None):
        to_chart_df = df[['date',var]].copy()

        to_chart_df['trend'] = 0
        if var[-5:] == '_stat':
            if label is None:
                label = var[0:-5]
            if with_trend:
                to_chart_df['trend'] += (to_chart_df.index-to_chart_df.index[0])*self.g_y/4

            to_chart_df[var] = np.log(to_chart_df[var]/to_chart_df[var].mean()) + to_chart_df['trend']
        else:
            to_chart_df['trend'] += to_chart_df[var].mean()

        chart = alt.Chart(to_chart_df,width=600,height=240)
        return chart.mark_line().encode(x='date:T',
                                        y=alt.Y(f'{var}:Q',scale=alt.Scale(zero=False),title=label))\
                        + chart.mark_line(color='black',strokeDash=[4, 4]).encode(x='date:T',
                                                                              y='trend:Q')
