import streamlit as st
import copy
import pandas as pd
import numpy as np

from lib.dataprocess import get_us_macro_data_for_rbcdemo,make_detrended_rbc_sample
from lib.rbcsim import RBC_model

@st.cache_data(ttl='1d')
def get_base_data():
    return get_us_macro_data_for_rbcdemo(refresh=True)

base_data_df = get_base_data()

with st.expander('Parameter controls:'):
    basic_params_cols = st.columns(2)
    shock_process_cols = st.columns(2)
    loglinear_savings_cols = st.columns(2)
    timepref_cols = st.columns(2)

with basic_params_cols[0]:
    alph = st.number_input('Capital share of production ($\\alpha$):',max_value=0.99,min_value=0.01,value=0.33)
with basic_params_cols[1]:
    delt = st.number_input('Capital depreciation rate ($\delta$, quarterly):',max_value=0.99,min_value=0.01,value=0.015,format="%.3f")

with shock_process_cols[0]:
    rho = st.number_input('TFP shock persistence ($\\rho$):',max_value=0.99,min_value=-.99,value=0.9)
with shock_process_cols[1]:
    sig = st.number_input('TFP shock variance ($\sigma$):',max_value=10.,min_value=0.01,value=0.099)

with loglinear_savings_cols[0]:
    aak = st.number_input('Log-linear savings response to capital ($a_{kk}$):',max_value=0.99,min_value=-0.99,value=0.95)
with loglinear_savings_cols[1]:
    aaA = st.number_input('Log-linear savings response to TFP ($a_{kA}$):',max_value=0.99,min_value=-0.99,value=0.08)

#with timepref_cols[0]:
#    bet = st.number_input('Time preference ($\\beta$):',max_value=0.99,min_value=0.01,value=0.95)

params_dict = dict(delt = delt,
                   sbar = .2,
                   alph = alph,
                   aak = aak,
                   aaA = aaA,
                   rho = rho,
                   sig = sig)

if 'params_dict' not in st.session_state:
    st.session_state['params_dict'] = copy.copy(params_dict)

if 'rbc_model' not in st.session_state:
    st.session_state['rbc_model'] = RBC_model(base_data_df,params_dict)
    st.session_state['rbc_model'].simulate_model()

equivalence_check = min(st.session_state['params_dict'][param] == params_dict[param] for param in st.session_state['params_dict'])

new_shocks = False
if st.button('Draw new sequence of TFP shocks'):
    new_shocks = True

if (not equivalence_check) or new_shocks:
    st.session_state['params_dict'] = copy.copy(params_dict)
    st.session_state['rbc_model'].update_params(st.session_state['params_dict'])
    st.session_state['rbc_model'].simulate_model(new_shocks=new_shocks)


#U_realized = np.nansum(st.session_state['rbc_model'].last_df_sim['u']*(bet**(st.session_state['rbc_model'].last_df_sim.index - st.session_state['rbc_model'].last_df_sim.index[0])))
#st.write(f"Realized discounted utility: {U_realized:.4f}")

#main_chart_cols = st.columns(2)

#with main_chart_cols[0]:
st.write('Data fluctuations:')

data_sig_y = 100*st.session_state['rbc_model'].get_sigma_data('y_stat')
data_sig_l_over_y = st.session_state['rbc_model'].get_sigma_data('l')/st.session_state['rbc_model'].get_sigma_data('y_stat')
st.markdown(f"""
| Std. dev., output/worker ($\sigma_y$): | Ratio of std. devs., hours/(output/worker): ($\\frac{{\sigma_l}}{{\sigma_y}}$) |
| --- | --- |
| {data_sig_y:.2f} | {data_sig_l_over_y:.2f} |
""")


st.altair_chart(st.session_state['rbc_model'].data_deviations_chart(width=600,height=300))

#with main_chart_cols[1]:
st.write('Simulated fluctuations:')


sim_sig_y = 100*st.session_state['rbc_model'].get_sigma_sim('y_stat')
sim_sig_l_over_y = st.session_state['rbc_model'].get_sigma_sim('l')/st.session_state['rbc_model'].get_sigma_sim('y_stat')
st.markdown(f"""
| Std. dev., output/worker ($\sigma_y$): | Ratio of std. devs., hours/(output/worker): ($\\frac{{\sigma_l}}{{\sigma_y}}$) |
| --- | --- |
| {sim_sig_y:.2f} | {sim_sig_l_over_y:.2f} |
""")
st.altair_chart(st.session_state['rbc_model'].sim_deviations_chart(width=600,height=300))

var_option_dict = {'A':'A_stat',
                   'k':'k_stat',
                   'l':'l',
                   'y':'y_stat'}
with st.expander('Individual variable charts:'):
    optioncols = st.columns(3)
    data_or_sim = optioncols[0].selectbox('Show data or simulation?',['Data','Simulation'])
    var_to_chart = optioncols[1].selectbox('Choose a variable',list(var_option_dict.keys()))
    no_trend = optioncols[2].checkbox('Stationary?')

    if data_or_sim == 'Data':
        st.altair_chart(st.session_state['rbc_model'].plot_data_variable(var_option_dict[var_to_chart],with_trend=not no_trend))
    if data_or_sim == 'Simulation':
        st.altair_chart(st.session_state['rbc_model'].plot_sim_variable(var_option_dict[var_to_chart],with_trend=not no_trend))


#st.altair_chart(st.session_state['rbc_model'].sim_deviations_chart())
