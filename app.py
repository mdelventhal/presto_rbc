import streamlit as st
import copy
import pandas as pd
import numpy as np

from lib.dataprocess import get_us_macro_data_for_rbcdemo,make_detrended_rbc_sample
from lib.rbcsim import RBC_model

default_parameters = dict(delt = .015,
                   sbar = .2,
                   alph = .33,
                   aak = .95,
                   aaA = .08,
                   rho = .9,
                   sig = .1)
def reset_to_defaults():
    for param in default_parameters:
        st.session_state[param] = default_parameters[param]

@st.cache_data(ttl='1d')
def get_base_data():
    return get_us_macro_data_for_rbcdemo(refresh=True,fred_api_key=st.secrets['FRED_KEY'])

base_data_df = get_base_data()

the_tabs = st.tabs(["Interactive model","Theory","Data notes"])

with the_tabs[0]:
    st.markdown("""A simulation of a very basic Real Business Cycles model.
 - Compare the detrended raw data to the simulation results!
 - Open the "parameters" panel to try a different set of parameters.
 - You can also click the button to draw a new sequence of random productivity shocks!
 - Click on the "Theory" tab for details on the underlying theory, or "Data notes" for more info on sources.
""")

    with st.expander('Parameter controls:'):
        basic_params_cols = st.columns(2)
        shock_process_cols = st.columns(2)
        loglinear_savings_cols = st.columns(2)
        timepref_cols = st.columns(2)
        st.button('Reset to default parameters', on_click=reset_to_defaults)



    with basic_params_cols[0]:
        #alph =
        st.number_input('Capital share of production ($\\alpha$):',max_value=0.99,min_value=0.01,key='alph') #,value=0.33)
    with basic_params_cols[1]:
        #delt =
        st.number_input('Capital depreciation rate ($\delta$, quarterly):',max_value=0.99,min_value=0.01,key='delt',format="%.3f") #,value=0.015,format="%.3f")

    with shock_process_cols[0]:
        #rho =
        st.number_input('TFP shock persistence ($\\rho$):',max_value=0.99,min_value=-.99,key='rho') #,value=0.9)
    with shock_process_cols[1]:
        #sig =
        st.number_input('TFP shock variance ($\sigma$):',max_value=10.,min_value=0.01,key='sig') #,value=0.099)

    with loglinear_savings_cols[0]:
        #aak =
        st.number_input('Log-linear savings response to capital ($a_{kk}$):',max_value=0.99,min_value=-0.99,key='aak') #,value=0.95)
    with loglinear_savings_cols[1]:
        #aaA =
        st.number_input('Log-linear savings response to TFP ($a_{kA}$):',max_value=0.99,min_value=-0.99,key='aaA') #,value=0.08)

    #with timepref_cols[0]:
    #    bet = st.number_input('Time preference ($\\beta$):',max_value=0.99,min_value=0.01,value=0.95)

    params_dict = dict(delt = st.session_state.delt,
                       sbar = .2,
                       alph = st.session_state.alph,
                       aak = st.session_state.aak,
                       aaA = st.session_state.aaA,
                       rho = st.session_state.rho,
                       sig = st.session_state.sig)

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


with the_tabs[1]:
    st.markdown(r"""A discrete-time adaptation of the model presented in Chapter 5 of David Romer's *Advanced Macroeconomics, 4th Edition*. Another possibly useful exposition can be found [in these slides from Jesus Fernandez-Villaverde](https://www.sas.upenn.edu/~jesusfv/lecture12_rbc.pdf).

### Households
At each point in time the economy is populated by $N_t$ identical households with 1 member each who own capital and are endowed with 1 unit of labor. They rent their capital $k_t$ for return $r_t$ and rent a fraction $l_t$ of their labor for wage $w_t$. They value consumption $c_t$ and leisure $1-l_t$.

The problem of a representative household can be written thus:""")
    st.latex(r"""
\begin{align*}&\max\limits_{c_t,l_t,k_{t+1}} \left \{\sum\limits_{t=0}^{\infty} \beta^t \left [\ln c_t + \psi \ln \left (1-l_t \right ) \right ]\right \} \\
& \quad \quad \quad \quad \quad \quad \quad \text{ such that } \\
& \quad \quad \quad c_t + k_{t+1} \leq w_t l_t + r_t k_t + (1-\delta) k_t
\end{align*}
""")
    st.markdown(r"""...where $\beta \in (0,1)$ represents time prefence, $\delta \in [0,1]$ represents the depreciation rate of capital, and $\psi>0$ determines the relative weight households place on leisure versus the consumption of goods and services.""")
    st.markdown(r"""### Firms
The economy is populated by a large number of identical firms who seek to maximize profits by combining capital and labor which they rent at rates $r_t$ and $w_t$, which the firms take as given. The aggregate quantity of capital in the economy at each point in time is equal to $N_t k_t = K_t$, and the aggregate labor supply is equal to $N_t l_t = L_t$.

All firms share an identical Cobb-Douglas production function with TFP $A_t > 0$ and capital share $\alpha \in (0,1)$ such that aggregate production $Y_t$ and market equilibrium capital return $r_t$ and wage $w_t$ can be characterized by the following:
""")
    st.latex(r"""
\begin{align*}Y_t &= A_t K_t^{\alpha} L_t^{1-\alpha}\\
\\
r_t &= \alpha A_t k_t^{\alpha-1} l_t^{1-\alpha}\\
\\
w_t &= \alpha A_t k_t^{\alpha} l_t^{-\alpha}
\end{align*}
""")
    st.markdown(r"""Output per capita $y_t$ can then be characterized as
    $$y_t = A_t k_t^{\alpha} l_t^{1-\alpha}$$
""")
    st.markdown(r"""### Stochastic TFP
TFP $A_t$ is made up of a deterministic component which grows at constant rate $g_A$ and a stochastic component $\tilde{A}_t$ such that
""")
    st.latex(r"""
\begin{align*}\ln A_t = \ln A_0 + (1 + g_A) \cdot t + \ln \tilde{A}_t
\end{align*}
""")
    st.markdown(r"""The evolution of $\tilde{A}_t$ is governed according to a first-order auto-regressive process:
""")
    st.latex(r"""
\begin{align*}\ln \tilde{A}_t = \rho \ln \tilde{A}_{t-1} + \varepsilon_t
\end{align*}
""")
    st.markdown(r"""...where $\rho \in [-1,1]$ governs the strength of shock persistence and $\varepsilon_t$ represents a series of independent random draws from a normal distribution with mean 0 and standard deviation $\sigma$.
""")
    st.markdown(r"""### Detrending
Define detrended variables as follows:
""")
    st.latex(r"""
\begin{align*}\tilde{k}_t &= \frac{k_t}{\left (A_0 (1+g_A)^t \right )^{\frac{1}{1-\alpha}}}\\
\\
\tilde{c}_t &= \frac{c_t}{\left (A_0 (1+g_A)^t \right )^{\frac{1}{1-\alpha}}}\\
\\
\tilde{y}_t &= \frac{y_t}{\left (A_0 (1+g_A)^t \right )^{\frac{1}{1-\alpha}}}\\
\end{align*}
""")
    st.markdown(r"""### Solution

#### Leisure-consumption tradeoff

The household's optimal tradeoff between consumption and leisure for $\alpha \in (0,1)$ is characterized by
""")
    st.latex(r"""
\begin{align*}l_t \left (\frac{\psi}{1-\alpha} + 1\right ) + l_t^{\alpha} \frac{\psi}{1-\alpha} \frac{k_t (1-\delta) - k_{t+1}}{A_t k_t^{\alpha}} - 1 = 0
\end{align*}
""")
    st.markdown(r"""Taking the choice of $k_{t+1}$ as given, the choice of labor supply $l_t$ can be solved for numerically-or in the special case of $\alpha = \frac{1}{2}$, analytically.

The de-trended equivalent of the above condition is""")
    st.latex(r"""
\begin{align*}l_t \left (\frac{\psi}{1-\alpha} + 1\right ) + l_t^{\alpha} \frac{\psi}{1-\alpha} \frac{\tilde{k}_t (1-\delta) - (1+g_A)^{\frac{-1}{1-\alpha}}\tilde{k}_{t+1}}{\tilde{A}_t \tilde{k}_t^{\alpha}} - 1 = 0
\end{align*}
""")
    st.markdown("""#### Savings decision

As a shortcut, a log-linearized function is used to determine $k_{t+1}$, such that
""")
    st.latex(r"""
\begin{align*}\ln \tilde{k}_{t+1} = a_{kA} \ln \tilde{A}_t + a_{kk} \ln \tilde{k}_t
\end{align*}
""")

with the_tabs[2]:
    st.markdown(r"""Data are at quarterly frequency and downloaded through the Federal Reserve Economic Data (FRED) API.

Specific series identifiers are as follows:
 - **nominal output**: [`GDP`](https://fred.stlouisfed.org/series/GDP) *($ P Y$)*
 - **nominal capital stock**: [`K1TTOTL1ES000`](https://fred.stlouisfed.org/series/K1TTOTL1ES000) *($ P K$)*
 - **price index**: [`GDPDEF`](https://fred.stlouisfed.org/series/GDPDEF) *($ P $)*
 - **hours worked**: [`HOANBS`](https://fred.stlouisfed.org/series/HOANBS) *($ \frac{L}{\tilde{L}} $)*
 - **population/labor force**: [`LFAC64TTUSQ647S`](https://fred.stlouisfed.org/series/LFAC64TTUSQ647S) *($ N $)*

Raw data for the nominal capital stock is at annual frequency and so the natural log is linearly interpolated across quarters.

The following calculations are then performed:
 - $Y = \frac{P Y}{P}$
 - $K = \frac{P K}{P}$
 - $L = \frac{L}{\tilde{L}} \tilde{L}$
 - $l = \frac{L}{N}$

The data source for both $L$ is given as an index with 2017 as the base year. Therefore $\tilde{L}$ is constructed as follows:

$$\tilde{L} = \overbrace{\left [\frac{\underbrace{150,000,000}_{\text{labor force in 2017}} \times \underbrace{2,000/4}_{\text{work hrs/quarter}}}{\underbrace{100}_{\text{base index value}}}\right ]}^{\text{(1) to hours based on typical work year}} \div \overbrace{\left [\underbrace{\left (\underbrace{365 \times \frac{5}{7} }_{\text{wkdays/yr}} - \underbrace{2 \times 7}_{\text{2 wks vacation}} \right ) \times \underbrace{24}_{\text{hrs/day}} \times \underbrace{\frac{1}{4}}_{\text{quarters}}}_{\text{total yearly hours minus weekends and 2 weeks vacation}} \right ]}^{\text{(2) to fraction of total available time}}$$
""")
#st.altair_chart(st.session_state['rbc_model'].sim_deviations_chart())
