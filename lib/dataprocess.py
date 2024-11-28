import requests
import os
import json
import pandas as pd
import numpy as np

#FRED_API_KEY = os.environ["FRED_API"]

def float_or_nan(x):
    try:
        return float(x)
    except:
        return float('nan')

def get_fred_data(label,name=None,api_key=''):
    try:
        data = json.loads(
                    requests.get(
                        f'https://api.stlouisfed.org/fred/series/observations?series_id={label}&api_key={api_key}&file_type=json'
                                ).text
                        )['observations']

    except Exception as e:
        print('yada')
        print(e)
        try:
            data = json.loads(
                    requests.get(
                        f'https://api.stlouisfed.org/fred/series/observations?series_id={label}&api_key={api_key}&file_type=json'
                                ).text
                        )['observations']
        except Exception as e:
            raise(e)
    df = pd.DataFrame(data)[['date','value']]
    df['date'] = df['date'].astype('datetime64[ns]')
    try:
        df['value'] = df['value'].astype(float)
    except:
        df['value'] = df['value'].apply(float_or_nan)
    if name is not None:
        df = df.rename(columns={'value':name})
    return df




def get_us_macro_data_for_rbcdemo(import_dict={'K1TTOTL1ES000':'PK',
                                               #'GPDIC1':'K',
                                               #'W987RC1Q027SBEA':'K',
                                               'B230RC0Q173SBEA':'N',
                                               #'GDPC1':'Y',
                                               'GDP':'PY',
                                               'GDPDEF':'P',
                                               'HOANBS':'H',
                                               #'PRS84006023':'l',
                                               'LFAC64TTUSQ647S':'L'
                                              },
                                  unrate_var='UNRATE',
                                  fred_api_key='',
                                  cache_file='data/fred_data_cached.gzip',
                                  refresh=False
                                 ):
    if (cache_file is not None) and not refresh:
        try:
            df = pd.read_parquet(cache_file)
        except:
            refresh = True
    if refresh:
        print('Getting UNRATE...')
        df_unrate = get_fred_data('UNRATE',name='unrate',api_key=fred_api_key)
        df_unrate['quarter'] = df_unrate.date.dt.to_period('Q')
        df_unrate = df_unrate.groupby('quarter').agg({'unrate':'mean','date':'first'}).reset_index(drop=True)
        print('...done.')
        df = None
        for var in import_dict:
            print(f'Getting {var}...')
            curdf = get_fred_data(var,name=import_dict[var],api_key=fred_api_key)
            if df is None:
                df = curdf.copy()
            else:
                df = df.merge(curdf,how='outer',left_on='date',right_on='date')
            print(f'...done.')

        df['unrate'] = df[['date']].merge(df_unrate,how='left',left_on='date',right_on='date')['unrate']
        df['date'] = df['date'].astype('datetime64[ns]')
        df['log_PK'] = np.log(df.PK)
        df['log_PK'] = df['log_PK'].interpolate(limit_area='inside')
        df['PK'] = np.exp(df['log_PK'])
        if cache_file is not None:
            print(f'Saving cached file to {cache_file}...')
            df.to_parquet(cache_file)
            print('...done.')

    return df

def make_detrended_rbc_sample(df_in,alph,
                              l_normalization_factor=(150000000*2000/((365*5/7-2*7)*24))/100,
                              K_normalization_factor=1./1000,
                              return_gy=False,
                              periodicity=4
                             ):
    df_in = df_in.copy()
    df_in['PK'] *= K_normalization_factor

    for var in ['Y','K']:
        df_in[var] = 100*df_in['P' + var]/df_in['P']
    df_in['A'] = df_in['Y']/((df_in['K']**alph)*(df_in['H']**(1-alph)))

    df_in['y'] = df_in.Y/df_in.L
    df_in['k'] = df_in['K']/df_in['L']
    df_in['l'] = l_normalization_factor*df_in['H']/df_in['L']

    incl = (df_in.A/df_in.L).isna().apply(lambda x: not x)

    df_samp = df_in[incl].copy().reset_index(drop=True)
    g_y = np.log(df_samp.y.iloc[-1]/df_samp.y.iloc[0])/((df_samp.index[-1] - df_samp.index[0])/periodicity)


    df_samp['y_stat'] = df_samp.y/((1 + g_y/periodicity)**df_samp.index)
    ystatmean = df_samp['y_stat'].mean()
    df_samp['y_stat'] /= ystatmean
    df_samp['k_stat'] = df_samp.k/((1 + g_y/periodicity)**df_samp.index)
    kstatmean = df_samp['k_stat'].mean()
    df_samp['k_stat'] /= kstatmean


    df_samp['y_trend'] = ystatmean*((1 + g_y/periodicity)**df_samp.index)
    df_samp['k_trend'] = kstatmean*((1 + g_y/periodicity)**df_samp.index)

    df_samp['yhat_stat'] = df_samp.k_stat**alph*df_samp.l**(1-alph)

    df_samp['A_stat'] = df_samp['y_stat']/df_samp['yhat_stat']

    df_samp['log_y_stat'] = np.log(df_samp['y_stat'])
    df_samp['log_k_stat'] = np.log(df_samp['k_stat'])
    df_samp['log_A_stat'] = np.log(df_samp['A_stat'])
    df_samp['log_l_stat'] = np.log(df_samp['l']/df_samp['l'].mean())

    if return_gy:
        return df_samp,g_y
    else:
        return df_samp
