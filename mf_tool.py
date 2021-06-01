import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings("ignore")
from datetime import date, timedelta
import quandl
quandl.ApiConfig.api_key = "KZ69tzkHfXscfQ1qcJ5K"
import streamlit as st
import streamlit.components.v1 as components
import base64
import edhec_risk_kit as erk
from io import BytesIO


import plotly.express as px
import plotly.graph_objects as go
import plotly

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.core.display import display, HTML
from mftool import Mftool


@st.cache(allow_output_mutation=True)
def get_mf_codes():
    mf = Mftool()
    code_list = mf.get_scheme_codes()
    codes_list = pd.DataFrame(code_list, index=['List']).T
    codes_list.index.name = 'Code'
    return codes_list

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="NAV_Data.xlsx">Export to Excel</a>' # decode b'abc' => abc


def get_mf_data(selection, start, end):
    """
    """
    selected_list = codes_list[codes_list['List'].isin(selection)]
    codes = list(selected_list.index)
    names = list(selected_list['List'])

    df = pd.DataFrame(index = pd.bdate_range(start, end))
    df.index.name = 'Date'
    
    for i in range(0, len(codes)):
        data = pd.DataFrame(quandl.get("AMFI/"+str(codes[i]), start_date=start, end_date=end)['Net Asset Value'])
        data.columns = [names[i]]
        df = df.join(data, on='Date')
        
    return df

def plot_chart(data):
    """
    Returns a Plotly Interactive Chart for the given timeseries data (price)
    data = price data for the ETFs (dataframe)
    """
    df = ((((1+data.dropna().pct_change().fillna(0.00))).cumprod()-1)).round(4)
    fig = px.line(df, x=df.index, y=df.columns)
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Return (%)', font=dict(family="Segoe UI, monospace", size=14, color="#7f7f7f"),
                      legend_title_text='Securities', plot_bgcolor = 'White', yaxis_tickformat = '%', width=950, height=600)
    fig.update_traces(hovertemplate='Date: %{x} <br>Return: %{y:.2%}')
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(automargin=True)
    fig.add_hline(y=0.0, line_dash="dash", line_color="red")
    return fig

def scatter_plot(rets):
    ann_ret = (((1+rets).cumprod().iloc[-1,:])**(252/len(rets))-1)
    ann_vol = (rets.std()*np.sqrt(252))
    ratio = ann_ret/ann_vol
    tabl = pd.DataFrame(pd.concat([ann_ret, ann_vol, ratio.round(2)], axis=1))
    tabl.index = list(rets.columns)
    tabl.index.name = 'Mutual Funds'
    tabl.columns = ['Annualized Return', 'Annualized Volatility', 'Reward/Risk']
    fig = px.scatter(tabl, y='Annualized Return', x='Annualized Volatility', size='Reward/Risk', hover_name=tabl.index, color=tabl.index)
    fig.update_layout(xaxis_title='Annualized Volatility (%)',
                      yaxis_title='Annualized Return (%)', font=dict(family="Segoe UI, monospace", size=14, color="#7f7f7f"),
                      legend_title_text='Securities', plot_bgcolor = 'White', yaxis_tickformat = '.2%', xaxis_tickformat = '.2%', width=950, height=600)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
    fig.update_yaxes(automargin=True)
    return fig

side_options = st.sidebar.radio('Please Select One:', ('MF Data Explorer', 'Performance Metrics: Custom Data'))

if side_options == 'MF Data Explorer':
    
    st.write("""
    # MUTUAL FUND DATA EXPLORER
    Data Source: Association of Mutual Funds in India (AMFI)
    """)
    
    codes_list = get_mf_codes()
    selection  = st.multiselect(label='Select Mutual Funds: ', options=list(codes_list['List']), default=None)
    start_date = st.date_input("Start Date: ", date(2020,2,1))
    end_date   = st.date_input("End Date: ", date.today()-timedelta(1))
    
    if st.button("Show NAV data"):
        #data_load_state = st.text('Loading data...')
        nav = get_mf_data(selection=selection, start=start_date, end=end_date)
        rets = nav.ffill().bfill().pct_change().fillna(0)
        summary_metrics = erk.summary_stats(rets, 0, 252)
        st.write(nav)
        #data_load_state.text('Done!')
        st.markdown(get_table_download_link(nav), unsafe_allow_html=True)
        st.markdown("## Mutual Fund Performance Metrics")
        #Performance Metrics for selected MF for given time frame
        st.write("Start Date: " +str(rets.index[0].day)+"/"+str(rets.index[0].month)+"/"+str(rets.index[0].year))
        st.write("End Date: " +str(rets.index[-1].day)+"/"+str(rets.index[-1].month)+"/"+str(rets.index[-1].year))
        st.write(summary_metrics)
        st.write(str("*Assumes 0% daily returns for funds with a different start date."))

        st.markdown("## Risk/Reward Scatter Plot", unsafe_allow_html=True)
        st.plotly_chart(scatter_plot(rets))
        perf_chart =  plot_chart(nav)

        st.markdown("## Performance Chart", unsafe_allow_html=True)
        st.plotly_chart(perf_chart) 


if side_options == 'Performance Metrics: Custom Data':

    st.write("""
    # PERFORMANCE ANALYTICS
    """)
    
    cust_df = st.file_uploader('Choose an Excel File')
    if cust_df:
        cust_df = pd.read_excel(cust_df, header=0, index_col=0)
        periodicity = st.selectbox(label='Select Data Frequency: ', options=['Daily', 'Monthly'])
        if periodicity =='Daily':
            freq = 252
        else:
            freq=12

        rets = cust_df.ffill().bfill().pct_change().dropna()
        summary_metrics = erk.summary_stats(rets, 0, freq)
        st.write(summary_metrics.T)
        st.markdown(get_table_download_link(summary_metrics), unsafe_allow_html=True)

        st.markdown("## Performance Chart", unsafe_allow_html=True)
        perf_chart = plot_chart(cust_df)
        st.plotly_chart(perf_chart)


