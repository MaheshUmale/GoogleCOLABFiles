
#%%
# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date


#import yfinance as yf
#from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from ks_api_client import ks_api
import logging
from tvDatafeed import TvDatafeed,Interval
import pandas as pd
import pandas_ta as ta
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import plotly.figure_factory as ff

from neuralprophet import NeuralProphet

# filenameFNO = "https://preferred.kotaksecurities.com/security/production/TradeApiInstruments_FNO_31_12_2021.txt"
# filenameCASH = "https://preferred.kotaksecurities.com/security/production/TradeApiInstruments_Cash_31_12_2021.txt"
# 

# FNO_InstrumentType = pd.read_csv(filenameFNO,sep="|")
# CASH_InstrumentType = pd.read_csv(filenameCASH,sep="|")


# instr = st.sidebar.text_input(label='Instrument ') 
# CASH_InstrumentType = CASH_InstrumentType[['instrumentToken', 'instrumentName','exchange']].copy()
# FNO_InstrumentType = FNO_InstrumentType[['instrumentToken', 'instrumentName','exchange','expiry']].copy()
# instr_cash = CASH_InstrumentType[CASH_InstrumentType.instrumentName==instr]
# instr_fno = FNO_InstrumentType[FNO_InstrumentType.instrumentName==instr]

# st.write('CASH SEGMENT ',instr_cash) 
# st.write('FNO SEGMENT ',instr_fno) 
# st.header('FOR GETTING INSTRUMENT TOKEN ')
# st.write('https://preferred.kotaksecurities.com/security/production/TradeApiInstruments_Cash_30_12_2021.txt')
# st.write('https://preferred.kotaksecurities.com/security/production/TradeApiInstruments_FNO_30_12_2021.txt')

#selected_indices = st.multiselect('Select rows:', FNO_InstrumentType['instrumentName'])

logging.basicConfig(level=logging.DEBUG)
otp = st.sidebar.text_input(label='InputOTP')  
# my_expander = st.sidebar.expander(label='CASH SEGMENT')

# my_expander2 = st.sidebar.expander(label='FNO SEGMENT')

# with my_expander:
#     #CASH_InstrumentType = CASH_InstrumentType['instrumentName']
#     # CASH_InstrumentType_choice = st.sidebar.selectbox('Select your CASH_InstrumentType:', CASH_InstrumentType)
#     # #st.write('CASH SEGMENT ',CASH_InstrumentType) 
#     # 
# with my_expander2:
#     #FNO_InstrumentType = FNO_InstrumentType['instrumentName']
#     FNO_InstrumentType_choice = st.sidebar.selectbox('Select your FNO_InstrumentType:', FNO_InstrumentType['instrumentName'])
    
#     category = FNO_InstrumentType['instrumentToken'].loc[FNO_InstrumentType['instrumentName'] == FNO_InstrumentType_choice].unique()  
#     Category_choice = Data_filtering[1].selectbox("instrumentToken", category)
#     #st.write('FNO SEGMENT ',FNO_InstrumentType) 
if st.sidebar.checkbox("Login"):
	# Defining the host is optional and defaults to https://tradeapi.kotaksecurities.com/apim
	# See configuration.py for a list of all supported configuration parameters.
	global client
	client = ks_api.KSTradeApi(access_token = "b9025537-6c88-3920-bd62-fae692ff2b7d", userid = "MUMALE09", \
					consumer_key = "vMGVEhTFkSltuxsHNc36XdD4o44a", ip = "127.0.0.1", app_id = "DefaultApplication")

	#For using sandbox environment use host as https://sbx.kotaksecurities.com/apim
	# client = ks_api.KSTradeApi(access_token = "25956127-cd72-3655-ac1e-152cddeb6b50", userid = "MUMALE09", \
	#                 consumer_key = "8wy3HbGiPUD3Ph10JYI03xF0Q_ka", ip = "127.0.0.1", app_id = "DefaultApplication", host = "https://sbx.kotaksecurities.com/apim")

	# Get session for user
	client.login(password = "July@2021")
	#Generated session token
	client.session_2fa(access_code = "1294")

	# from pynse import *
	# nse=Nse()
	# nse.market_status()
	# get credentials for tradingview
	username = 'mumale@gmail.com'
	password = 'Kiaan@123'
	# initialize tradingview

	tv=TvDatafeed(username=username,password=password)


	# START = "2018-01-01"
	# TODAY = date.today().strftime("%Y-%m-%d")

	st.title('Stock Forecast App')
	#exchange=st.input()
	stocks = ('NIFTY', 'RELIANCE', 'SBIN')
	selected_stock = st.selectbox('Select dataset for prediction', stocks)
	days = st.slider('days of prediction:', 3, 1000)
	period = days
	interval_type =("in_daily","in_30_minute","in_1_minute","in_3_minute","in_5_minute","in_15_minute","in_45_minute","in_1_hour","in_2_hour","in_3_hour","in_4_hour","in_daily","in_weekly","in_monthly")
	interval_time =st.selectbox('Select interval ', interval_type)
	intr = Interval[interval_time]
	n_bars=st.number_input('Insert a number',min_value =100,max_value =5000,value=500)
	@st.cache
	def load_data(ticker,interval_time,bars):
		#data = nse.get_hist(ticker)
		data = tv.get_hist(ticker,'NSE',interval= interval_time,n_bars=bars)
		data.reset_index(inplace=True)
		return data
		
	data_load_state = st.text('Loading data...')
	data_ = load_data(selected_stock,intr,n_bars)
	data_load_state.text('Loading data... done!')

	st.subheader('Raw data')
	st.write(data_.tail())

	# Plot raw data
	def plot_raw_data():
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=data_['datetime'], y=data_['open'], name="stock_open"))
		fig.add_trace(go.Scatter(x=data_['datetime'], y=data_['close'], name="stock_close"))
		fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
		st.plotly_chart(fig)
		
	plot_raw_data()


	# # Predict forecast with Prophet.
	df_train = data_[['datetime','close']]
	df_train = df_train.rename(columns={"datetime": "ds", "close": "y"})
	
	
	# m = NeuralProphet() # default model
	# our model
	m = NeuralProphet(
		n_forecasts=60,
		n_lags=60,
		n_changepoints=50,
		yearly_seasonality=True,
		weekly_seasonality=True,
		daily_seasonality=True,
		batch_size=64,
		epochs=100,
		learning_rate=1.0,
	)
	metrics = m.fit(df_train,freq="D")  # fit the model using all data
	# with cross-validation
	# metrics = m.fit(data, 
	#                 freq="D",
	#                 valid_p=0.2, # validation proportion of data (20%)
	#                 epochs=100)

	# Predictions
	# future = m.make_future_dataframe(df_train, periods=60, n_historic_predictions=period) #we need to specify the number of days in future
	# prediction = m.predict(future)
 
	# m.fit(df_train)
	future = m.make_future_dataframe(df_train,periods=period, n_historic_predictions=len(df_train))
	forecast = m.predict(future)

	# # Plotting
	# forecast = m.plot(prediction)
	# plt.title("Prediction of the "+selected_stock+"Stock Price using NeuralProphet for the next 60 days")
	# plt.xlabel("Date")
	# plt.ylabel("Close Stock Price")
	# #plt.show()
	# st.pyplot(plt)
	# fig =forecast
	# fig.layout.update(title_text='Time Series FORECAST data with Rangeslider', xaxis_rangeslider_visible=True)
	# st.plotly_chart(fig)

	st.write(f'Forecast plot for {days} days')
	forecast_fig2 = m.plot(forecast)
 
	st.write(forecast_fig2)
	forecast_fig = go.Figure()
	forecast_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name="predict",mode="markers+lines", marker_symbol="star"))
	forecast_fig.add_trace(go.Scatter(x=future['ds'], y=future['y'], name="close"))
	forecast_fig.layout.update(title_text='forecast Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(forecast_fig)
 
	st.write(f'Forecast plot no 2 for {days} days')
	
	st.write(forecast_fig)
 
	fig2 = go.Figure()
	fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], name="predict",mode="markers+lines", marker_symbol="star"))
	fig2.add_trace(go.Scatter(x=future['ds'], y=future['y'], name="close"))
	fig2.layout.update(title_text='forecast Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig2)
	# fig1 = plot_plotly(m, forecast)
	# st.plotly_chart(fig1)

	st.write("Forecast components")
	fig2 = m.plot_components(forecast)
	st.write(fig2)
 
	# m = Prophet()
	# m.fit(df_train)
	# future = m.make_future_dataframe(periods=period)
	# forecast = m.predict(future)

	# forecast_lower=forecast['yhat_lower'].iloc[-1]

	# forecast_upper=forecast['yhat_upper'].iloc[-1]
	# # Show and plot forecast
	# st.subheader('Forecast dataupper')
	# st.write(forecast_upper)
	# st.subheader('Forecast data')
	# st.write(forecast)
	# st.subheader('Forecast datalower')
	# st.write(forecast_lower)
		
	# st.write(f'Forecast plot for {days} days')
	# fig1 = plot_plotly(m, forecast)
	# st.plotly_chart(fig1)

	# # st.write("Forecast components")
	# # fig2 = m.plot_components(forecast)
	# # st.write(fig2)
