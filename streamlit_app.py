import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Trading Bot LSTM - Acciones, ETFs y BTC")

# -------------------
# Input del usuario
# -------------------
ticker = st.text_input("Ingresa el ticker (ej. AAPL, GLD, BTC-USD)", "BTC-USD")
N_future = st.number_input("Días a predecir", min_value=1, max_value=30, value=5)

if st.button("Ejecutar predicción"):
    # -------------------
    # Descargar datos
    # -------------------
    data = yf.download(ticker, start='2015-01-01')
    if data.empty:
        st.error("No se encontraron datos para ese ticker.")
    else:
        st.write("Últimos datos descargados:")
        st.dataframe(data.tail())

        # -------------------
        # Funciones indicadores
        # -------------------
        def calculate_SMA(data, period):
            return data['Close'].rolling(period).mean()
        
        def calculate_RSI(data, period=14):
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        # -------------------
        # Optimización SMA/RSI
        # -------------------
        best_gain = -np.inf
        best_params = (20,50,14)
        for sma_short in [10,20,30]:
            for sma_long in [50,100,200]:
                for rsi_period in [10,14,20]:
                    df = data.copy()
                    df['SMA_short'] = calculate_SMA(df, sma_short)
                    df['SMA_long'] = calculate_SMA(df, sma_long)
                    df['RSI'] = calculate_RSI(df, rsi_period)
                    df = df.dropna()
                    if df.empty:
                        continue
                    capital = 100
                    cash = capital
                    position = 0
                    for i in range(len(df)):
                        price = df['Close'].iloc[i]
                        if df['SMA_short'].iloc[i] > df['SMA_long'].iloc[i] and df['RSI'].iloc[i] < 30 and cash>0:
                            position = cash/price
                            cash=0
                        elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] and df['RSI'].iloc[i] >70 and position>0:
                            cash = position*price
                            position=0
                    final_value = cash + (position*df['Close'].iloc[-1])
                    if final_value>best_gain:
                        best_gain=final_value
                        best_params=(sma_short,sma_long,rsi_period)

        SMA_short,SMA_long,RSI_period=best_params
        st.write(f"Parámetros optimizados: SMA_short={SMA_short}, SMA_long={SMA_long}, RSI_period={RSI_period}")

        # -------------------
        # Aplicar indicadores
        # -------------------
        data['SMA_short'] = calculate_SMA(data,SMA_short)
        data['SMA_long'] = calculate_SMA(data,SMA_long)
        data['RSI'] = calculate_RSI(data,RSI_period)
        data = data.dropna()

        # -------------------
        # Preparar datos para LSTM
        # -------------------
        features=['Close','SMA_short','SMA_long','RSI']
        scaler=MinMaxScaler()
        scaled = scaler.fit_transform(data[features])

        X,y=[],[]
        for i in range(60,len(scaled)-N_future+1):
            X.append(scaled[i-60:i])
            y.append(scaled[i:i+N_future,0])
        X=np.array(X)
        y=np.array(y)

        # -------------------
        # Entrenar LSTM
        # -------------------
        model=Sequential([
            LSTM(100, return_sequences=True, input_shape=(X.shape[1],X.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(N_future)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X,y,epochs=30,batch_size=32,verbose=0)
        st.write("Modelo LSTM entrenado.")

        # -------------------
        # Predicción futura
        # -------------------
        last_60 = scaled[-60:].reshape(1,60,scaled.shape[1])
        pred_future = model.predict(last_60)
        zeros = np.zeros((pred_future.shape[0], len(features)-1))
        pred_concat = np.concatenate([pred_future, zeros], axis=1)
        pred_prices = scaler.inverse_transform(pred_concat)[0,:N_future]
        st.write(f"Predicción siguiente {N_future} días:")
        st.write(pred_prices)

        # -------------------
        # Señales de trading
        # -------------------
        signals=[]
        for i in range(len(data)):
            if data['SMA_short'].iloc[i]>data['SMA_long'].iloc[i] and data['RSI'].iloc[i]<30:
                signals.append("Buy")
            elif data['SMA_short'].iloc[i]<data['SMA_long'].iloc[i] and data['RSI'].iloc[i]>70:
                signals.append("Sell")
            else:
                signals.append("Hold")
        data['Signal']=signals

        # -------------------
        # Backtesting
        # -------------------
        capital=100
        position=0
        cash=capital
        trade_log=[]
        for i in range(len(data)):
            price = data['Close'].iloc[i]
            sig = data['Signal'].iloc[i]
            if sig=="Buy" and cash>0:
                position=cash/price
                cash=0
                trade_log.append((data.index[i],'Buy',price))
            elif sig=="Sell" and position>0:
                cash=position*price
                position=0
                trade_log.append((data.index[i],'Sell',price))
        final_value=cash+position*data['Close'].iloc[-1]
        st.write(f"Valor final tras backtesting: {final_value:.2f}")

        # -------------------
        # Gráficos
        # -------------------
        plt.figure(figsize=(12,6))
        plt.plot(data['Close'],label='Precio')
        plt.plot(data['SMA_short'],label=f"SMA{SMA_short}")
        plt.plot(data['SMA_long'],label=f"SMA{SMA_long}")
        buys = data[data['Signal']=="Buy"]
        sells = data[data['Signal']=="Sell"]
        plt.scatter(buys.index,buys['Close'],marker='^',color='green',s=100,label='Buy Signal')
        plt.scatter(sells.index,sells['Close'],marker='v',color='red',s=100,label='Sell Signal')
        plt.legend()
        plt.title(f"{ticker} - Precio + Señales")
        st.pyplot()
        