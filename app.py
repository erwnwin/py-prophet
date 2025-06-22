import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from datetime import datetime
import holidays

# Fungsi untuk menghitung SMAPE
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Harga Bahan Pokok Kab. Gowa",
    page_icon="📈",
    layout="wide"
)

# CSS untuk tampilan yang lebih baik
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stDataFrame {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.title("📈 Prediksi Harga Bahan Pokok Kab. Gowa")
st.markdown("Aplikasi ini menggunakan Facebook Prophet dengan implementasi lengkap untuk prediksi harga bahan pokok")

# ======================
# PREPROCESSING DATA
# ======================
def preprocess_data(df):
    # Konversi tanggal dan pastikan terurut
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    
    # Hapus duplikat tanggal
    df = df.drop_duplicates('ds')
    
    # Deteksi dan handle outlier dengan IQR
    Q1 = df['y'].quantile(0.25)
    Q3 = df['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Catat outlier tanpa menghapus (kita smoothing)
    df['outlier'] = (df['y'] < lower_bound) | (df['y'] > upper_bound)
    
    # Smoothing outlier dengan rolling median
    if df['outlier'].any():
        df.loc[df['outlier'], 'y'] = df['y'].rolling(7, min_periods=1).median()
    
    # Isi missing dates dengan interpolasi
    all_dates = pd.date_range(start=df['ds'].min(), end=df['ds'].max())
    df = df.set_index('ds').reindex(all_dates)
    df['y'] = df['y'].interpolate(method='time')
    df = df.reset_index().rename(columns={'index': 'ds'})
    
    return df

# ======================
# LIBURAN INDONESIA
# ======================
def get_indonesia_holidays(year):
    id_holidays = holidays.Indonesia(years=year)
    holiday_df = pd.DataFrame.from_dict(id_holidays, orient='index').reset_index()
    holiday_df.columns = ['ds', 'holiday']
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    holiday_df['lower_window'] = -2  # 2 hari sebelum libur
    holiday_df['upper_window'] = 1   # 1 hari setelah libur
    return holiday_df

# ======================
# MAIN APP
# ======================
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca data
        data = pd.read_csv(uploaded_file)
        
        # Konversi kolom tanggal
        data['Tanggal'] = pd.to_datetime(data['Tanggal'])
        
        # Pilih bahan pokok
        bahan_pokok = st.selectbox(
            "Pilih bahan pokok:",
            options=data.columns[1:]
        )
        
        # Siapkan data untuk Prophet
        df = data[['Tanggal', bahan_pokok]].rename(columns={'Tanggal': 'ds', bahan_pokok: 'y'})
        
        # Preprocessing
        df = preprocess_data(df)
        
        # Tampilkan data
        st.subheader("📊 Data Historis (Setelah Preprocessing)")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.dataframe(df.sort_values('ds', ascending=False).style.highlight_max(color='lightgreen').highlight_min(color='#ffcccb'))
        with col2:
            st.metric("Jumlah Data", len(df))
        with col3:
            st.metric("Range Tanggal", f"{df['ds'].min().date()} to {df['ds'].max().date()}")
        
        # Visualisasi data historis
        st.subheader("📈 Tren Historis")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df['ds'], 
            y=df['y'], 
            mode='lines+markers',
            name='Harga Aktual',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig_hist.update_layout(
            xaxis_title='Tanggal',
            yaxis_title='Harga',
            template='plotly_white',
            height=500,
            hovermode='x unified',
            xaxis=dict(
                rangeslider=dict(visible=True),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Parameter model
        with st.expander("⚙️ Advanced Parameters", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                periods = st.number_input("Jumlah Hari Prediksi", min_value=1, max_value=365, value=30)
                seasonality_mode = st.selectbox("Mode Musiman", ['additive', 'multiplicative'])
                changepoint_prior_scale = st.slider("Sensitivitas Tren", 0.001, 0.5, 0.05, 0.001)
                holidays_prior_scale = st.slider("Pengaruh Liburan", 0.01, 10.0, 1.0)
            with col2:
                yearly_seasonality = st.checkbox("Musiman Tahunan", value=True)
                weekly_seasonality = st.checkbox("Musiman Mingguan", value=True)
                daily_seasonality = st.checkbox("Musiman Harian", value=False)
                log_transform = st.checkbox("Gunakan Log Transform", value=False)
        
        if st.button("🚀 Buat Prediksi (Advanced)"):
            with st.spinner('Membuat prediksi...'):
                # Persiapan data
                train_df = df.copy()
                
                # Log transform jika dipilih
                if log_transform:
                    train_df['y_original'] = train_df['y']
                    train_df['y'] = np.log(train_df['y'])
                
                # Siapkan kalender libur
                years = list(range(train_df['ds'].min().year, train_df['ds'].max().year + 1))
                id_holidays = get_indonesia_holidays(years)
                
                # Inisialisasi model
                model = Prophet(
                    seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    holidays=id_holidays,
                    uncertainty_samples=1000
                )
                
                # Tambahkan musiman khusus
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                
                # Fit model
                model.fit(train_df)
                
                # Buat prediksi
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                # Transformasi balik jika menggunakan log
                if log_transform:
                    forecast['yhat'] = np.exp(forecast['yhat'])
                    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
                    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
                    train_df['y'] = train_df['y_original']
                
                # Gabungkan data aktual dengan prediksi
                results = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly', 'weekly']]
                results = results.join(train_df.set_index('ds')[['y']])
                results['error'] = results['y'] - results['yhat']
                results = results.reset_index()
                results.columns = ['Tanggal', 'Peramalan', 'Peramalan Bawah', 'Peramalan Atas', 
                                 'Trend', 'Musiman Tahunan', 'Musiman Mingguan', 'Aktual', 'Error']
                
                # Cross-validation
                st.subheader("📊 Cross-Validation")
                try:
                    df_cv = cross_validation(
                        model,
                        initial=f'{len(train_df)//2} days',
                        period='30 days',
                        horizon='90 days'
                    )
                    df_p = performance_metrics(df_cv)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(df_p.style.background_gradient(cmap='Blues'))
                    with col2:
                        fig_cv = go.Figure()
                        fig_cv.add_trace(go.Box(
                            y=df_p['smape'],
                            name='SMAPE',
                            boxpoints='all'
                        ))
                        fig_cv.update_layout(
                            title='Distribusi SMAPE dari CV',
                            yaxis_title='SMAPE (%)'
                        )
                        st.plotly_chart(fig_cv, use_container_width=True)
                except Exception as e:
                    st.warning(f"Cross-validation error: {str(e)}")
                
                # Tampilkan hasil
                st.subheader("📋 Hasil Prediksi")
                st.dataframe(results.tail(periods + 30).style.format({
                    'Peramalan': '{:,.0f}',
                    'Peramalan Bawah': '{:,.0f}',
                    'Peramalan Atas': '{:,.0f}',
                    'Aktual': '{:,.0f}',
                    'Error': '{:,.0f}'
                }).background_gradient(subset=['Error'], cmap='RdYlGn'))
                
                # Visualisasi prediksi
                st.subheader("🔮 Grafik Prediksi")
                fig_pred = go.Figure()
                
                # Data aktual
                fig_pred.add_trace(go.Scatter(
                    x=results['Tanggal'],
                    y=results['Aktual'],
                    mode='lines+markers',
                    name='Aktual',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                # Prediksi
                fig_pred.add_trace(go.Scatter(
                    x=results['Tanggal'],
                    y=results['Peramalan'],
                    mode='lines',
                    name='Peramalan',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Interval keyakinan
                fig_pred.add_trace(go.Scatter(
                    x=pd.concat([results['Tanggal'], results['Tanggal'][::-1]]),
                    y=pd.concat([results['Peramalan Atas'], results['Peramalan Bawah'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line_color='rgba(255,255,255,0)',
                    name='Interval Keyakinan 80%',
                    hoverinfo='skip'
                ))
                
                fig_pred.update_layout(
                    xaxis_title='Tanggal',
                    yaxis_title='Harga',
                    template='plotly_white',
                    height=500,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Komponen prediksi
                st.subheader("🧩 Komponen Prediksi")
                fig_comp = go.Figure()
                
                fig_comp.add_trace(go.Scatter(
                    x=results['Tanggal'],
                    y=results['Trend'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='green', width=2)
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=results['Tanggal'],
                    y=results['Musiman Tahunan'],
                    mode='lines',
                    name='Musiman Tahunan',
                    line=dict(color='purple', width=1)
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=results['Tanggal'],
                    y=results['Musiman Mingguan'],
                    mode='lines',
                    name='Musiman Mingguan',
                    line=dict(color='orange', width=1)
                ))
                
                fig_comp.update_layout(
                    xaxis_title='Tanggal',
                    yaxis_title='Komponen',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Hitung SMAPE
                actuals = results['Aktual'].dropna()
                preds = results['Peramalan'][:len(actuals)]
                smape_value = smape(actuals, preds)
                
                # Tampilkan evaluasi
                st.subheader("📊 Evaluasi Akurasi")
                cols = st.columns(4)
                cols[0].metric("SMAPE", f"{smape_value:.2f}%")
                
                interpretation = "Sangat Baik" if smape_value < 10 else \
                                "Baik" if smape_value < 20 else \
                                "Cukup" if smape_value < 50 else "Buruk"
                cols[1].metric("Interpretasi", interpretation)
                
                cols[2].metric("Jumlah Data Training", len(train_df))
                cols[3].metric("Jumlah Hari Prediksi", periods)
                
                st.markdown("""
                **Interpretasi SMAPE:**
                - < 10%: Prediksi sangat akurat
                - 10-20%: Prediksi baik
                - 20-50%: Prediksi cukup
                - > 50%: Prediksi kurang akurat
                """)
                
                # Download hasil prediksi
                csv = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi (CSV)",
                    data=csv,
                    file_name=f"prediksi_{bahan_pokok}.csv",
                    mime='text/csv'
                )
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("© 2023 - Aplikasi Prediksi Harga Bahan Pokok (Advanced)")