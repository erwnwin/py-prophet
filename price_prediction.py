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

        # =====================================
        # PEMBAGIAN DATA TRAINING DAN TESTING
        # =====================================
        st.subheader("✂️ Pembagian Data Training-Testing")
        split_ratio = st.slider("Persentase Data Training", 50, 95, 80, 5)
        split_date = df['ds'].quantile(split_ratio/100)

        train_df = df[df['ds'] <= split_date].copy()
        test_df = df[df['ds'] > split_date].copy()

        col1, col2 = st.columns(2)
        col1.metric("Data Training", f"{len(train_df)} titik ({train_df['ds'].min().date()} - {train_df['ds'].max().date()})")
        col2.metric("Data Testing", f"{len(test_df)} titik ({test_df['ds'].min().date()} - {test_df['ds'].max().date()})")

        # Visualisasi pembagian data
        fig_split = go.Figure()
        fig_split.add_trace(go.Scatter(
            x=train_df['ds'], 
            y=train_df['y'], 
            name='Training', 
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig_split.add_trace(go.Scatter(
            x=test_df['ds'], 
            y=test_df['y'], 
            name='Testing', 
            mode='lines',
            line=dict(color='red', width=2)
        ))
        fig_split.update_layout(
            title='Pembagian Data Training-Testing',
            xaxis_title='Tanggal',
            yaxis_title='Harga',
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig_split, use_container_width=True)

        # Parameter model
        with st.expander("⚙️ Pilih Parameters Prediksi", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                # Tambahkan pilihan periode prediksi
                pred_option = st.radio("Jangkauan Prediksi", 
                                     ["30 Hari", "1 Tahun", "Custom"])
                
                if pred_option == "Custom":
                    periods = st.number_input("Jumlah Hari Prediksi", min_value=1, max_value=365*3, value=30)
                elif pred_option == "30 Hari":
                    periods = 30
                else:  # 1 Tahun
                    periods = 365
                    
                seasonality_mode = st.selectbox("Mode Musiman", ['additive', 'multiplicative'])
                changepoint_prior_scale = st.slider("Sensitivitas Tren", 0.001, 0.5, 0.05, 0.001)
                holidays_prior_scale = st.slider("Pengaruh Liburan", 0.01, 10.0, 1.0)
            with col2:
                yearly_seasonality = st.checkbox("Musiman Tahunan", value=True)
                weekly_seasonality = st.checkbox("Musiman Mingguan", value=True)
                daily_seasonality = st.checkbox("Musiman Harian", value=False)
                log_transform = st.checkbox("Gunakan Log Transform", value=False)
        
        if st.button("🚀 Buat Prediksi"):
            with st.spinner('Membuat prediksi...'):
                # Persiapan data training
                train_df = train_df.copy()
                
                # Log transform jika dipilih
                if log_transform:
                    train_df['y_original'] = train_df['y']
                    train_df['y'] = np.log(train_df['y'])
                
                # Siapkan kalender libur
                years = list(range(train_df['ds'].min().year, train_df['ds'].max().year + 2))  # +2 untuk prediksi tahun depan
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
                
                # =====================================
                # EVALUASI MODEL PADA DATA TESTING
                # =====================================
                st.subheader("📊 Evaluasi Model pada Data Testing")
                
                # Buat prediksi untuk periode testing
                test_period = len(test_df)
                test_future = model.make_future_dataframe(periods=test_period)
                test_forecast = model.predict(test_future)
                
                # Transformasi balik jika menggunakan log
                if log_transform:
                    test_forecast['yhat'] = np.exp(test_forecast['yhat'])
                    test_forecast['yhat_lower'] = np.exp(test_forecast['yhat_lower'])
                    test_forecast['yhat_upper'] = np.exp(test_forecast['yhat_upper'])
                    train_df['y'] = train_df['y_original']
                
                # Gabungkan dengan data testing aktual
                test_results = pd.merge(
                    test_df,
                    test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                    on='ds'
                )
                
                # Rename kolom dengan cara yang lebih aman
                test_results = test_results.rename(columns={
                    'ds': 'Tanggal',
                    'y': 'Aktual',
                    'yhat': 'Peramalan',
                    'yhat_lower': 'Peramalan Bawah',
                    'yhat_upper': 'Peramalan Atas'
                })
                test_results['Error'] = test_results['Aktual'] - test_results['Peramalan']
                
                # Hitung SMAPE untuk data testing
                smape_value = smape(test_results['Aktual'], test_results['Peramalan'])
                
                # Tampilkan metrik evaluasi
                col1, col2, col3 = st.columns(3)
                col1.metric("SMAPE pada Data Testing", f"{smape_value:.2f}%")
                
                interpretation = "Sangat Baik" if smape_value < 10 else \
                                "Baik" if smape_value < 20 else \
                                "Cukup" if smape_value < 50 else "Buruk"
                col2.metric("Interpretasi", interpretation)
                
                col3.metric("Jumlah Data Testing", len(test_df))
                
                st.markdown("""
                **Interpretasi SMAPE:**
                - < 10%: Prediksi sangat akurat
                - 10-20%: Prediksi baik
                - 20-50%: Prediksi cukup
                - > 50%: Prediksi kurang akurat
                """)
                
                # Visualisasi evaluasi pada data testing
                fig_test = go.Figure()
                fig_test.add_trace(go.Scatter(
                    x=test_results['Tanggal'],
                    y=test_results['Aktual'],
                    mode='lines+markers',
                    name='Aktual (Testing)',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                fig_test.add_trace(go.Scatter(
                    x=test_results['Tanggal'],
                    y=test_results['Peramalan'],
                    mode='lines',
                    name='Peramalan',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig_test.add_trace(go.Scatter(
                    x=pd.concat([test_results['Tanggal'], test_results['Tanggal'][::-1]]),
                    y=pd.concat([test_results['Peramalan Atas'], test_results['Peramalan Bawah'][::-1]]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line_color='rgba(255,255,255,0)',
                    name='Interval Keyakinan 80%',
                    hoverinfo='skip'
                ))
                fig_test.update_layout(
                    title='Evaluasi Model pada Data Testing',
                    xaxis_title='Tanggal',
                    yaxis_title='Harga',
                    template='plotly_white',
                    height=500
                )
                st.plotly_chart(fig_test, use_container_width=True)
                
                # =====================================
                # PREDIKSI KE DEPAN
                # =====================================
                st.subheader("🔮 Prediksi ke Depan")
                
                # Buat prediksi untuk periode yang diminta
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                
                # Transformasi balik jika menggunakan log
                if log_transform:
                    forecast['yhat'] = np.exp(forecast['yhat'])
                    forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
                    forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
                
                # Gabungkan semua hasil
                full_results = forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly', 'weekly']]
                full_results = full_results.join(df.set_index('ds')[['y']])
                full_results = full_results.reset_index()
                full_results.columns = ['Tanggal', 'Peramalan', 'Peramalan Bawah', 'Peramalan Atas', 
                                      'Trend', 'Musiman Tahunan', 'Musiman Mingguan', 'Aktual']
                
                # Tampilkan hasil prediksi dalam tab
                tab1, tab2, tab3 = st.tabs(["Grafik Prediksi", "Prediksi per Bulan", "Tabel Prediksi"])
                
                with tab1:
                    fig_pred = go.Figure()
                    
                    # Data aktual
                    fig_pred.add_trace(go.Scatter(
                        x=full_results['Tanggal'],
                        y=full_results['Aktual'],
                        mode='lines+markers',
                        name='Aktual',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ))
                    
                    # Data training
                    fig_pred.add_trace(go.Scatter(
                        x=train_df['ds'],
                        y=train_df['y'],
                        mode='lines',
                        name='Training',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Data testing
                    fig_pred.add_trace(go.Scatter(
                        x=test_results['Tanggal'],
                        y=test_results['Aktual'],
                        mode='lines',
                        name='Testing',
                        line=dict(color='orange', width=2)
                    ))
                    
                    # Prediksi
                    fig_pred.add_trace(go.Scatter(
                        x=full_results['Tanggal'],
                        y=full_results['Peramalan'],
                        mode='lines',
                        name='Peramalan',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Interval keyakinan
                    fig_pred.add_trace(go.Scatter(
                        x=pd.concat([full_results['Tanggal'], full_results['Tanggal'][::-1]]),
                        y=pd.concat([full_results['Peramalan Atas'], full_results['Peramalan Bawah'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line_color='rgba(255,255,255,0)',
                        name='Interval Keyakinan 80%',
                        hoverinfo='skip'
                    ))
                    
                    fig_pred.update_layout(
                        title='Prediksi Harga dengan Evaluasi Model',
                        xaxis_title='Tanggal',
                        yaxis_title='Harga',
                        template='plotly_white',
                        height=600,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with tab2:
                    # Prediksi per Bulan
                    pred_data = full_results[full_results['Tanggal'] > df['ds'].max()]
                    pred_data['Bulan'] = pred_data['Tanggal'].dt.to_period('M').astype(str)
                    
                    # Agregasi per bulan
                    monthly_pred = pred_data.groupby('Bulan').agg({
                        'Peramalan': 'mean',
                        'Peramalan Bawah': 'mean',
                        'Peramalan Atas': 'mean'
                    }).reset_index()
                    
                    # Format angka
                    monthly_pred = monthly_pred.round(0)
                    
                    st.write(f"### Prediksi Rata-rata Bulanan untuk {bahan_pokok}")
                    st.dataframe(monthly_pred.style.format({
                        'Peramalan': '{:,.0f}',
                        'Peramalan Bawah': '{:,.0f}',
                        'Peramalan Atas': '{:,.0f}'
                    }).background_gradient(cmap='Blues'))
                    
                    # Tampilkan sebagai grafik batang
                    fig_monthly = go.Figure()
                    fig_monthly.add_trace(go.Bar(
                        x=monthly_pred['Bulan'],
                        y=monthly_pred['Peramalan'],
                        name='Rata-rata Prediksi',
                        marker_color='rgb(55, 83, 109)'
                    ))
                    
                    fig_monthly.add_trace(go.Scatter(
                        x=monthly_pred['Bulan'],
                        y=monthly_pred['Peramalan Bawah'],
                        name='Batas Bawah',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_monthly.add_trace(go.Scatter(
                        x=monthly_pred['Bulan'],
                        y=monthly_pred['Peramalan Atas'],
                        name='Batas Atas',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    fig_monthly.update_layout(
                        title=f'Prediksi Bulanan {bahan_pokok}',
                        xaxis_title='Bulan',
                        yaxis_title='Harga',
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig_monthly, use_container_width=True)
                
                with tab3:
                    # Tampilkan data prediksi ke depan
                    pred_data = full_results[full_results['Tanggal'] > df['ds'].max()]
                    st.dataframe(pred_data[['Tanggal', 'Peramalan', 'Peramalan Bawah', 'Peramalan Atas']]
                               .style.format({
                                   'Peramalan': '{:,.0f}',
                                   'Peramalan Bawah': '{:,.0f}',
                                   'Peramalan Atas': '{:,.0f}'
                               }))
                
                # Komponen prediksi
                st.subheader("🧩 Komponen Prediksi")
                fig_comp = go.Figure()
                
                fig_comp.add_trace(go.Scatter(
                    x=full_results['Tanggal'],
                    y=full_results['Trend'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='green', width=2)
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=full_results['Tanggal'],
                    y=full_results['Musiman Tahunan'],
                    mode='lines',
                    name='Musiman Tahunan',
                    line=dict(color='purple', width=1)
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=full_results['Tanggal'],
                    y=full_results['Musiman Mingguan'],
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
                
                # Download hasil prediksi
                csv = full_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil Prediksi Lengkap (CSV)",
                    data=csv,
                    file_name=f"prediksi_{bahan_pokok}.csv",
                    mime='text/csv'
                )
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("© 2023 - Aplikasi Prediksi Harga Bahan Pokok")