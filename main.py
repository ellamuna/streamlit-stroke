import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import numpy as np

# Membaca data yang diunggah oleh pengguna
data_path = 'data_stroke.csv'
clean_data_path = 'data_stroke_cleaned.csv'
data = pd.read_csv(data_path)
data_clean = pd.read_csv(clean_data_path)

# Membuat navbar
selected = option_menu(
    menu_title=None,
    options=["Informasi Stroke", "Visualisasi Data", "Prediksi Risiko Stroke"],
    icons=["heart-pulse", "bar-chart", "activity"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Konten berdasarkan pilihan menu
if selected == "Informasi Stroke":
    st.title("Stroke: Apa yang Perlu Anda Ketahui")
    st.image("image.png", use_container_width=True)
    st.subheader("Apa itu Stroke?")
    st.write("""
        Stroke terjadi ketika ada penyumbatan atau pecahnya pembuluh darah di otak, 
        menyebabkan sel-sel otak kekurangan oksigen dan nutrisi. Ada dua jenis stroke:
        - **Stroke Iskemik**: Disebabkan oleh penyumbatan pada pembuluh darah.
        - **Stroke Hemoragik**: Disebabkan oleh perdarahan di otak.
    """)
    st.subheader("Gejala dan Penyebab Stroke")
    st.write("""
        Gejala stroke umumnya terjadi di bagian tubuh yang dikendalikan oleh area otak yang rusak. Gejala yang dialami penderita stroke bisa meliputi:
        - Kehilangan rasa atau kelemahan mendadak pada wajah, lengan, atau kaki, terutama di satu sisi tubuh.
        - Kebingungan, kesulitan berbicara, atau memahami ucapan.
        - Kesulitan melihat di satu atau kedua mata.
        - Kesulitan berjalan, pusing, kehilangan keseimbangan, atau koordinasi.
    """)
    st.subheader("Pengobatan dan Pencegahan Stroke")
    st.write("""
        Pada umumnya, pencegahan stroke hampir sama dengan cara mencegah penyakit jantung, yaitu dengan menerapkan pola hidup sehat, seperti:
        - Jaga pola makan yang sehat.
        - Lakukan olahraga secara teratur.
        - Menjaga berat badan ideal.
        - Kelola tekanan darah, kolesterol, dan diabetes.
        - Menjalani pemeriksaan rutin untuk kondisi medis yang diderita.
        - Tidak merokok dan tidak mengonsumsi minuman beralkohols.
    """)

elif selected == "Visualisasi Data":
    st.title("Visualisasi Data Stroke")
    st.write("Pada bagian ini, kita akan mengeksplorasi data stroke melalui beberapa visualisasi dan analisis deskriptif.")

    # Deskripsi variabel
    st.subheader("Deskripsi Variabel")
    st.write("""
        - **id**: ID unik pasien.
        - **gender**: Jenis kelamin (Male atau Female).
        - **age**: Umur pasien.
        - **hypertension**: Riwayat hipertensi (0: Tidak, 1: Ya).
        - **heart_disease**: Riwayat penyakit jantung (0: Tidak, 1: Ya).
        - **ever_married**: Status pernikahan (Yes atau No).
        - **work_type**: Jenis pekerjaan (Private, Self-employed, Govt_job, children, Never_worked).
        - **Residence_type**: Jenis tempat tinggal (Urban atau Rural).
        - **avg_glucose_level**: Rata-rata tingkat glukosa dalam darah.
        - **bmi**: Indeks massa tubuh (BMI).
        - **smoking_status**: Status merokok (formerly smoked, never smoked, smokes, Unknown).
        - **stroke**: Kasus stroke (0: Tidak, 1: Ya).
    """)

    # Data Preprocessing
    age_bins = range(int(data['age'].min()), int(data['age'].max()) + 10, 10)
    data['age_group'] = pd.cut(data['age'], bins=age_bins, right=False, labels=[f"{i}-{i+9}" for i in age_bins[:-1]])

    # Descriptive Statistics
    st.subheader("Deskriptif Statistik Data")
    desc_stats = data.describe().T
    st.write("Deskriptif Statistik Data ini memberikan gambaran umum mengenai distribusi setiap kolom dalam dataset.")
    st.dataframe(desc_stats)

    # Distribution of Stroke Cases
    st.subheader("Distribusi Kasus Stroke")
    st.markdown("""
    Diagram pie ini menunjukkan proporsi pasien yang mengalami stroke dan yang tidak. 
    Hal ini membantu kita memahami prevalensi stroke dalam dataset.
    """)
    stroke_distribution = data['stroke'].map({0: 'Non Stroke', 1: 'Stroke'}).value_counts(normalize=True) * 100
    fig4, ax4 = plt.subplots(figsize=(6, 6))
    stroke_distribution.plot(kind='pie', autopct='%1.1f%%', colors=['#82caff', '#ff6961'], ax=ax4, startangle=90)
    ax4.set_title("Distribusi Kasus Stroke")
    ax4.set_ylabel("")
    st.pyplot(fig4)

    # Gender Distribution
    st.subheader("Distribusi Jenis Kelamin")
    st.markdown("""
    Visualisasi ini menunjukkan distribusi jenis kelamin pasien dalam dataset. 
    Informasi ini dapat digunakan untuk melihat apakah terdapat bias gender dalam data.
    """)
    # Filter data to include only 'Male' and 'Female'
    gender_filtered_data = data[data['gender'].isin(['Male', 'Female'])]
    gender_distribution = gender_filtered_data['gender'].value_counts(normalize=True) * 100
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    gender_distribution.plot(kind='bar', color=['#82caff', '#ff6961'], ax=ax5)
    ax5.set_title("Distribusi Jenis Kelamin")
    ax5.set_ylabel("Persen (%)")
    st.pyplot(fig5)

    # Work Type Distribution
    st.subheader("Distribusi Jenis Pekerjaan")
    st.markdown("""
    Visualisasi ini menunjukkan distribusi pasien berdasarkan jenis pekerjaan mereka. 
    Informasi ini dapat membantu mengidentifikasi kelompok pekerjaan yang mungkin lebih rentan terhadap stroke.
    """)
    work_type_distribution = data['work_type'].value_counts(normalize=True) * 100
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    work_type_distribution.plot(kind='bar', color='#ffa500', ax=ax6)
    ax6.set_title("Distribusi Jenis Pekerjaan")
    ax6.set_ylabel("persen (%)")
    st.pyplot(fig6)

    # Smoking Status Distribution
    st.subheader("Distribusi Status Merokok")
    st.markdown("""
    Visualisasi ini menunjukkan proporsi pasien berdasarkan status merokok. 
    Hal ini dapat membantu memahami peran kebiasaan merokok terhadap risiko stroke.
    """)
    smoking_status_distribution = data['smoking_status'].value_counts(normalize=True) * 100
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    smoking_status_distribution.plot(kind='bar', color='#9acd32', ax=ax7)
    ax7.set_title("Distribusi Status Merokok")
    ax7.set_ylabel("persen (%)")
    st.pyplot(fig7)

    # Correlation Matrix
    st.subheader("Analisis Korelasi")
    st.markdown("""
    Heatmap ini menunjukkan hubungan antara beberapa variabel numerik dalam dataset, 
    seperti usia, tingkat glukosa, BMI, hipertensi, dan riwayat penyakit jantung.
    """)
    correlation_matrix = data_clean.corr()
    fig8, ax8 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="viridis", ax=ax8)
    ax8.set_title("Analisis Korelasi")
    st.pyplot(fig8)

    # Distribusi berdasarkan kelompok usia
    st.subheader("Distribusi Stroke Berdasarkan Kelompok Usia")
    st.markdown("""
    Diagram batang bertumpuk ini menunjukkan persentase pasien dengan dan tanpa stroke di setiap kelompok usia.
    """)
    age_stroke = data.groupby('age_group')['stroke'].value_counts(normalize=True).unstack().fillna(0) * 100
    age_stroke.columns = age_stroke.columns.map({0: 'Non Stroke', 1: 'Stroke'})
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    age_stroke.plot(kind='bar', stacked=True, color=['#82caff', '#ff6961'], ax=ax1)
    ax1.set_title("Distribusi Stroke Berdasarkan Kelompok Usia")
    ax1.set_xlabel("Kelompok Usia")
    ax1.set_ylabel("persen (%)")
    ax1.legend(title="Stroke Status")
    st.pyplot(fig1)
    
    # Prevalensi Stroke Berdasarkan Kelompok Usia
    st.subheader("Prevalensi Stroke Berdasarkan Kelompok Usia")
    st.markdown("""
    Grafik garis ini menunjukkan persentase pasien dengan stroke pada setiap kelompok usia.
    Hal ini membantu mengidentifikasi tren stroke berdasarkan usia.
    """)
    age_stroke_prevalence = data.groupby('age_group')['stroke'].mean() * 100
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=age_stroke_prevalence.index, y=age_stroke_prevalence.values, marker='o', ax=ax2, color='#ffa500')
    ax2.set_title("Prevalensi Stroke Berdasarkan Kelompok Usia")
    ax2.set_xlabel("Kelompok Usia")
    ax2.set_ylabel("persen (%)")
    ax2.grid(axis='y')
    st.pyplot(fig2)

    # Distribusi stroke berdasarkan status pernikahan
    st.subheader("Distribusi Stroke Berdasarkan Status Pernikahan")
    st.markdown("""
    Diagram batang bertumpuk ini menunjukkan distribusi pasien dengan dan tanpa stroke berdasarkan status pernikahan.
    """)
    marital_stroke = data.groupby('ever_married')['stroke'].value_counts(normalize=True).unstack().fillna(0) * 100
    marital_stroke.columns = marital_stroke.columns.map({0: 'Non Stroke', 1: 'Stroke'})
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    marital_stroke.plot(kind='bar', stacked=True, color=['#82caff', '#ff6961'], ax=ax3)
    ax3.set_title("Stroke Distribution by Marital Status")
    ax3.set_xlabel("Marital Status")
    ax3.set_ylabel("persen (%)")
    ax3.legend(title="Stroke Status")
    st.pyplot(fig3)

elif selected == "Prediksi Risiko Stroke":
    st.title("🧠 Prediksi Penyakit Stroke")
    st.write("Gunakan alat ini untuk memprediksi kemungkinan terjadinya stroke berdasarkan beberapa faktor risiko.")

    # Memuat model dan scaler
    @st.cache_resource
    def load_scaler():
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return scaler

    @st.cache_resource
    def load_model():
        with open('KNN_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model

    scaler = load_scaler()
    model = load_model()

    # Membuat layout grid untuk input pengguna
    st.header("Masukkan Data Pasien")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio('Masukkan Jenis Kelamin', ['Perempuan','Laki-Laki'], index=1)
        heart_disease = st.radio("Apakah ada Riwayat Penyakit Jantung", ['Ya','Tidak'], index=1)
        ever_married = st.radio("Pernah Menikah", ['Ya','Tidak'], index=1)
        hypertension = st.radio("Hipertensi", ['Ya','Tidak'], index=1)
        residence_type = st.radio("Tipe Tempat Tinggal", ['Perkotaan','Pedesaan'], index=1)

    with col2:
        work_type = st.selectbox("Tipe Pekerjaan", ["Tidak Bekerja",  "Anak-anak",  "Rahasia", "Wiraswasta", "Pemerintah"], index=0)
        smoking_status = st.selectbox("Status Merokok", ["Tidak Pernah", "Pernah", "Masih Merokok", "Tidak Tahu"], index=0)
        age = st.number_input("Usia",  min_value=0, max_value=120, value=25)
        avg_glucose_level = st.number_input("Rata-rata Glukosa Darah", 0.0, 300.0, 100.0)
        bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
        
        # Encode work_type dan smoking_status ke nilai numerik
        work_type_encoded = {"Anak-anak": 0, "Pemerintah": 1, "Tidak Bekerja": 2, "Rahasia": 3, "Wiraswasta": 4}
        smoking_status_encoded = {"Pernah": 0, "Tidak Pernah": 1, "Masih Merokok": 2, "Tidak Tahu": 3}
        
        work_type_value = work_type_encoded[work_type]
        smoking_status_value = smoking_status_encoded[smoking_status]
    
        # Normalisasi avg_glucose_level dan bmi menggunakan MinMaxScaler
        input_data = np.array([[age, avg_glucose_level, bmi]])
        # Menormalkan kedua fitur
        normalized_data = scaler.transform(input_data)
        
    # Membuat dictionary untuk data input
    data_input = {
        'gender': [1 if gender == 'Laki-Laki' else 0],  # 1 untuk Perempuan, 0 untuk Laki-Laki
        'hypertension': [1 if hypertension == 'Ya' else 0],  # 1 untuk Ya, 0 untuk Tidak
        'heart_disease': [1 if heart_disease == 'Ya' else 0],  # 1 untuk Ya, 0 untuk Tidak
        'ever_married': [1 if ever_married == 'Ya' else 0],  # 1 untuk Ya, 0 untuk Tidak
        'work_type': [work_type_value],
        'residence_type': [1 if residence_type == 'Perkotaan' else 0],  # 1 untuk Perkotaan, 0 untuk Pedesaan
        'smoking_status': [smoking_status_value],
        'age (norm)': [normalized_data[0][0]],
        'avg_glucose_level (norm)': [normalized_data[0][1]],
        'bmi (norm)': [normalized_data[0][2]],
    }
    
    # Membuat DataFrame
    df = pd.DataFrame(data_input)

    # Pastikan data input adalah array 2D
    # input_array = df.values.reshape(1, -1)

    # Menampilkan DataFrame
    st.write("Data Input yang dimasukkan:", df)

    # Tombol prediksi
    if st.button("🔍 Prediksi"):
        prediction = model.predict(df)
        
        # Menampilkan hasil prediksi
        st.subheader("Hasil Prediksi")
        if prediction[0] == 1:
            st.error("❌ Prediksi: **Positif** - Pasien berisiko terkena stroke.")
        else:
            st.success("✅ Prediksi: **Negatif** - Pasien tidak berisiko terkena stroke.")
        
        # Menampilkan informasi tambahan
        st.markdown("""
        **Catatan:**
        - Hasil prediksi ini hanya sebagai alat bantu dan bukan diagnosis medis.
        - Konsultasikan dengan dokter untuk analisis lebih lanjut.
        """)

    