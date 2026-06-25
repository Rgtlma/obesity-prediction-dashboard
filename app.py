import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Obesity Prediction Dashboard",
    page_icon="⚕️",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

/* Background utama */
.stApp {
    background: linear-gradient(
        135deg,
        #EFF6FF,
        #F8FAFC
    );
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #2563EB,
        #10B981
    );
}

/* Tulisan sidebar */
[data-testid="stSidebar"] * {
    color: white;
}

/* Card */
div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Judul */
h1 {
    color: #1E3A8A;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "ObesityDataSet_raw_and_data_sinthetic.csv"
)

model = pickle.load(
    open(
        "obesity_model.pkl",
        "rb"
    )
)

encoder = pickle.load(
    open(
        "encoder.pkl",
        "rb"
    )
)

target_encoder = pickle.load(
    open(
        "target_encoder.pkl",
        "rb"
    )
)

importance = pd.read_csv(
    "feature_importance.csv"
)

# ==========================
# SIDEBAR
# ==========================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=100
)

menu = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "📊 EDA",
        "🤖 Model Performance",
        "📈 Feature Importance",
        "🔍 Prediction"
    ]
)

# ==========================
# HOME
# ==========================

if menu == "🏠 Home":

    st.markdown("""
    <div class="hero">

    <h1>⚕️ Obesity Prediction Dashboard</h1>

    <p>
    Analyze obesity levels using machine learning
    and interactive data visualization.
    </p>

    </div>
    """,
    unsafe_allow_html=True)

    st.write(
        """
        Dashboard ini digunakan untuk memprediksi tingkat obesitas
        berdasarkan karakteristik individu dan gaya hidup.
        """
    )

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "📊Jumlah Data",
        len(df)
    )

    col2.metric(
        "🧩Jumlah Fitur",
        16
    )

    col3.metric(
        "🏷 Jumlah Kelas",
        df["NObeyesdad"].nunique()
    )

    col4.metric(
        "🥇Best Accuracy",
        "95.51%"
    )

    st.subheader(
        "Preview Dataset"
    )

    st.dataframe(
        df.head()
    )

# ==========================
# EDA
# ==========================

elif menu == "📊 EDA":

    st.title(
        "Exploratory Data Analysis"
    )

    gender_filter = st.multiselect(
        "Filter Gender",
        df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    filtered_df = df[
        df["Gender"].isin(
            gender_filter
        )
    ]

    fig1 = px.histogram(
        filtered_df,
        x="NObeyesdad",
        color="Gender",
        title="Distribusi Tingkat Obesitas"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.scatter(
        df,
        x="Weight",
        y="Height",
        color="NObeyesdad",
        title="Hubungan Berat dan Tinggi Badan"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.pie(
        filtered_df,
        names="NObeyesdad",
        title="Proportion of Obesity Classes"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ==========================
# MODEL PERFORMANCE
# ==========================

elif menu == "🤖 Model Performance":

    st.title(
        "Perbandingan Model Machine Learning"
    )

    hasil = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy":[
            0.8676,
            0.9551,
            0.9527
        ]

    })

    st.dataframe(
        hasil
    )

    best_model = hasil.loc[
        hasil["Accuracy"].idxmax(),
        "Model"
    ]

    best_acc = hasil["Accuracy"].max()

    col1,col2 = st.columns(2)

    col1.metric(
        "Best Model",
        best_model
    )

    col2.metric(
        "Best Accuracy",
        f"{best_acc:.2%}"
    )

    fig = px.bar(
        hasil,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        color_discrete_sequence=[
            "#2563EB",
            "#10B981",
            "#F59E0B"
        ]
    )

    fig.update_layout(
        template="plotly_white",
        title="Perbandingan Akurasi Model",
        title_x=0.5,
        xaxis_title="Model",
        yaxis_title="Accuracy",
        height=500
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================
# FEATURE IMPORTANCE
# ==========================

elif menu == "📈 Feature Importance":

    st.title(
        "📈 Faktor yang Memengaruhi Prediksi"
    )

    st.markdown(
        """
        Halaman ini menampilkan faktor-faktor yang paling berpengaruh
        terhadap hasil prediksi tingkat obesitas berdasarkan model
        Random Forest yang memiliki performa terbaik.
        """
    )

    # ==========================
    # UBAH NAMA FITUR
    # ==========================

    feature_name_map = {

        "Age":"Usia",
        "Gender":"Jenis Kelamin",
        "Height":"Tinggi Badan",
        "Weight":"Berat Badan",

        "CALC":"Konsumsi Alkohol",
        "FAVC":"Konsumsi Makanan Tinggi Kalori",
        "FCVC":"Frekuensi Konsumsi Sayuran",
        "NCP":"Jumlah Makan Utama",

        "SCC":"Monitoring Kalori",
        "SMOKE":"Merokok",

        "CH2O":"Konsumsi Air Putih",

        "family_history_with_overweight":"Riwayat Keluarga Obesitas",

        "FAF":"Aktivitas Fisik",

        "TUE":"Penggunaan Perangkat Elektronik",

        "CAEC":"Frekuensi Ngemil",

        "MTRANS":"Moda Transportasi"
    }

    importance_display = importance.copy()

    importance_display["Feature"] = (
        importance_display["Feature"]
        .map(feature_name_map)
    )

    # ==========================
    # TOP FEATURE
    # ==========================

    top_feature = (
        importance_display
        .iloc[0]["Feature"]
    )

    st.success(
        f"Faktor yang paling berpengaruh terhadap prediksi obesitas adalah **{top_feature}**."
    )

    # ==========================
    # SLIDER
    # ==========================

    top_n = st.slider(
        "Jumlah faktor yang ditampilkan",
        5,
        16,
        10
    )

    st.info(
        """
        Semakin besar nilai importance,
        semakin besar kontribusi faktor tersebut
        terhadap hasil prediksi model.
        """
    )

    # ==========================
    # TABEL
    # ==========================

    st.subheader(
        "Tabel Feature Importance"
    )

    st.dataframe(
        importance_display.head(top_n),
        use_container_width=True
    )

    # ==========================
    # GRAFIK
    # ==========================

    st.subheader(
        "Visualisasi Feature Importance"
    )

    fig = px.bar(

        importance_display.head(top_n),

        x="Importance",
        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale="Blues",

        text="Importance"

    )

    fig.update_layout(

        template="plotly_white",

        title="Faktor yang Paling Berpengaruh",

        title_x=0.5,

        height=550,

        plot_bgcolor="#F8FAFC",

        paper_bgcolor="#F8FAFC"

    )

    fig.update_traces(

        texttemplate="%{text:.3f}",

        textposition="outside"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================
    # SHAP
    # ==========================

    st.subheader(
        "Analisis SHAP"
    )

    st.markdown(
        """
        Visualisasi SHAP menunjukkan kontribusi masing-masing fitur
        terhadap keputusan model pada seluruh observasi.
        """
    )

    st.image(
        "shap.png",
        use_container_width=True
    )

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "🥇 Faktor Utama",
        importance_display.iloc[0]["Feature"]
    )

    col2.metric(
        "🥈 Faktor Kedua",
        importance_display.iloc[1]["Feature"]
    )

    col3.metric(
        "🥉 Faktor Ketiga",
        importance_display.iloc[2]["Feature"]
    )

    st.divider()

    # ==========================
    # INTERPRETASI
    # ==========================

    st.subheader(
        "Interpretasi"
    )

    st.info(
        """
        Berdasarkan hasil analisis, berat badan merupakan faktor yang
        paling berpengaruh dalam menentukan tingkat obesitas.

        Selain itu, tinggi badan, usia, frekuensi konsumsi sayuran,
        jenis kelamin, jumlah makan utama, aktivitas fisik,
        penggunaan perangkat elektronik, konsumsi air putih,
        dan frekuensi ngemil juga memberikan kontribusi terhadap
        hasil prediksi model.

        Temuan ini menunjukkan bahwa karakteristik fisik individu
        serta pola hidup sehari-hari memiliki peranan penting dalam
        menentukan tingkat obesitas seseorang.
        """
    )


# ==========================
# PREDICTION
# ==========================

elif menu == "🔍 Prediction":

    st.title(
        "🔍 Prediksi Tingkat Obesitas"
    )

    st.markdown(
        """
        Masukkan karakteristik individu dan gaya hidup untuk
        memprediksi tingkat obesitas menggunakan model
        Random Forest.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Usia (tahun)",
            min_value=1,
            max_value=100,
            value=25
        )

        gender = st.selectbox(
            "Jenis Kelamin",
            ["Female", "Male"]
        )

        height = st.number_input(
            "Tinggi Badan (meter)",
            min_value=1.00,
            max_value=2.50,
            value=1.70,
            step=0.01
        )

        weight = st.number_input(
            "Berat Badan (kg)",
            min_value=20.0,
            max_value=300.0,
            value=70.0
        )

        calc = st.selectbox(
            "Frekuensi Konsumsi Alkohol",
            ["no","Sometimes","Frequently","Always"],
            help="Frekuensi konsumsi minuman beralkohol"
        )

        favc = st.selectbox(
            "Sering Mengonsumsi Makanan Tinggi Kalori?",
            ["no","yes"]
        )

        fcvc = st.slider(
            "Frekuensi Konsumsi Sayuran",
            1.0,
            3.0,
            2.0,
            help="1 = rendah, 3 = tinggi"
        )

        ncp = st.slider(
            "Jumlah Makan Utama per Hari",
            1.0,
            4.0,
            3.0
        )

    with col2:

        scc = st.selectbox(
            "Memantau Konsumsi Kalori Harian?",
            ["no","yes"]
        )

        smoke = st.selectbox(
            "Kebiasaan Merokok",
            ["no","yes"]
        )

        ch2o = st.slider(
            "Konsumsi Air Putih",
            1.0,
            3.0,
            2.0,
            help="Semakin tinggi menunjukkan konsumsi air yang lebih banyak"
        )

        family = st.selectbox(
            "Riwayat Keluarga Mengalami Obesitas",
            ["no","yes"]
        )

        faf = st.slider(
            "Frekuensi Aktivitas Fisik",
            0.0,
            3.0,
            1.0,
            help="0 = tidak pernah, 3 = sangat sering"
        )

        tue = st.slider(
            "Durasi Penggunaan Perangkat Elektronik",
            0.0,
            3.0,
            1.0,
            help="TV, laptop, komputer, atau smartphone"
        )

        caec = st.selectbox(
            "Frekuensi Ngemil di Antara Waktu Makan",
            ["no","Sometimes","Frequently","Always"]
        )

        mtrans = st.selectbox(
            "Moda Transportasi Utama",
            [
                "Walking",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Automobile"
            ]
        )


    st.divider()
    predict_button = st.button(
        "🔎 Prediksi Sekarang",
        use_container_width=True
    )
    
    if predict_button:

        input_df = pd.DataFrame({

            'Age':[age],
            'Gender':[gender],
            'Height':[height],
            'Weight':[weight],
            'CALC':[calc],
            'FAVC':[favc],
            'FCVC':[fcvc],
            'NCP':[ncp],
            'SCC':[scc],
            'SMOKE':[smoke],
            'CH2O':[ch2o],
            'family_history_with_overweight':[family],
            'FAF':[faf],
            'TUE':[tue],
            'CAEC':[caec],
            'MTRANS':[mtrans]

        })

        for col in encoder.keys():

            if col in input_df.columns:

                input_df[col] = encoder[col].transform(
                    input_df[col]
                )

        import time

        with st.spinner(
            "Analyzing..."
        ):

            progress = st.progress(0)

            for i in range(100):

                time.sleep(0.01)

                progress.progress(i+1)
        
        pred = model.predict(
            input_df
        )

        hasil = target_encoder.inverse_transform(
            pred
        )

        prediction = hasil[0]

        label_map = {
            "Insufficient_Weight":"Berat Badan Kurang",
            "Normal_Weight":"Berat Badan Normal",
            "Overweight_Level_I":"Kelebihan Berat Badan Tingkat I",
            "Overweight_Level_II":"Kelebihan Berat Badan Tingkat II",
            "Obesity_Type_I":"Obesitas Tipe I",
            "Obesity_Type_II":"Obesitas Tipe II",
            "Obesity_Type_III":"Obesitas Tipe III"
        }

        hasil_indonesia = label_map[prediction]

        if "Normal" in prediction:

            st.balloons()

            st.success(
                f"Hasil Prediksi : {hasil_indonesia}"
            )

        elif "Overweight" in prediction:

            st.warning(
                f"Hasil Prediksi : {hasil_indonesia}"
            )

        else:

            st.error(
                f"Hasil Prediksi : {hasil_indonesia}"
            )

        risk_map = {
            "Insufficient_Weight":20,
            "Normal_Weight":40,
            "Overweight_Level_I":60,
            "Overweight_Level_II":70,
            "Obesity_Type_I":80,
            "Obesity_Type_II":90,
            "Obesity_Type_III":100
        }


        risk_score = risk_map[prediction]

        if risk_score <= 40:
            risk_label = "Low Risk"
            risk_desc = "Berat badan berada pada rentang yang relatif aman."

        elif risk_score <= 70:
            risk_label = "Moderate Risk"
            risk_desc = "Perlu memperhatikan pola makan dan aktivitas fisik."

        else:
            risk_label = "High Risk"
            risk_desc = "Disarankan melakukan pengelolaan berat badan dan konsultasi lebih lanjut."

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,

                title={
                    "text":"Risk Level"
                },

                gauge={
                    "axis":{
                        "range":[0,100]
                    },

                    "bar":{
                        "color":"darkblue"
                    },

                    "steps":[
                        {
                            "range":[0,40],
                            "color":"lightgreen"
                        },
                        {
                            "range":[40,70],
                            "color":"yellow"
                        },
                        {
                            "range":[70,100],
                            "color":"salmon"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if risk_score <= 40:
            st.success(
                f"🟢 {risk_label}"
            )

        elif risk_score <= 70:
            st.warning(
                f"🟡 {risk_label}"
            )

        else:
            st.error(
                f"🔴 {risk_label}"
            )

        st.info(
            risk_desc
        )


# ==========================
