import numpy as np
import streamlit as st
import pandas as pd
import pickle
import skfuzzy as fuzz

model = pickle.load(open('kmeans.pkl', 'rb'))
features = pd.read_csv("data_survei_softcom.csv")
feature_columns = [
    "IPK_Rendah", "IPK_Sedang", "IPK_Tinggi",
    "Pendapatan_Ortu_Rendah", "Pendapatan_Ortu_Sedang", "Pendapatan_Ortu_Tinggi",
    "Tanggungan_Keluarga_Sedikit", "Tanggungan_Keluarga_Banyak",
    "Jumlah_Pencapaian_Sedikit", "Jumlah_Pencapaian_Banyak",
    "Tingkat_Motivasi_Rendah", "Tingkat_Motivasi_Sedang", "Tingkat_Motivasi_Tinggi",
    "Potensi_Kepemimpinan_Rendah", "Potensi_Kepemimpinan_Sedang", "Potensi_Kepemimpinan_Tinggi",
    "Urgensi_Finansial_Sedikit", "Urgensi_Finansial_Banyak",
    "Kondisi_Pribadi_Sedikit", "Kondisi_Pribadi_Banyak"
]

def get_membership(df, col, n_terms=3):
    x_min = df[col].min()
    x_max = df[col].max()
    x_range = np.linspace(x_min, x_max, 100)

    if n_terms == 3:
        rendah = fuzz.trimf(x_range, [x_min, x_min, (x_min + x_max) / 2])
        sedang = fuzz.trimf(x_range, [x_min, (x_min + x_max) / 2, x_max])
        tinggi = fuzz.trimf(x_range, [(x_min + x_max) / 2, x_max, x_max])

        df[f'{col}_Rendah'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, rendah, v))
        df[f'{col}_Sedang'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, sedang, v))
        df[f'{col}_Tinggi'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, tinggi, v))

    elif n_terms == 4:
        q1 = x_min + (x_max - x_min) * 0.25
        q2 = x_min + (x_max - x_min) * 0.5
        q3 = x_min + (x_max - x_min) * 0.75

        sangat_sedikit = fuzz.trimf(x_range, [x_min, x_min, q1])
        sedikit = fuzz.trimf(x_range, [x_min, q1, q2])
        banyak = fuzz.trimf(x_range, [q2, q3, x_max])
        sangat_banyak = fuzz.trimf(x_range, [q3, x_max, x_max])

        df[f'{col}_Sangat_Sedikit'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, sangat_sedikit, v))
        df[f'{col}_Sedikit'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, sedikit, v))
        df[f'{col}_Banyak'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, banyak, v))
        df[f'{col}_Sangat_Banyak'] = df[col].apply(
            lambda v: fuzz.interp_membership(x_range, sangat_banyak, v))

    return df


def get_membership_single(value, min_val, max_val, mode='3'):
    x_range = np.linspace(min_val, max_val, 100)

    if mode == '3':
        rendah = fuzz.trimf(
            x_range, [min_val, min_val, (min_val + max_val) / 2])
        sedang = fuzz.trimf(
            x_range, [min_val, (min_val + max_val) / 2, max_val])
        tinggi = fuzz.trimf(
            x_range, [(min_val + max_val) / 2, max_val, max_val])

        return [
            round(fuzz.interp_membership(x_range, rendah, value), 3),
            round(fuzz.interp_membership(x_range, sedang, value), 3),
            round(fuzz.interp_membership(x_range, tinggi, value), 3)
        ]

    elif mode == '4':
        q1 = min_val + (max_val - min_val) * 0.25
        q2 = min_val + (max_val - min_val) * 0.5
        q3 = min_val + (max_val - min_val) * 0.75

        sangat_sedikit = fuzz.trimf(x_range, [min_val, min_val, q1])
        sedikit = fuzz.trimf(x_range, [min_val, q1, q2])
        banyak = fuzz.trimf(x_range, [q2, q3, max_val])
        sangat_banyak = fuzz.trimf(x_range, [q3, max_val, max_val])

        return [
            round(fuzz.interp_membership(x_range, sangat_sedikit, value), 3),
            round(fuzz.interp_membership(x_range, sedikit, value), 3),
            round(fuzz.interp_membership(x_range, banyak, value), 3),
            round(fuzz.interp_membership(x_range, sangat_banyak, value), 3)
        ]

st.title("Prediksi Eligibilitas Kandidat Beasiswa Menggunakan Fuzzy Membership")

ipk = st.number_input("IPK", 0.0, 4.0, 3.0)
pendapatan = st.number_input("Pendapatan Orang Tua", 0.0, 1_000_000_000.0, 3_000_000.0)
tanggungan = st.slider("Tanggungan Keluarga", 1, 5, 3)
pencapaian = st.slider("Jumlah Pencapaian", 0, 5, 2)
motivasi = st.slider("Tingkat Motivasi", 1, 5, 5)
kepemimpinan = st.slider("Potensi Kepemimpinan", 1, 5, 5)
urgensi = st.slider("Urgensi Finansial", 1, 5, 4)
kondisi = st.slider("Kondisi Pribadi", 1, 5, 5)

stat = {
    'IPK': (features['IPK'].min(), features['IPK'].max()),
    'Pendapatan_Ortu': (features['Pendapatan_Ortu'].min(), 30000000),
    'Tanggungan_Keluarga': (features['Tanggungan_Keluarga'].min(), features['Tanggungan_Keluarga'].max()),
    'Jumlah_Pencapaian': (features['Jumlah_Pencapaian'].min(), features['Jumlah_Pencapaian'].max()),
    'Tingkat_Motivasi': (features['Tingkat_Motivasi'].min(), features['Tingkat_Motivasi'].max()),
    'Potensi_Kepemimpinan': (features['Potensi_Kepemimpinan'].min(), features['Potensi_Kepemimpinan'].max()),
    'Urgensi_Finansial': (features['Urgensi_Finansial'].min(), features['Urgensi_Finansial'].max()),
    'Kondisi_Pribadi': (features['Kondisi_Pribadi'].min(), features['Kondisi_Pribadi'].max()),
}

fuzzy_vector = []

fuzzy_vector += get_membership_single(ipk, *stat['IPK'], '3')
fuzzy_vector += get_membership_single(pendapatan, *stat['Pendapatan_Ortu'], '3')
fuzzy_vector += get_membership_single(tanggungan, *stat['Tanggungan_Keluarga'], '4')
fuzzy_vector += get_membership_single(pencapaian, *stat['Jumlah_Pencapaian'], '4')
fuzzy_vector += get_membership_single(motivasi, *stat['Tingkat_Motivasi'], '3')
fuzzy_vector += get_membership_single(kepemimpinan, *stat['Potensi_Kepemimpinan'], '3')
fuzzy_vector += get_membership_single(urgensi, *stat['Urgensi_Finansial'], '4')
fuzzy_vector += get_membership_single(kondisi, *stat['Kondisi_Pribadi'], '4')

if st.button("Prediksi Kelayakan"):
    input_array = np.array(fuzzy_vector).reshape(-1, 1)
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
        test_data=input_array,
        cntr_trained=model.cluster_centers_,
        m=2,
        error=0.005,
        maxiter=1000
    )
    
    cluster_membership = u[:, 0]
    predicted_cluster = np.argmax(cluster_membership)

    if predicted_cluster == 1:
        st.success(f"Hasil Prediksi: Anda **cenderung layak** menerima beasiswa")
    elif predicted_cluster == 0:
        st.warning(f"Hasil Prediksi: Anda **cenderung tidak layak** menerima beasiswa")
    st.write("Persentase Kemungkinan:")
    st.markdown(f"Layak {round(cluster_membership[1], 2) * 100}%")
    st.markdown(f"Tidak Layak {round(cluster_membership[0], 2) * 100}%")
    
    # cluster = model.predict(input_array)[0]
    # if cluster == 1:
    #     st.success(f"Hasil Prediksi: Anda **layak** menerima beasiswa")
    # elif cluster == 0: 
    #     st.warning(f"Hasil Prediksi: Anda **tidak layak** menerima beasiswa")

    # st.subheader("Detail Fuzzy Vector:")
    # fuzzy_vector = np.array(fuzzy_vector)
    # fuzzy_df = pd.DataFrame(fuzzy_vector.reshape(1, -1), columns=feature_columns)
    # st.dataframe(fuzzy_df)