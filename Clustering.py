import pandas as pd  
import streamlit as st  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN  
from sklearn.metrics import silhouette_score  
from sklearn.preprocessing import StandardScaler  
import numpy as np  
import io  
from mpl_toolkits.mplot3d import Axes3D  
  
# Judul aplikasi  
st.title('Clustering Demografis: Analisis Segmen Berdasarkan Usia dan Pendapatan')  
  
# Fungsi untuk memuat data  
def load_data(file_path):  
    """Memuat data dari file CSV ke DataFrame pandas."""  
    return pd.read_csv(file_path)  
  
# Fungsi untuk menampilkan informasi dataset  
def display_dataset_info(data):  
    """Menampilkan informasi dataset termasuk deskripsi dan analisis univariat."""  
    with st.expander('Dataset'):  
        st.write(data)  
        st.success('Informasi Dataset')  
        buffer = io.StringIO()  
        data.info(buf=buffer)  
        s = buffer.getvalue()  
        st.text(s)  
  
# Fungsi untuk menampilkan visualisasi univariat  
def display_univariate_visualizations(data):  
    """Menampilkan visualisasi univariat untuk fitur-fitur tertentu."""  
    with st.expander('Visualisasi Univariate'):  
        st.info('Visualisasi per Kolom')  
          
        # Histogram untuk Usia  
        fig, ax = plt.subplots()  
        sns.histplot(data['Age'], color='blue', kde=True)  
        plt.xlabel('Usia')  
        plt.title('Distribusi Usia')  
        st.pyplot(fig)  
          
        # Histogram untuk Pendapatan  
        fig, ax = plt.subplots()  
        sns.histplot(data['Income'], color='red', kde=True)  
        plt.xlabel('Pendapatan')  
        plt.title('Distribusi Pendapatan')  
        st.pyplot(fig)  
          
        # Box Plot untuk Usia  
        fig, ax = plt.subplots()  
        sns.boxplot(x=data['Age'])  
        plt.title('Box Plot Usia')  
        st.pyplot(fig)  
          
        # Box Plot untuk Pendapatan  
        fig, ax = plt.subplots()  
        sns.boxplot(x=data['Income'])  
        plt.title('Box Plot Pendapatan')  
        st.pyplot(fig)  
          
        # Statistik Deskriptif  
        st.success('Statistik Deskriptif:')  
        st.write(data[['Age', 'Income']].describe())  
  
# Fungsi untuk menampilkan plot outlier  
def plot_outlier(data, column):  
    """Menampilkan plot outlier untuk fitur tertentu menggunakan box plot dan histogram."""  
    plt.figure(figsize=(12, 5))  
      
    # Box Plot  
    plt.subplot(1, 2, 1)  
    sns.boxplot(x=data[column])  
    plt.title(f'{column} - Box Plot')  
      
    # Histogram  
    plt.subplot(1, 2, 2)  
    sns.histplot(data[column], bins=30, kde=True)  
    plt.title(f'{column} - Histogram')  
      
    st.pyplot(plt)  
  
# Fungsi untuk menghapus outlier menggunakan IQR  
def remove_outlier(data, column):  
    """Menghapus outlier dari fitur tertentu menggunakan metode IQR."""  
    Q1 = data[column].quantile(0.25)  
    Q3 = data[column].quantile(0.75)  
    IQR = Q3 - Q1  
    lower = Q1 - 1.5 * IQR  
    upper = Q3 + 1.5 * IQR  
    return data[(data[column] >= lower) & (data[column] <= upper)]  
  
# Fungsi untuk menentukan jumlah cluster optimal menggunakan Elbow Method  
def determine_optimal_clusters(scaled_features):  
    """Menentukan jumlah cluster optimal menggunakan Elbow Method."""  
    distortions = []  
    K = range(1, 11)  
    for k in K:  
        kmeans = KMeans(n_clusters=k, random_state=42)  
        kmeans.fit(scaled_features)  
        distortions.append(kmeans.inertia_)  
      
    fig, ax = plt.subplots(figsize=(8, 6))  
    plt.plot(K, distortions, 'bx-')  
    plt.xlabel('Jumlah Cluster')  
    plt.ylabel('Distorsi')  
    plt.title('Elbow Method untuk Menentukan Jumlah Cluster Optimal')  
    st.pyplot(fig)  
      
    num_clusters = st.slider('Pilih jumlah cluster', min_value=2, max_value=10, value=3)  
    return num_clusters  
  
# Fungsi untuk melakukan clustering K-Means  
def perform_kmeans_clustering(scaled_features, num_clusters):  
    """Melakukan clustering menggunakan KMeans."""  
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)  
    kmeans.fit(scaled_features)  
    return kmeans  
  
# Fungsi untuk melakukan clustering Agglomerative  
def perform_agglomerative_clustering(scaled_features, num_clusters):  
    """Melakukan clustering menggunakan Agglomerative Clustering."""  
    agglomerative = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')  
    labels = agglomerative.fit_predict(scaled_features)  
    return labels  
  
# Fungsi untuk melakukan clustering DBSCAN  
def perform_dbscan_clustering(scaled_features):  
    """Melakukan clustering menggunakan DBSCAN."""  
    dbscan = DBSCAN(eps=0.5, min_samples=5)  
    labels = dbscan.fit_predict(scaled_features)  
    return labels  
  
# Fungsi untuk menampilkan visualisasi hasil clustering  
def display_clustering_results(data, labels, method):  
    """Menampilkan visualisasi hasil clustering."""  
    data['Cluster'] = labels  
      
    st.info('Visualisasi Hasil Clustering')  
      
    # Scatter plot untuk Usia vs Pendapatan  
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.scatterplot(x=data['Age'], y=data['Income'], hue=data['Cluster'], palette='viridis', s=100)  
    plt.xlabel('Usia')  
    plt.ylabel('Pendapatan')  
    plt.title(f'Scatter Plot Usia vs Pendapatan ({method})')  
    st.pyplot(fig)  
  
    # 3D Scatter Plot  
    if 'Education' in data.columns:  # Pastikan kolom 'Education' ada untuk 3D plot  
        fig = plt.figure(figsize=(10, 8))  
        ax = fig.add_subplot(111, projection='3d')  
        ax.scatter(data['Age'], data['Income'], data['Education'], c=data['Cluster'], cmap='viridis', s=100)  
        ax.set_xlabel('Usia')  
        ax.set_ylabel('Pendapatan')  
        ax.set_zlabel('Pendidikan')  
        plt.title(f'3D Scatter Plot ({method})')  
        st.pyplot(fig)  
  
# Fungsi untuk menghitung dan menampilkan skor silhouette  
def display_silhouette_score(scaled_features, labels):  
    """Menghitung dan menampilkan skor silhouette."""  
    if len(set(labels)) > 1:  # Pastikan ada lebih dari 1 cluster  
        silhouette_avg = silhouette_score(scaled_features, labels)  
        st.success(f'Skor Silhouette: {silhouette_avg:.2f}')  
    else:  
        st.warning('Silhouette score tidak dapat dihitung karena hanya ada satu cluster.')  
  
# Fungsi untuk menampilkan statistik deskriptif per cluster  
def display_cluster_statistics(data):  
    """Menampilkan statistik deskriptif per cluster."""  
    st.success('Statistik Deskriptif per Cluster:')  
    cluster_stats = data.groupby('Cluster').describe()  
    st.write(cluster_stats)  
  
# Fungsi untuk menampilkan data terfilter berdasarkan rentang usia dan pendapatan  
def display_filtered_data(data, age_range, income_range):  
    """Menampilkan data terfilter berdasarkan rentang usia dan pendapatan."""  
    if 'Cluster' not in data.columns:  
        st.warning('Clustering belum dilakukan. Silakan lakukan clustering terlebih dahulu.')  
        return  
      
    filtered_data = data[(data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1]) &  
                         (data['Income'] >= income_range[0]) & (data['Income'] <= income_range[1])]  
      
    st.write(f'Jumlah Data Terfilter: {filtered_data.shape[0]}')  
      
    if not filtered_data.empty:  
        fig, ax = plt.subplots(figsize=(10, 6))  
        sns.scatterplot(x=filtered_data['Age'], y=filtered_data['Income'], hue=filtered_data['Cluster'], palette='viridis', s=100)  
        plt.xlabel('Usia')  
        plt.ylabel('Pendapatan')  
        plt.title('Scatter Plot Usia vs Pendapatan (Data Terfilter)')  
        st.pyplot(fig)  
    else:  
        st.warning('Tidak ada data yang sesuai dengan rentang yang dipilih.')  
  
# Fungsi untuk menampilkan data berdasarkan cluster yang dipilih  
def display_data_by_cluster(data):  
    """Menampilkan data berdasarkan cluster yang dipilih."""  
    if 'Cluster' not in data.columns:  
        st.warning('Clustering belum dilakukan. Silakan lakukan clustering terlebih dahulu.')  
        return  
      
    selected_cluster = st.selectbox('Pilih Cluster untuk Dilihat', options=data['Cluster'].unique())  
    filtered_by_cluster = data[data['Cluster'] == selected_cluster]  
    st.write(f'Data untuk Cluster {selected_cluster}:')  
    st.write(filtered_by_cluster)  
  
# Main function to run the Streamlit app  
def main():  
    # Muat data  
    data = load_data('Clustering.csv')  
      
    # Navigasi menggunakan sidebar  
    st.sidebar.title('Navigasi')  
    page = st.sidebar.radio("Pilih Halaman", ["Dataset", "Visualisasi", "Clustering", "Interaksi"])  
      
    # Profil di sidebar  
    st.sidebar.title('Profil')  
    st.sidebar.write('Alif Dorisandi Ramadhan')  
    st.sidebar.write('211220097')  
      
    if page == "Dataset":  
        # Tampilkan informasi dataset  
        display_dataset_info(data)  
      
    elif page == "Visualisasi":  
        # Tampilkan visualisasi univariat  
        display_univariate_visualizations(data)  
          
        # Plot outliers untuk fitur-fitur tertentu  
        columns_to_plot = ['Age', 'Income']  
        for col in columns_to_plot:  
            plot_outlier(data, col)  
          
        # Hapus outlier untuk fitur-fitur tertentu  
        for col in columns_to_plot:  
            data = remove_outlier(data, col)  
          
        st.success('Dataset Terkini')  
        st.write(f'Dataset : {data.shape}')  
      
    elif page == "Clustering":  
        # Clustering expander  
        with st.expander('Clustering'):  
            st.info('Pilih Fitur untuk Clustering')  
              
            # Pilihan fitur untuk clustering  
            selected_features = ['Age', 'Income']  
              
            # Standarisasi data  
            scaler = StandardScaler()  
            scaled_features = scaler.fit_transform(data[selected_features])  
              
            # Tentukan jumlah cluster menggunakan Elbow Method  
            num_clusters = determine_optimal_clusters(scaled_features)  
              
            # Pilih metode clustering  
            clustering_method = st.selectbox("Pilih Metode Clustering", ["K-Means", "Agglomerative", "DBSCAN"])  
              
            if clustering_method == "K-Means":  
                kmeans = perform_kmeans_clustering(scaled_features, num_clusters)  
                display_clustering_results(data, kmeans.labels_, "K-Means")  
                display_silhouette_score(scaled_features, kmeans.labels_)  
              
            elif clustering_method == "Agglomerative":  
                agglomerative_labels = perform_agglomerative_clustering(scaled_features, num_clusters)  
                display_clustering_results(data, agglomerative_labels, "Agglomerative Clustering")  
                display_silhouette_score(scaled_features, agglomerative_labels)  
              
            elif clustering_method == "DBSCAN":  
                dbscan_labels = perform_dbscan_clustering(scaled_features)  
                display_clustering_results(data, dbscan_labels, "DBSCAN")  
                display_silhouette_score(scaled_features, dbscan_labels)  
              
            # Statistik Deskriptif per Cluster  
            display_cluster_statistics(data)  
      
    elif page == "Interaksi":  
        # Sidebar untuk interaksi  
        with st.sidebar:  
            st.header('Interaksi')  
            age_range = st.slider('Rentang Usia', min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=(int(data['Age'].min()), int(data['Age'].max())))  
            income_range = st.slider('Rentang Pendapatan', min_value=int(data['Income'].min()), max_value=int(data['Income'].max()), value=(int(data['Income'].min()), int(data['Income'].max())))  
              
            # Tampilkan data terfilter  
            display_filtered_data(data, age_range, income_range)  
              
            # Tampilkan data berdasarkan cluster yang dipilih  
            display_data_by_cluster(data)  
          
        # Penyimpanan dan Unduhan Hasil Clustering  
        st.download_button(  
            label="Unduh Hasil Clustering",  
            data=data.to_csv(index=False).encode('utf-8'),  
            file_name='hasil_clustering.csv',  
            mime='text/csv'  
        )  
  
# Run the main function  
if __name__ == '__main__':  
    main()  
