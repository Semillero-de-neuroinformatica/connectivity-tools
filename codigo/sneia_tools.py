import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne
import scipy.io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SNEIATools():
    """ Esta clase se encarga de agrupar los métodos necesarios para el tratamiento de EEG
    """

    def read_data(self, data: np.ndarray):
        """ Retorna los indices y los electrodos de un EEG (data)

        Args:
            data (np.ndarray): Datos de un EEG

        Returns:
            tuple (np.array, np.array):
                (indices_muestra, electrodos)
                electrodos es un arreglo de arreglos, cada uno representa los datos de un electrodo
        """
        indice_muestra = data[:, 0]

        electrodos = np.empty((data.shape[0], data.shape[1] - 1))
        for i in range(data.shape[1] - 1):
            electrodos[:,i] = data[:, i + 1]

        return indice_muestra, electrodos

    def get_gfp(self, electrodes: np.ndarray):
        """ Retorna la GFP de un grupo de electrodos

        Args:
            electrodes (np.ndarray):
                Contiene los valores de un EEG, separado por electrodos

        Returns:
            GFP: Potencia de campo global de un grupo de electrodos
        """
        N = electrodes.shape[1]
        v_mean = np.mean(electrodes, axis=1)
        GFP = np.zeros(electrodes.shape[0])

        for i in range(0,N):
            GFP = GFP+(electrodes[:,i]-v_mean)**2
        GFP = np.sqrt(GFP/N)

        return GFP

    def get_microstates_sequences(self, gfp: np.ndarray, percentage: float, sample_frq: int):
        """ Retorna una tupla con los indices y los valores de la GFP que superan el
            umbral, separados por microestado

        Args:
            gfp (np.ndarray): Potencia de campo global GFP
            percentage (float): Porcentaje de umbral

        Returns:
            tuple (np.array, np.array):
                (indices_sequencia, valores_sequencia)
                Ambos arreglos pueden ser NO homogeneos. Se crean de tipo object
        """
        min_length_sequence = np.ceil(60*sample_frq/1000)
        max_value = np.max(gfp)
        indices_mayores = np.where(gfp > max_value*percentage)
        
        sequence_indexes = []
        sequence = np.empty((0,), dtype=int)

        for i in range(len(indices_mayores[0])):
            if i == 0 or indices_mayores[0][i] == indices_mayores[0][i-1] + 1:
                sequence = np.append(sequence, indices_mayores[0][i])
            else:
                if len(sequence) >= min_length_sequence:
                    sequence_indexes.append(sequence)
                sequence = np.array([indices_mayores[0][i]], dtype=int)

        if len(sequence) >= min_length_sequence:
            sequence_indexes.append(sequence)

        sequence_indexes = np.array(sequence_indexes, dtype=object)
        sequence_values = np.array(
            [gfp[sequence_indexes[i]] for i in range(len(sequence_indexes))],
            dtype=object
        )

        return sequence_indexes, sequence_values

    def get_microstates_samples(self, gfp: np.ndarray, percentage: float, sample_frq: int):
        """ Retorna una tupla con los indices y los valores de las muestras del
            gfp que superan el umbral, sin separar por microestado

        Args:
            gfp (np.ndarray): Potencia de campo global GFP
            percentage (float): Porcentaje de umbral

        Returns:
            tuple (np.array, np.array): (indices_muestras, valores_muestras)
        """
        min_length_sequence = np.ceil(60*sample_frq/1000)
        max_value = np.max(gfp)
        indices_mayores = np.where(gfp > max_value*percentage)

        sequences = []
        sequence = []

        for i in range(len(indices_mayores[0])):
            if i == 0 or indices_mayores[0][i] == indices_mayores[0][i-1] + 1:
                sequence.append(indices_mayores[0][i])
            else:
                if len(sequence) >= min_length_sequence:
                    sequences += sequence
                sequence = [indices_mayores[0][i]]

        if len(sequence) >= min_length_sequence:
            sequences += sequence

        return np.array(sequences), gfp[sequences]

    def get_electrodes_value(self, indices: np.array, electrodes: np.array): # type: ignore
        """ Retorna un array con los valores de cada electrodo para los indices dados.

        Args:
            indices (np.array): Indices separados por microestado
            electrodes (np.array): Array con los valores de los electrodos

        Returns:
            np.array:
                Array que contiene los valores de los electrodos por cada muestra
                que supera el umbral. Si los indices están separados por microestado,
                el array también lo estará. En el caso de que NO estén
                separados por microestado, al array contiene todas las muestras independientes
        """
        electrodes_values = []

        for indice in indices:
            electrodes_values.append(electrodes[indice])

        return np.array(electrodes_values, dtype=object)

    def apply_kmeans(self, value_matrix):
        """ Aplicación de kmeans para agrupar una lista de microestados alrededor
            de 4 microestados generales.

        Args:
            value_matrix (np.array):
                Matriz de valores de EEG, separados por microestado. Es decir, cada
                fila de la matriz representa los n valores para cada uno de los n
                electrodos del EEG que representan un microestado

        Returns:
            df (pandas.DataFrame):
                Dataframe que contiene la matriz de valores de la EEG que representan
                un microestado y el cluster al que pertenece cada uno
        """
        scaler = StandardScaler()
        value_matrix_norm = scaler.fit_transform(value_matrix)
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(value_matrix_norm)

        df = pd.DataFrame(value_matrix)
        df['cluster'] = kmeans.labels_

        return df

    def get_time_series(self, df):
        """ Aplicación de kmeans para agrupar una lista de microestados alrededor
            de 4 microestados generales.

        Args:
            df (pandas.DataFrame):
                Dataframe que contiene la matriz de valores de la EEG que representan
                un microestado y el cluster al que pertenece cada uno

        Returns:
            time_series (list): 
                Lista que contiene los microestados separados por el cluster al que 
                pertenece. Cada serie de tiempo representa un cluster
        """
        time_series = []

        for i in range(4):
            cluster_data = df[df['cluster'] == i].drop('cluster', axis=1)
            time_series.append(cluster_data.values)

        return time_series

    def time_series_topomap(self, serie, channels, freq: int, standard: str):
        """ Imprime mapa topográfico a partir de una serie de tiempo
        
        Args:
            serie (np.array):
                Serie de tiempo
            
            channels ():
                Lista de nombres de los canales
            
            freq (integer):
                Frecuencia de muestreo
                
            standard (string): 
                Cadena con información del estandar usado para la EEG. Por ej: "standard_1020"
        """
        ch_types_str = ['eeg']*len(channels)

        info = mne.create_info(ch_names=channels, sfreq=freq, ch_types=ch_types_str)
        raw_data = mne.io.RawArray(serie.T, info)
        raw_data.set_montage(standard)

        _, ax = plt.subplots(figsize=(5, 5))
        mne.viz.plot_topomap(raw_data.get_data()[:, 2], raw_data.info, axes=ax)
        plt.show()
