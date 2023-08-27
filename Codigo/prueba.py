import scipy.io
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sneia_tools import SNEIATools


# Cargar el archivo MAT
data = scipy.io.loadmat('MD5E-s111284160--f39bd1e24dbf93dd25b4df19d989baf7.set')

# Ver las claves (nombres de variables) en el archivo MAT
#print(data.keys())

chanlocs=data['chanlocs']  # los nombres de los electrodos estan en la key 'chanlocs'

# chanlocs tiene una estructura muy rara y larga por lo que se hacen los siguientes
# dos pasos para sacar en una lista solo con los nombres de los canales

ch_name = [chanlocs[0][i][0] for i in range(0,64)]

ch_names = [item[0] for item in ch_name]

señales = data['data'] # los datos estan en la key 'data'

#print(señales.shape[0])
#print(señales.shape[1])
#print(señales[:3,:])

señales_primera_capa = señales[:,:,0]

datos_invertidos = np.transpose(señales_primera_capa)

# Los siguientes dos pasos se hace porque la primer funcion que se aplica es read_data
# y a ella le debe entrar una matriz de m muestras por n electrodos más una primer 
# columna con los indices (numero de fila)

# Generar la secuencia de números enumerados desde 0
numeros_enumerados = np.arange(datos_invertidos.shape[0]).reshape(-1, 1)

# Añadir la columna de números enumerados a filas_columnas_primera_capa
dataset_csv = np.hstack((numeros_enumerados, datos_invertidos))

# Se guarda la matriz final como un archivo .csv
np.savetxt('dataset3_preprocesados.csv', dataset_csv, delimiter=',', fmt='%s')  

# Ruta completa del archivo CSV
ruta_archivo = 'dataset3_preprocesados.csv' # se usa la matriz que habiamos guardado anteriormente

# Cargar el archivo CSV
data = np.genfromtxt(ruta_archivo, delimiter=',', dtype=float, skip_header=1)

# Obtener los datos de los electrodos
tools = SNEIATools()
indice_muestra, electrodos = tools.read_data(data)

# Calcular la gfp para los electrodos
gfp = tools.get_gfp(electrodos)

# Constante que define el porcentaje para el umbral
PERCENTAGE = 0.7

# Obtener los microestados (secuencias en la gfp que superan 60 ms de actividad)
ms_sequence_indexes, ms_sequence_values = tools.get_microstates_sequences(gfp, percentage=PERCENTAGE, sample_frq=256)
print("microstates_indexes: (", len(ms_sequence_indexes), ")\n", ms_sequence_indexes)
print("microstates_values: (", len(ms_sequence_values), ")\n", ms_sequence_values)

ms_electrodes_values = tools.get_electrodes_value(ms_sequence_indexes, electrodos)
print("valores de los electrodos:\n", ms_electrodes_values)
print()

# Obtener las muestras (muestras individuales en la gfp que pertenecen a una secuencia de 60 ms o más)
sample_indexes, sample_values = tools.get_microstates_samples(gfp, percentage=PERCENTAGE, sample_frq=256)
print("sample_indexes: (", len(sample_indexes), ")\n", sample_indexes)
print("sample_values: (", len(sample_values), ")\n", sample_values)

sample_electrodes_values = tools.get_electrodes_value(sample_indexes, electrodos)
print("valores de los electrodos:\n", sample_electrodes_values)
print()

# Preparación de gráfico de gfp
plt.plot(indice_muestra, gfp, label="gfp")
plt.axhline(np.max(gfp)*PERCENTAGE, color='red', linestyle='--')

# Agregar etiquetas y leyenda
plt.xlabel('Índice de muestra')
plt.ylabel('Amplitud')
plt.legend()

# Mostrar el gráfico de la gfp
plt.show()

# Clusterizamos
df = tools.apply_kmeans(sample_electrodes_values)

# Obtener series de tiempo
time_series = tools.get_time_series(df)

# Presentar mapas topográficos
tools.time_series_topomap(time_series[0],ch_names,256,"standard_1020")
tools.time_series_topomap(time_series[1],ch_names,256,"standard_1020")
tools.time_series_topomap(time_series[2],ch_names,256,"standard_1020")
tools.time_series_topomap(time_series[3],ch_names,256,"standard_1020")
