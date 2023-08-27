===============
 Metodología 1
===============

Para esta motodología se tienen los siguientes pasos

**Preparar la lectura de los datos**

Se procesa el archivo de la base de datos. Se extraen la cantidad y los 
nombres de los canales, se enumeran las muestras y se inserta la enumeración en
los datos en caso de ser necesario. 
Luego exportamos los datos preparados en un archivo .csv

**Obtener los datos de la EEG**

Se toman el archivo .csv preparado en el paso anterior y se lee, para luego 
separar los índices de las muestras y los valores de los electrodos

**Calcular la GFP**

Se calcula el campo global de potencia (GFP) para los valores de los electrodos.

**Obtener los microestados**

Para esta metodología, en la que estamos teniendo en cuenta un factor de duración
en el tiempo, consideramos que un microestado hará presencia en el momento en que
en la GFP se presenten *picos de potencia* con una duración mayor a 60 ms. 
Un *pico de potencia* se refiere a un pico en la GFP que supere un *umbral*. 
Para esta metodología, el umbral es arbitrario y se entrega como parámetro a la 
función que verifica la existencia de los microestados.

Cada microestado estará representado por una serie de indices en la GFP (los
cuales concuerdan con los índices de la EEG) que son *secuenciales*. 

**Obtener los valores de cada microestado**

Una vez tenemos los índices que representan un microestado, hallamos los valores
que representan cada uno de estos índices en la EEG. Estos valores pueden o no 
ser agrupados por microestado, según cómo se hayan obtenido los microestados, 
si agrupados (sequencias) o no (muestras). Cada índice representa una muestra, 
la cual tendrá tantos elementos como canales tenga la EEG.

**Clusters**

A partir de los valores de los microestados obtenidos, buscamos agruparlos en 
*4* microestados generales usando *Kmeans*.

**Obtener series de tiempo**

Separamos los microestados según el microestado general al que pertenecen (cluster)
en series de tiempo

**Presentamos los mapas topográficos**

Escojemos un microestado *arbitrario*, de una serie de tiempo, para presentar
su mapa topográfico
