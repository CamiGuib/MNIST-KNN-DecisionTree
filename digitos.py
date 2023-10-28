#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
TP 2
grupo ACA
Materia: Laboratorio de datos - FCEyN - UBA
Autores :   Antony Suarez
            Camila Guibaudo
            Ariel Dembling
Fecha  : 14-06-2023

Tiempo estimado requerido para ejecución: 35 minutos

'''
#########################################
##### Imports
#########################################
from time import time, strftime, localtime
from datetime import timedelta
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text, plot_tree
from sklearn import metrics

#%%
#########################################
##### Carga de datos
#########################################

rootdir = '~/LaboDatos/TP2'
file_path = '/data_sets/'

df = pd.read_csv(rootdir + file_path + "mnist_desarrollo.csv", header=None)
dftestbin = pd.read_csv(rootdir + file_path + "mnist_test_binario.csv", header=None)
dftest = pd.read_csv(rootdir + file_path + "mnist_test.csv", header=None)


#%%
#########################################
##### Funciones
#########################################

#####
# Esta función calcula el tiempo transcurrido durante el lapso
# pasado como parámetro..
# Para medir el tiempo que toma una función, hacer:
#   start = time()
#   correr_funcion()
#   end = time()
#   print(secs2str(end-start))
def secs2str(segundos=None):
    if segundos is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    return str(timedelta(seconds=segundos))

######
# Devuelve un vector con los valores de vec rescalados al
# intervalo [newmin, newmax].
def reescalar_vector(vec, newmin, newmax):
    newvec = np.interp(vec, (vec.min(), vec.max()), (newmin, newmax))
    return newvec


######
# Proyector de labels del dataframe:
# La columna 0 contiene la etiqueta correspondiente a los datos
# contenidos en el resto de la fila:
def labels(df):
    return pd.DataFrame(df.iloc[:,0])

# Proyector de imagenes del dataframe:
def imagenes(df):
    return df.iloc[:,1:].set_axis(range(len(df.columns)-1), axis="columns")

# Devuelve lista con las etiquetas distintas existentes en el dataframe:
def labels_distintos(df):
    return list(set(labels(df).values.ravel()))

#####
# Dado un dataframe conteniendo la tabla MNIST o un subconjunto de filas
# de ella, más un número de fila, esta función devuelve la
# etiqueta correspondiente a dicha fila (que es un int).
def label(df, imgidx):
    return (labels(df).iloc[imgidx,:]).loc[0]

# Dado un dataframe conteniendo la tabla MNIST o un subconjunto de filas
# de ella, más un número de fila, esta función devuelve la serie
# de datos de la imagen contenida en dicha fila.
def imagen(df, imgidx):
    return imagenes(df).iloc[imgidx,:]

######
# Devuelve el tamaño (pixels) de las imágenes contenidas en
# el dataframe.
def cant_pixels_por_imagen(df):
    return imagenes(df).shape[1]

# Devuelve las cantidades y proporciones de cada clase contenida
# en el dataframe df.
def analizar_clases(df):
    tot = df.shape[0]
    cantl=0
    for l in labels_distintos(df):
        cantl = filtro_label(df,l).sum()
        print("Cantidad y porcentaje de imágenes etiquetadas como", l, "en el dataframe:", cantl, " ", round(cantl*100/tot,2))

# Devuelve los valores mínimo y máximo de grises considerando
# todas las imágenes del dataframe df.
# Si verbose==True, también emite texto describiendo el dataframe.
def describir_matriz(df, verbose=True):

    ims = imagenes(df)
    pixels_list = list(np.unique(ims))
    tipos_de_datos_en_pixels = list(set(type(i) for i in pixels_list))
    vmin = min(pixels_list)
    vmax = max(pixels_list)

    if verbose:
        # Cantidad de datos:
        print("Tamaño de la matriz:", df.shape)

        labels_list = labels_distintos(df)
        print("Valores de las etiquetas:", labels_list)

        print("Datos de los atributos:", tipos_de_datos_en_pixels, "con valores entre", vmin, "y", vmax)

        px_por_imagen = cant_pixels_por_imagen(df)
        print("Cantidad total de pixels por imagen:", px_por_imagen)
        ppl = int(np.sqrt(px_por_imagen))
        if ppl**2 != px_por_imagen:
            print("ATENCIÓN: el dataframe provisto no contiene una imagen cuadrada.")
            ppl=0
        else:
            print("La imagen es cuadrada.")
            print("Cantidad de pixels por lado:", ppl)

        # Cantidad de imágenes:
        print("Cantidad de imágenes:", df.shape[0])

        # Cantidad de clases de la variable de interés (etiqueta),
        print("Cantidad de clases:", len(labels_list))

    return (vmin, vmax)

######
# Función que cuenta la cantidad de muestras en las que cierta posición de
# pixel contiene trazos. (Si un pixel de una muestra contiene un valor mayor
# que cero, entonces contiene un trazo con alguna intensidad que no nos
# interesa aquí; caso contrario, el pixel no contiene trazo en esa muestra).
def contar_trazos_en_pixelpos(df, pixelpos):
    ima = imagenes(df)
    maxpos = len(ima.columns)
    if pixelpos < 0 or pixelpos > maxpos-1:
        print("ERROR: el parámetro pixelpos para este dataframe debe valer entre 0 y "+str(len(ima.columns)-1))
        return np.nan
    return sum(ima[pixelpos]>0)

# Función que devuelve un dataframe con valores entre 0 y 255, compatible con
# una imagen raster MNIST, donde 255 representa a el o los pixels con la
# cantidad máxima de trazos en las muestras del dataframe provisto, y 0 la
# cantidad mínima de trazos.
def trazos_por_pixel(df):
    tam = cant_pixels_por_imagen(df)
    im=np.zeros(tam)
    for i in range(tam):
        im[i]=contar_trazos_en_pixelpos(df,i)
    return pd.DataFrame(reescalar_vector(im, 0, 255).astype(np.int64))


# Función de relevancia basada en el criterio de "trazo en rango":
# se priorizan los pixels del dataframe trazo tr cuyo valor está
# comprendido en el intervalo [inf, sup]. Generaliza las funciones
# de los criterios específicos, definidas más abajo.
def relevancia_por_pixel_trazo_en_rango(df, tr, inf, sup):
    tam = cant_pixels_por_imagen(df)
    im=np.zeros(tam)
    for i in range(tam):
        if int(tr.iloc[i]) > inf and int(tr.iloc[i]) < sup:
            im[i]=255
        else:
            im[i]=int(int(tr.iloc[i])/2)
    return pd.DataFrame(im)

# Función de relevancia basada en el criterio de "trazo máximo":
# se priorizan los pixels del dataframe trazo tr cuyo valor ronde el
# valor máximo, 255 (con un margen dado por el parámetro eps). La idea es
# que, a priori, un pixel por el cual pasan los trazos de una gran cantidad
# de muestras puede contener más información sobre el dataset que aquellos
# pixels por el cual pasa una cantidad menor de trazos.
def relevancia_por_pixel_trazo_maximo(df, tr, eps=50):
    return relevancia_por_pixel_trazo_en_rango(df, tr, 255-eps, 256)

# Función de relevancia basada en el criterio de "trazo medio":
# se priorizan los pixels del dataframe trazo tr cuyo valor ronde el
# valor medio, 255/2 (con un margen dado por el parámetro eps).
def relevancia_por_pixel_trazo_medio(df, tr, eps=10):
    return relevancia_por_pixel_trazo_en_rango(df, tr, 255/2-eps, 255/2+eps)

# Función de relevancia basada en el criterio de "trazo único":
# se priorizan los pixels del dataframe de trazo tr cuyo valor ronde 255/n,
# donde n es la cantidad de dígitos representados en el dataframe df
# provisto. Cuando n==2, equivale al criterio de "trazo medio".
# La idea es que un pixel por el cual pasan los trazos de un único dígito,
# y no de los demás, debería rondar (con un margen eps) dicho valor 255/n.
# Los pixels con valores en 0 son irrelevantes. Sin embargo, los valores
# máximos tampoco son demasiado relevantes ya que la información que aportan
# es escasa pues tienden a ser ocupados por trazos de varios o todos
# los dígitos. Los pixels más informativos idealmente serían aquellos
# que solo son ocupados por (todos los) trazos de un único dígito, lo cual
# ocurriría en aquellos que ronden un valor de 255/n (con n la cantidad de
# dígitos distintos en el dataframe).
# Desde ya, nada garantiza que no pueda tenerse un dataset donde, por ejemplo,
# un pixel valga 255/n pero la mitad de dicho valor en tr provenga de
# un dígito y la otra mitad de otro dígito. Sin embargo, la relativa
# homogeneidad en el recorrido de los trazos de un mismo dígito haría
# razonable que este no sea un caso frecuente, y que un pixel que sea
# compartido por los trazos de dos dígitos distintos tienda a formar parte
# de todas las imágenes de ambos dígitos, aproximando entonces 2*255/n y no
# 255/n. Por supuesto, esto es un razonamiento simplificado a priori,
# no hay garantías sobre las ventajas de este criterio en general y
# seguramente se pueden crear otros criterios más convenientes o generales.
def relevancia_por_pixel_trazo_unico(df, tr, eps=10):
    n = len(labels_distintos(df))
    return relevancia_por_pixel_trazo_en_rango(df, tr, 255/n-eps, 255/n+eps)

# Esta función devuelve la lista de columnas de un dataframe de
# relevancia reldf tales que los valores son iguales a 255, que es
# el valor asignado en las funciones de relevancia.
def obtener_lista_de_pixelpos(reldf):
    return list(reldf[reldf.iloc[:]==255].dropna().index.values)

# Dada una serie de datos correspondiente a una imagen MNIST
# (cuadrada, greyscale de 8 bits), esta función genera su gráfico.
# Recibe como parámetros los valores de gris mínimo vmin y máximo vmax.
def plotear_raster(imagen, vmin, vmax, titulo=""):
    dim = int(np.sqrt(imagen.shape[0]))
    plt.imshow(imagen.values.reshape((dim,dim)), cmap="binary", vmin=vmin, vmax=vmax)
    plt.title(titulo)
    plt.show()
    plt.close()

# Dado un dataframe conteniendo la tabla MNIST o un subconjunto de filas
# de ella, esta función genera el gráfico de su i-ésima fila y lo titula
# con la etiqueta que tiene asociada en el dataframe.
def plotear_mnist(df, i):
    vmin, vmax = describir_matriz(df, verbose=False)
    plotear_raster(imagen(df, i), vmin, vmax, "Etiqueta: "+str(label(df, i)))

# Dado un dataframe conteniendo la tabla MNIST o un subconjunto de filas
# de ella, y dada una etiqueta, esta función devuelve un filtro
# sobre ese dataframe indicando las filas cuya etiqueta coincide con la dada.
def filtro_label(df, label):
    return df.iloc[:,0]==label

######
# Esta función toma un set de datos (separados en atributos X y etiquetas Y)
# más un número de vecinos y devuelve una serie con las predicciones del
# modelo KNN entrenado con esos parámetros.
def entrenar_knn(X, Y, neighbors):
    modelo = KNeighborsClassifier(n_neighbors = neighbors) # modelo abstracto
    modelo.fit(X, Y.values.ravel()) # entrenamos el modelo con los datos X e Y
    return modelo

# Esta función aplica un modelo dado al set de atributos X y devuelve
# las predicciones del modelo sobre dicho set.
def predecir(modelo, X):
    # vemos qué clases les asigna el modelo a los datos de X
    return modelo.predict(X)

# Esta función recibe dos sets de datos (uno de training y otro
# de validación), separados en frames de atributos y de etiquetas,
# y aplica el modelo kNN (para un k también dado).
# Devuelve los dos valores de exactitud logrados para ambos sets.
# También devuelve el modelo ajustado.
def ajustar_kNN(X_train, Y_train, X_valid, Y_valid, k):
    # entrenamos el modelo sobre el set de training,
    # ajustado para el k dado.
    modelo = entrenar_knn(X_train, Y_train, k)
    # aplicamos el modelo a los sets de training y validación,
    # obteniendo las predicciones del modelo
    Y_train_pred = predecir(modelo, X_train)
    Y_valid_pred = predecir(modelo, X_valid)
    # medimos la exactitud lograda para cada set, comparando
    # con las etiquetas provistas
    acc_train = metrics.accuracy_score(Y_train, Y_train_pred)
    acc_valid = metrics.accuracy_score(Y_valid, Y_valid_pred)
    # devolvemos los valores de exactitud obtenidos, y el modelo
    return (acc_train, acc_valid, modelo)


def validar_knn(X_devel, Y_devel, X_test, Y_test, k, test_size=0.3, stratify=True):
    # Por default dividimos el dataset en forma estratificada tomando
    # las clases desde Y_devel.
    if stratify:
        stratify = Y_devel
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_devel, Y_devel, test_size=test_size, stratify=stratify)
    modelo = entrenar_knn(X_train, Y_train, k)
    Y_valid_pred = predecir(modelo, X_valid)
    Y_test_pred = predecir(modelo, X_test)
    ### Ojo, tenemos cruzada la nomenclatura valid <--> test
    ## Para consistencia con la nomenclatura utilizada en todas las
    ## funciones de exploración y de kNN, debería ser así:
    #return {'Y_valid_pred': Y_valid_pred, 'Y_test_pred': Y_test_pred, 'Y_valid':Y_valid}
    ## Pero para compatibilidad con la función metricas_de_performance()
    ## que sigue otra nomenclatura, devuelvo esta estructura con nombres
    ## cruzados:
    return {'Y_train_pred_test': Y_valid_pred, 'Y_valid_pred': Y_test_pred, 'Y_test':Y_valid}


# Esta función recibe datasets de training y de validación, e
# itera sobre una lista de valores de k dada, llamando a
# ajustar_kNN() sobre cada uno de ellos.
# Devuelve una lista con los resultados de accuracy obtenidos para
# cada dataset, y los modelos ajustados.
def kNN_iterando_sobre_k_con_train_y_valid(X_train, Y_train, X_valid, Y_valid, klist):
    # Se reciben sets de training y de validación.
    # Se corre kNN sobre ambos sets, sobre la lista de k dada.

    acc_train_vector_knn = np.zeros((len(klist)))
    acc_valid_vector_knn = np.zeros((len(klist)))
    lista_modelos=[]

    # Itero sobre la lista de k provista
    for k in klist:
        # Obtenemos los valores de exactitud de training y de
        # validación tras ajustar un modelo kNN para el valor
        # de k actual
        acc_train, acc_valid, modelo = ajustar_kNN(X_train, Y_train, X_valid, Y_valid, k)
        # Guardamos los valores de exactitud obtenidos en la
        # repetición actual
        acc_train_vector_knn[k-1] = acc_train
        acc_valid_vector_knn[k-1] = acc_valid
        lista_modelos.append(modelo)
        print("Vecinos de kNN (k):", k, "Exactitud training:", acc_train, "Exactitud validación:", acc_valid)

    lista_acc_train = list(acc_train_vector_knn)
    lista_acc_valid = list(acc_valid_vector_knn)

    # devolvemos listas con los valores de exactitud
    # obtenidos para los sets de training y de validación,
    # correspondientes a la lista klist provista.
    return (lista_acc_train, lista_acc_valid, lista_modelos)


def kNN_iterando_sobre_atributos_con_train_y_valid(X_train, Y_train, X_valid, Y_valid, maxk, sets_de_atributos):
    # Guardaremos los modelos y resultados para cada set de atributos
    # en este diccionario:
    resultados_por_set_de_atributos = {}

    # Sobre esos sets de training y validación, corremos kNN para cada
    # conjunto de atributos definido.
    for s in sets_de_atributos:

        print("Utilizando", s)

        X_train_sel_atr = X_train[sets_de_atributos[s]]
        X_valid_sel_atr = X_valid[sets_de_atributos[s]]

        # Lista de valores k a intentar
        klist = range(1,maxk)

        # Entrenamos modelos ajustándolos a cada k. Los guardamos para
        # poder volver a utilizarlos posteriormente.
        # Obtenemos listas de exactitudes para cada k y cada set.
        lista_acc_train, lista_acc_valid, lista_modelos = kNN_iterando_sobre_k_con_train_y_valid(X_train_sel_atr, Y_train, X_valid_sel_atr, Y_valid, klist)

        # Graficamos exactitud vs. k para los sets de training y validación.
        plotear_knn(klist, lista_acc_train, "Training "+s, lista_acc_valid, "Validación "+s, 1)

        resultados_por_set_de_atributos[s] = {"klist": klist, "lista_acc_train": lista_acc_train, "lista_acc_valid": lista_acc_valid, "lista_modelos": lista_modelos}

    return resultados_por_set_de_atributos

def kNN_iterando_sobre_atributos(X_devel, Y_devel, maxk, sets_de_atributos, test_size=0.3, stratify=True):
    # Por default dividimos el dataset en forma estratificada tomando
    # las clases desde Y_devel.
    if stratify:
        stratify = Y_devel

    # De lo que hay en development, separamos 70% para training y
    # 30% para validación. Repartido mediante muestreo estratificado
    # para mantener las proporciones de ceros y unos en los subsets.
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_devel, Y_devel, test_size=test_size, stratify=stratify)
    return kNN_iterando_sobre_atributos_con_train_y_valid(X_train, Y_train, X_valid, Y_valid, maxk, sets_de_atributos)


# Esta función aplica modelos kNN con la lista de valores k provista.
# Lo hace sobre un frame de datos de desarrollo provisto (separado en
# un frame de atributos y otro de etiquetas).
# Al realizar el procedimiento, divide el frame de datos de desarrollo
# en subsets de entrenamiento y de validación, y por cada k ajusta el
# modelo kNN a ese k sobre ambos subsets, obteniendo valores de exactitud
# para cada uno.
# Recibe también un diccionario de listas o slices de atributos.
# Cada ítem del diccionario es aplicado separadamente como filtro
# a los frames de atributos X_train y X_valid. Si el diccionario es
# vacío, se seleccionan todos los atributos del frame.
# Devuelve dos diccionarios de listas conteniendo la exactitud resultante de
# los subsets para cada k y cada conjunto de atributos.
# Cuando reps>1, repite este procedimiento reps veces y los vectores
# devueltos contienen promedios de las exactitudes observadas,
# constituyendo un método de cross-validation.
def kNN_repetido_con_atributos(X_devel, Y_devel, reps, maxk, dicsel=False):

    # Si no me pasan conjunto de listas de atributos o slices, creo
    # un slice para seleccionar todos los atributos.
    if not dicsel:
        dicsel={}
        #dicsel['Todos los atributos'] = slice(0,X_devel.shape[1]-1,None)
        dicsel['Todos los atributos'] = slice(None,None,None)

    # Hacemos varias repeticiones que luego promediaremos.
    # En cada repetición se parten de manera diferente los
    # frames de development en frames de training y de
    # validación.

    resultados_por_rep={}

    # Diccionario de matrices de resultados obtenidos para cada set
    # de atributos
    matriz_resultados_train={}
    matriz_resultados_valid={}
    # Diccionario de listas de resultados promedio obtenidos por cada
    # set de atributos
    lista_avg_acc_train={}
    lista_avg_acc_valid={}
    for s in dicsel:
        matriz_resultados_train[s] = np.zeros((reps, maxk))
        matriz_resultados_valid[s] = np.zeros((reps, maxk))
        lista_avg_acc_train[s]=[]
        lista_avg_acc_valid[s]=[]

    klist=[]
    for rep in range(reps):
        print("Repetición:", rep)
        resultados_por_rep[rep] = kNN_iterando_sobre_atributos(X_devel, Y_devel, maxk, dicsel)
        # resultados_por_rep es un diccionario cuyas claves son
        # los valores de rep, y su valor es un diccionario cuyas
        # claves son los valores de dicsel, y cuyos valores son
        # diccionarios con 4 claves posibles:
        #   "klist": rango de valores de k,
        #   "lista_acc_train": lista de exactitudes resultantes de
        #       aplicar el modelo al set de training, relativa a k,
        #   "lista_acc_valid": lista de exactitudes resultantes de
        #       aplicar el modelo al set de validación, relativa a k,
        #   "lista_modelos": lista de los modelos, relativa a k, para
        #       poder reutilizarlos.
        for s in resultados_por_rep[rep]:
            klist = resultados_por_rep[rep][s]["klist"]
            for k in klist:
                matriz_resultados_train[s][rep, k-1] = resultados_por_rep[rep][s]["lista_acc_train"][k-1]
                matriz_resultados_valid[s][rep, k-1] = resultados_por_rep[rep][s]["lista_acc_valid"][k-1]

    # Calculamos los promedios
    res_por_set_de_atributos={}
    for s in dicsel:
        lista_avg_acc_train[s] = list(np.mean(matriz_resultados_train[s], axis = 0))
        lista_avg_acc_valid[s] = list(np.mean(matriz_resultados_valid[s], axis = 0))
        res_por_set_de_atributos[s] = {"klist": klist,
                "lista_acc_train": lista_avg_acc_train[s], 
                "lista_acc_valid": lista_avg_acc_valid[s], 
                "lista_modelos": None}


    # Devolvemos dos diccionarios con listas de valores de
    # exactitud promedio obtenidos para los sets de training
    # y de validación, correspondientes a la lista klist provista.
    # Las claves de los diccionarios son los nombres de los
    # conjuntos de atributos provistos.
    #return (lista_avg_acc_train, lista_avg_acc_valid)

    # Devolvemos un diccionarios con listas de valores de
    # exactitud promedio obtenidos para los sets de training
    # y de validación, correspondientes a la lista klist provista,
    # también inclida en el diccionario.
    # Las claves de los diccionarios son los nombres de los
    # conjuntos de atributos provistos.
    return res_por_set_de_atributos

def plotear_knn(klist, accu1_list, accu1_label, accu2_list, accu2_label, reps):
    plt.plot(klist, accu1_list, label = accu1_label)
    plt.plot(klist, accu2_list, label = accu2_label)
    plt.legend()
    plt.title('Exactitud del modelo de knn con '+str(reps)+' iteraciones')
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Exactitud (accuracy)')
    plt.show()
    plt.close()

# Función que toma los modelos y resultados obtenidos en el
# development y aplica dichos modelos sobre un dataset de
# testing provisto. Emite gráficos comparando la exactitud
# lograda con el dataset de testing versus la obtenida durante
# la validación.
def testear_y_graficar_exactitud(resultados_por_set_de_atributos, X_test, Y_test):
    for s in resultados_por_set_de_atributos:
        klist = resultados_por_set_de_atributos[s]["klist"]
        lista_acc_train = resultados_por_set_de_atributos[s]["lista_acc_train"]
        lista_acc_valid = resultados_por_set_de_atributos[s]["lista_acc_valid"]
        lista_modelos = resultados_por_set_de_atributos[s]["lista_modelos"]

        acc_test_vector_knn = np.zeros((len(klist)))

        for i_k in range(len(klist)):
            k = klist[i_k]
            Y_test_pred = predecir(lista_modelos[i_k], X_test[sets_de_atributos[s]])
            acc_test_vector_knn[k-1] = metrics.accuracy_score(Y_test, Y_test_pred)

        lista_acc_test = list(acc_test_vector_knn)

        # Graficamos exactitud vs. k para los sets de validación y testing.
        plotear_knn(klist, lista_acc_valid, "Validación "+s, lista_acc_test, "Testing "+s,1)



def imag(df):
    return df[list(df.columns[1:])]

def entrenar_tree(X, Y, profundidad=4, criterio=''):
    if criterio == '':
        modelo = DecisionTreeClassifier(max_depth=profundidad)
    else:
        modelo = DecisionTreeClassifier(criterion=criterio, max_depth=profundidad) # modelo abstracto
    modelo.fit(X, Y) # entrenamos el modelo con los datos X e Y
    return modelo


def plotear_tree(tree, X_lista, Y, plot_type='text', tamaño_fuente=8, tamaño_imagen=[20,10]):
    if plot_type == 'img':
        plt.figure(figsize=tamaño_imagen)
        plot_tree(tree, feature_names = X_lista, class_names = Y,filled = True, rounded = True, fontsize = tamaño_fuente)
    elif plot_type == 'text':
        text_plot = export_text(tree, feature_names=X_lista)
        print(text_plot)
    else:
        print("Plot_type incorrecto")

def plotear_resultados(dicc_res, dicc_rotulos):
    plt.figure(figsize=dicc_rotulos['tamaño imagen'])
    claves = list(dicc_res.keys())
    valores = list(dicc_res.values())
    plt.plot(claves, valores)
    plt.xlabel(dicc_rotulos['label x'])
    plt.ylabel(dicc_rotulos['label y'])
    plt.title(dicc_rotulos['titulo'])
    plt.show()
    plt.close()


def validar_tree(df_train, df_valid, profundidad=4, criterio="entropy", test_ratio=0.3):
    X = imag(df_train)
    Y = labels(df_train)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_ratio, stratify=Y)
    # entrenamos el modelo sobre el set de training
    modelo = DecisionTreeClassifier(criterion=criterio, max_depth=profundidad) # modelo abstracto
    modelo.fit(X_train, Y_train) # entrenamos el modelo con los datos X e Y
    # aplicamos el modelo a los sets de training y validación,
    # obteniendo las predicciones del modelo
    Y_train_pred_test = predecir(modelo, X_test)
    # Validamos con df_valid
    X_train = imag(df_valid)
    Y_valid_pred = predecir(modelo, X_train)
    # devolvemos los valores de exactitud obtenidos
    return {'Y_train_pred_test': Y_train_pred_test, 'Y_valid_pred': Y_valid_pred, 'Y_test':Y_test}


def arboles_de_decision_por_prof(df,prof_max=4,criterio='entropy',n_folds=5, verbose=True):
    X = imagenes(df)
    Y = labels(df)
    exactitudes = {}
    skf = StratifiedKFold(n_splits = n_folds)
    for i in range(1,prof_max+1):
        modelo = DecisionTreeClassifier(criterion = criterio, max_depth = i) #Creamos el árbol de decisión
        exactitudes_por_arbol = [] #Lista de exactitudes para este árbol en particular. Va a tener tantos valores como folds haya.
        for j, (train_index, test_index) in enumerate(skf.split(X,Y)):  #Los datos de X son los que va a dividir; a Y lo necesita para que se verifique la estratificación
            X_train = X.loc[train_index]
            X_valid = X.loc[test_index]
            Y_train = Y.loc[train_index]
            Y_valid = Y.loc[test_index]
            modelo = modelo.fit(X_train, Y_train) #Entrenamos el árbol de decisión
            Y_pred = modelo.predict(X_valid) #Usamos el modelo para predecir los valores de X_valid
            accuracy = metrics.accuracy_score(Y_valid, Y_pred) #Calculamos la accuracy comparando los valores de Y_pred con Y_valid
            exactitudes_por_arbol.append(accuracy) #Registramos esa accuracy
        promedio_acc_arbol = sum(exactitudes_por_arbol)/len(exactitudes_por_arbol)
        exactitudes[i] = promedio_acc_arbol #Registramos la acc promedio de este árbol en el diccionario de acc
        if verbose:
            print(f"Profundidad {i} : ", exactitudes_por_arbol)
            print(f"Promedio accuracy árbol de profundidad {i} -> {promedio_acc_arbol}")
    return exactitudes

# Métricas
# Confusion matrix, Accuracy, Precision, TP rate, Harmonic mean

def metricas_de_performance(Y_true, Y_pred, es_multi_label=True, verbose=True):
    dicc = {}
    dicc['accuracy'] = metrics.accuracy_score(Y_true, Y_pred)
    dicc['precision'] = metrics.precision_score(Y_true, Y_pred, average='macro')
    dicc['cubrimiento'] = metrics.recall_score(Y_true, Y_pred, average='macro')
    if es_multi_label:
        dicc['confusion matrix'] = metrics.multilabel_confusion_matrix(Y_true, Y_pred, labels=list(range(10)))
    else:
        dicc['confusion matrix'] = metrics.multilabel_confusion_matrix(Y_true, Y_pred)
    if verbose:
        print("Medidas de perfomance")
        print(f"Exactitud (accuracy) = {dicc['accuracy']}")
        print(f"Cubrimiento (recall) = {dicc['cubrimiento']}")
        print(f"Precisión (precision) = {dicc['precision']}")
        print("---------------------------------------------")
        print(f"Matriz de confusión = {dicc['confusion matrix']}")
        print("---------------------------------------------")
    return dicc


def rotularDF(df,rotulo='',label_Y='label'):
    columnas = [label_Y]
    for index in range(df.shape[1]-1):
        columnas.append(rotulo+'_'+str(index))
    df.columns = columnas


def ajuste_por_criterio_importancia_atributos(df, criterio_relevancia='trazo unico'):
    trazo = trazos_por_pixel(df)
    if criterio_relevancia == 'trazo unico':
        df_r = relevancia_por_pixel_trazo_unico(df, trazo)
    elif criterio_relevancia == 'trazo maximo':
        df_r = relevancia_por_pixel_trazo_maximo(df, trazo)
    elif criterio_relevancia == 'existencia de trazos':
        print('aca va aalgoo')

    return df_r

#%%
#########################################
##### Resto del código
#########################################

#%%
# Ejercicio 1: análisis exploratorio.


# Se observa que las etiquetas son los dígitos de 0 a 9.
# El resto de las columnas contiene una imagen en escala de grises.
# Cada columna contiene, para una fila dada, un pixel que toma valores
# entre 0 y 255. Esto se evidencia porque pixels_list es de tipo lista
# de np.int64 (y no de np.float64), y por los valores de vmin y vmax.
# La imagen es cuadrada, de 28x28.
vmin, vmax = describir_matriz(df)

# Hay 10 clases y sus cantidades de muestras están relativamente
# equilibradas en el dataset.
analizar_clases(df)

#%%

## Relevancia de atributos.
# No todos los pixels de la imagen resultan igualmente relevantes para
# la identificación del dígito, ya que el trazo de los dígitos no se
# extiende a todos los pixels por igual.

# Calculamos la cantidad de trazos que pasan por cada pixel,
# reescalada al intervalo [0,255]. Esta operación toma hasta
# un par de minutos, dependiendo del tamaño del dataset.
trazos = trazos_por_pixel(df)

# Generamos su imagen.
plotear_raster(trazos, vmin, vmax, "Superposición de trazos por pixel")

#%%
# Calculamos la relevancia de cada pixel aplicando cierto criterio (ver
# descripción del criterio en el comentario de la definición de la función
# invocada aquí).
relevancia = relevancia_por_pixel_trazo_maximo(df, trazos, 10)
#relevancia = relevancia_por_pixel_trazo_unico(df, trazos, 10)
#relevancia = relevancia_por_pixel_trazo_medio(df, trazos, 10)

# Graficamos la relevancia. Los pixels con valor 255 son los
# considerados relevantes por la función aplicada. Los demás
# pixels corresponden a trazos sin relevancia para el criterio utilizado,
# y se incluyen aquí para dar contexto a los pixels seleccionados.
plotear_raster(relevancia, vmin, vmax, "Relevancia por pixel")

#%%

## Graficación por muestra específica.
# Fila de la imagen a graficar. El título del gráfico indica
# la etiqueta correspondiente a la fila.
imgidx = 511
plotear_mnist(df, imgidx)

#%%
# Ejercicio 2: dataframe df01 conteniendo solo los dígitos 0 y 1.

# Obtenemos un subset con los dígitos 0 y 1 únicamente:
df01 = df[filtro_label(df, 0) | filtro_label(df, 1)]

# Se observa que las etiquetas de df01 son los dígitos 0 y 1.
vmin, vmax = describir_matriz(df01)

#%%
# Ejercicio 3: análisis del dataframe df01

# Se observa que las cantidades de muestras de cada clase
# están relativamente equilibradas en el dataset.
analizar_clases(df01)

#%%
# Ejercicio 4: kNN sobre df01 con distintos sets de atributos. Análisis
# de resultados.

# Para los ejercicios 4 y 5 trabajaremos únicamente sobre el
# dataset df01. Pero acá lo parametrizamos por si se quiere probar
# con otro dataset.
# Nota: para trabajar con el datasets más grandes, algunos
# cálculos requieren una cantidad de RAM significativa (más de 8 GB).
df_ej4 = df01

# Nota: kNN es sensible a la escala de los distintos atributos, pero como
# todos los atributos (pixels) varían en el mismo intervalo [0,255],
# no es necesario reescalarlos aquí.

#%%
# Calculamos la cantidad de trazos que pasan por cada pixel
# en el dataset actual. Esta operación toma hasta un par de minutos,
# dependiendo del tamaño del dataset.
trazos_ej4 = trazos_por_pixel(df_ej4)
plotear_raster(trazos_ej4, vmin, vmax, "Superposición de trazos por pixel")

#%%

# Calculamos algunos criterios de relevancia para este dataset
relevancia_ej4_a = relevancia_por_pixel_trazo_maximo(df_ej4, trazos_ej4, 5)
plotear_raster(relevancia_ej4_a, vmin, vmax, "Relevancia por pixel")

relevancia_ej4_b = relevancia_por_pixel_trazo_maximo(df_ej4, trazos_ej4, 150)
plotear_raster(relevancia_ej4_b, vmin, vmax, "Relevancia por pixel")

relevancia_ej4_c = relevancia_por_pixel_trazo_unico(df_ej4, trazos_ej4, 10)
plotear_raster(relevancia_ej4_c, vmin, vmax, "Relevancia por pixel")

relevancia_ej4_d = relevancia_por_pixel_trazo_medio(df_ej4, trazos_ej4, 10)
plotear_raster(relevancia_ej4_d, vmin, vmax, "Relevancia por pixel")

#%%
## Seleccionamos conjuntos de atributos

maxcol = cant_pixels_por_imagen(df_ej4)-1

# sel3random: selector de 3 atributos al azar en runtime
r1 = random.randint(0,maxcol)
r2 = random.randint(0,maxcol)
r3 = random.randint(0,maxcol)
sel3random = [r1, r2, r3]

# sel3random2: selector de 3 atributos al azar específicos
# (Con sucesivas ejecuciones de sel3random encontramos
# conjuntos de atributos que han dado "buenos" resultados
# en algunas corridas. Los ponemos aquí para compararlos en
# otra corrida).
#[r1, r2, r3] = [298, 315, 460]
[r1, r2, r3] = [507, 189, 343]
sel3random2 = [r1, r2, r3]

# selrelevantes_a: selector de atributos relevantes, criterio A
selrelevantes_a = obtener_lista_de_pixelpos(relevancia_ej4_a)

# selrelevantes_b: selector de atributos relevantes, criterio B
selrelevantes_b = obtener_lista_de_pixelpos(relevancia_ej4_b)

# selrelevantes_c: selector de atributos relevantes, criterio C
selrelevantes_c = obtener_lista_de_pixelpos(relevancia_ej4_c)

# selrelevantes_d: selector de atributos relevantes, criterio D
selrelevantes_d = obtener_lista_de_pixelpos(relevancia_ej4_d)

# selpares: selector de todos los atributos pares
selpares = slice(0,maxcol,2)

# seltodos: selector de todos los atributos
seltodos = slice(0,maxcol,None)

### Definimos los conjuntos de atributos que nos interesará explorar.
## set más completo
#sets_de_atributos = {"Set 1: 3 atributos al azar en runtime": sel3random,
#                     "Set 2: 3 atributos al azar preseleccionados": sel3random2,
#                     "Set 3: los atributos más relevantes, crit. A": selrelevantes_a,
#                     "Set 4: los atributos más relevantes, crit. B": selrelevantes_b,
#                     "Set 5: los atributos más relevantes, crit. C": selrelevantes_c,
#                     "Set 6: los atributos más relevantes, crit. D": selrelevantes_c,
#                     "Set 7: atributos pares": selpares,
#                     "Set 8: todos los atributos": seltodos}

# set acotado, relativamente rápido y apropiado para df01
sets_de_atributos = {"Set 1: 3 atributos al azar en runtime": sel3random,
                     "Set 2: 3 atributos al azar preseleccionados": sel3random2,
                     "Set 3: los atributos más relevantes, crit. A": selrelevantes_a,
                     "Set 5: los atributos más relevantes, crit. C": selrelevantes_c}


#%%

# Definimos los datasets iniciales
X = imagenes(df_ej4)
Y = labels(df_ej4)

#%%

# Ahora particionamos el dataset en un 70% para development y un 30%
# para testing. Se reparte mediante muestreo estratificado para mantener
# las proporciones de ceros y unos en los subsets.
X_devel, X_test, Y_devel, Y_test = train_test_split(X, Y, test_size = 0.3, stratify=Y)

#%%
# kNN sobre df01, aplicado a los distintos sets de atributos definidos
# previamente.

# De lo que hay en development, separamos 70% para training y
# 30% para validación. Repartido mediante muestreo estratificado
# para mantener las proporciones de ceros y unos en los subsets.
X_train, X_valid, Y_train, Y_valid = train_test_split(X_devel, Y_devel, test_size = 0.3, stratify=Y_devel)

#%%

# Correremos y graficaremos kNN para k variando entre 1 y maxk:
maxk = 20

resultados_por_set_de_atributos = kNN_iterando_sobre_atributos_con_train_y_valid(X_train, Y_train, X_valid, Y_valid, maxk, sets_de_atributos)

#%%

# Finalmente, aplicamos los modelos obtenidos sobre el dataset de
# testing para comparar su exactitud con la obtenida durante
# la validación.

testear_y_graficar_exactitud(resultados_por_set_de_atributos, X_test, Y_test)




#%%
# Ejercicio 5: cross-validation (kNN con repetición) sobre df01
# variando k sobre los distintos sets de atributos.


# Definimos los conjuntos de atributos que nos interesa analizar.
sets_de_atrib = {"Set 2: 3 atributos al azar preseleccionados": sel3random2,
                "Set 3: los atributos más relevantes, crit. A": selrelevantes_a,
                "Set 5: los atributos más relevantes, crit. C": selrelevantes_c}

# Correremos y graficaremos kNN para k variando entre 1 y maxk:
maxk = 20

# Cantidad de iteraciones para cross-validation. En cada iteración dividiremos
# el set de development en distintos sets de training y de validación,
# ajustando modelos kNN para los distintos valores de k que nos interesan.
reps = 10

# Definimos los datasets iniciales
X = imagenes(df_ej4)
Y = labels(df_ej4)

# Nota: el objeto de resultado obtenido de esta función
# no contiene modelos.
res_por_set_de_atributos_notexe = kNN_repetido_con_atributos(X, Y, reps, maxk, sets_de_atrib)

#%%

# Generación de gráficos de promedios de exactitud alcanzados
# mediante kNN repetido para los sets de atributos elegidos.
for s in res_por_set_de_atributos_notexe:
    klist = res_por_set_de_atributos_notexe[s]["klist"]
    lista_avg_acc_train = res_por_set_de_atributos_notexe[s]["lista_acc_train"]
    lista_avg_acc_valid = res_por_set_de_atributos_notexe[s]["lista_acc_valid"]
    # Graficamos exactitud vs. k para los sets de development y testing
    # tras reps repeticiones de kNN.
    plotear_knn(klist, lista_avg_acc_train[:-1], "Promedio Devel "+s,
                lista_avg_acc_valid[:-1], "Promedio Testing  "+s,reps)
        # ^ (workaround a bug del tipo off-by-one en algún lado)
    print( "Resultados promedio para conjunto de atributos:\n",
        s,
        "\ntras",
        reps,
        "repeticiones.\n  Vecinos de kNN (k):\n",
        klist,
        "\nExactitud development:\n",
        lista_avg_acc_train,
        "\nMáximo k development:\n",
        lista_avg_acc_train[1:].index(max(lista_avg_acc_train[1:])),
        "\nExactitud testing:\n",
        lista_avg_acc_valid,
        "\nMáximo k testing:\n",
        lista_avg_acc_valid[1:].index(max(lista_avg_acc_valid[1:])))


#%%
# Ejercicios 6 y 7:
# Dado el dataset MNIST, entrenamos muchos modelos de árbol de decisión
# (de clasificación) y los validamos haciendo cross-validation con k-folding.
#

profundidad_max = 40

dicc_res = arboles_de_decision_por_prof(df,prof_max=profundidad_max)

# Definimos un diccionario para poder usar la función plotear_resultados
rotulos = {"titulo": "Exactitud promedio en función de la profundidad del árbol",
           "label x" : "P", "label y" : "Exactitud (Accuracy)", 
           "tamaño imagen": [20,10] }

# Graficamos exactitud promedio vs P (profundidad del árbol)
plotear_resultados(dicc_res, rotulos)

# La exactitud promedio crece conforme aumenta la profundidad del árbol,
# sin embargo este número a partir de P≈13 oscila en un intervalo muy
# pequeño, entonces parece que el gráfico tiene un comportamiento
# cuasi asintótico.
# P = 13 parece ser un valor de profundidad óptimo para estos árboles
# de decisión.



#%%

# Ejercicio 8: evaluación, utilizando datasets provistos específicamente para
# ello, de los modelos que consideramos los mejores dentro de los que 
# hemos explorado, tanto en kNN como en árbol de decisión.


# kNN classifier

X_test_dado = imagenes(dftestbin)
Y_test_dado = labels(dftestbin)

# Definimos los conjuntos de atributos que nos interesa analizar.
sets_de_atrib = {"Set 2: 3 atributos al azar preseleccionados": sel3random2,
                "Set 3: los atributos más relevantes, crit. A": selrelevantes_a,
                "Set 5: los atributos más relevantes, crit. C": selrelevantes_c}

# Los valores de k fueron elegidos a partir de los gráficos de 
# exactitud obtenidos con cross-validation del estilo kNN repetido
# (reiterando el procedimiento de partir el dataset original en devel
# y test, aplicando kNN a devel, prediciendo las etiquetas de test
# y calculando la exactitud lograda). Los valores de precisión
# obtenidos para cada k en cada repetición son promediados,
# obteniéndose valores promedio de exactitud en función de k.
# Esto se realiza por cada set de atributos que hemos seleccionado.
# Para cada set de atributos se elige un valor de k
# que coincida con un máximo local de exactitud tal que es k lo más
# pequeño posible. Para evitar overfitting se desestiman los 
# k en los que el máximo local ocurre en k==1.
k_elegido = {"Set 2: 3 atributos al azar preseleccionados": 4,
                "Set 3: los atributos más relevantes, crit. A": 17,
                "Set 5: los atributos más relevantes, crit. C": 2}

for s in sets_de_atrib:
    X_evaluar = imagenes(df01)[sets_de_atrib[s]]
    X_test_dado_con_atributos = X_test_dado[sets_de_atrib[s]]
    Y_evaluar = labels(df01)
    print("Set de atributos:", s)
    
    resultados_test_binario = validar_knn(X_evaluar, Y_evaluar, X_test_dado_con_atributos, Y_test_dado, k_elegido[s])
    metrica_test_bin = metricas_de_performance(labels(dftestbin), resultados_test_binario['Y_valid_pred'], es_multi_label=False)
    
    print(metrica_test_bin['confusion matrix'])

#%%

# Decision tree classifier
profundidad = 13
resultados_test_binario= validar_tree(df01, dftestbin, profundidad=profundidad)
resultados_test_digitos= validar_tree(df, dftest, profundidad=profundidad)

metrica_test_bin = metricas_de_performance(labels(dftestbin), resultados_test_binario['Y_valid_pred'], es_multi_label=False)
metrica_test_dig = metricas_de_performance(labels(dftest), resultados_test_digitos['Y_valid_pred'])

print(metrica_test_dig['confusion matrix'])
