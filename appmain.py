from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn


#push demo
import pandas as pd

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning


warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def select_best_model(Results_All_Models):
    # Inicializar DataFrame para almacenar los mejores modelos
    best_models = pd.DataFrame(columns=['Registro', 'Mejor_Modelo', 'Puntuacion_Mejor', 'Segundo_Mejor_Modelo', 'Puntuacion_Segundo','REG'])

    # Agrupar por Cod_Item, Canal, Region
    grouped = Results_All_Models.groupby(['REG'])

    for name, group in grouped:
        group = group.copy()

        # Asignar puntos por Average MFA
        group['MFA_Score'] = group['Average MFA'].rank(method='min', ascending=True)
        group['MFA_Points'] = group['MFA_Score'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else 0))

        # Asignar puntos por Average BIAS
        group['BIAS_Score'] = group['Average BIAS'].abs().rank(method='min', ascending=True)
        group['BIAS_Points'] = group['BIAS_Score'].apply(lambda x: 3 if x == 1 else (2 if x == 2 else 0))

        # Asignar puntos por Average MAE
        group['MAE_Score'] = group['Average MAE'].rank(method='min', ascending=True)
        group['MAE_Points'] = group['MAE_Score'].apply(lambda x: 2 if x == 1 else (1 if x == 2 else 0))

        # Asignar puntos por Average RMSE
        group['RMSE_Score'] = group['Average RMSE'].rank(method='min', ascending=True)
        group['RMSE_Points'] = group['RMSE_Score'].apply(lambda x: 2 if x == 1 else (1 if x == 2 else 0))

        # Calcular puntuación total
        group['Total_Score'] = group[['MFA_Points', 'BIAS_Points', 'MAE_Points', 'RMSE_Points']].sum(axis=1)

        # Encontrar el mejor y segundo mejor modelo
        #best_models_group = group.sort_values(by='Total_Score', ascending=False).head(2)
        best_models_group = group.sort_values(
            by=['Total_Score', 'Average MAE', 'Average MFA'],
            ascending=[False, True, True]
        ).head(2)
        best_model = best_models_group.iloc[0]
        second_best_model = best_models_group.iloc[1] if len(best_models_group) > 1 else best_model
        print(best_models_group)
        # Añadir al DataFrame de resultados
        new_row = pd.DataFrame({
            'Mejor_Modelo': [best_model['Modelo']],
            'Puntuacion_Mejor': [best_model['Total_Score']],
            'Average MAE Mejor': [best_model['Average MAE']],
            'Average MFA Mejor': [best_model['Average MFA']],
            'Average RMSE Mejor': [best_model['Average RMSE']],
            'Average BIAS Mejor': [best_model['Average BIAS']],
            'Segundo_Mejor_Modelo': [second_best_model['Modelo']],
            'Puntuacion_Segundo': [second_best_model['Total_Score']],
            'Average MAE Segundo': [second_best_model['Average MAE']],
            'Average MFA Segundo': [second_best_model['Average MFA']],
            'Average RMSE Segundo': [second_best_model['Average RMSE']],
            'Average BIAS Segundo': [second_best_model['Average BIAS']],
            'REG': [name[0]]
        })
        best_models = pd.concat([best_models, new_row], ignore_index=True)
        
    return best_models

def Forecast (best_models_report, opt_params_df, df_data, TS, DF, SP, FP):
    # %%
    # DataFrame para almacenar los resultados detallados de las predicciones futuras
    future_detailed_results = pd.DataFrame()
    print(best_models_report)
    # se define un diccionario donde se relacionan los nombres de los modelos con las funciones de los modelos.
    model_name_to_function = {
        'NAIVE' : NAIVE,
        'Simple_Exp_Opt' : Simple_Exp_Opt,
        'Holt_Double_Opt': Holt_Double_Opt,
        'HoltWinters_Opt': HoltWinters_Opt,
        #'ARIMA' : ARIMA,
        #'SARIMA' : SARIMA,
        'AUTO_THETA_Opt' : AUTO_THETA_Opt, 
        'AUTO_ETS_Opt' : AUTO_ETS_Opt,
        'AUTO_CES_Opt': AUTO_CES_Opt}
    df_data.reset_index(drop=True, inplace=True)
    models_to_run = [best_models_report.iloc[0]['Mejor_Modelo']]
    #print(models_to_run)
    model_func = model_name_to_function.get(models_to_run[0])
    print(f"Procesando modelo: {model_func.__name__}")
    df_data = df_data[df_data['Modelo'] == model_func.__name__]    
    V = df_data[['V']].rename(columns={'V': 'y'})
    predictions, _, _ = model_func(V, TS, DF, SP, FP,mode='forecast')
    print('armando table')
    if predictions is not None:
        # Aquí, se incluyen todos los valores de 'Unidades', tanto de entrenamiento como de testeo
        
        # Calcular la cantidad de periodos a predecir
        num_periods_to_predict = min(TS, len(predictions))

        last_date = pd.to_datetime(df_data['Fecha']).max()
        print('post fehca')
        # Generar fechas futuras en frecuencia mensual
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                             periods=num_periods_to_predict, freq=DF)
        
        # Convertir opt_params a una representación adecuada para incluir en el DataFrame
        #opt_params_str = str(opt_params) if opt_params else "Default"
        
        future_df = pd.DataFrame({
        'Fecha': future_dates,
        #'Cod_Item': series['Cod_Item'],
        #'Canal': series['Canal'],
        #'Region': series['Region'],
        #'Modelo': model_name,
        'F': predictions.values[:num_periods_to_predict]
        #'Opt_parameters': [opt_params_str] * len(future_dates)
        })
        print(future_df)
        # Concatenar con los resultados existentes
        #future_detailed_results = pd.concat([future_detailed_results, future_df], ignore_index=True)

    return future_df

def ForecastErrors(V, F):
    # Verificación de que ambos inputs sean Series de pandas y tengan el mismo índice
    if not (isinstance(V, pd.Series) and isinstance(F, pd.Series)):
        raise ValueError("V and F must be pandas Series.")  # V y F deben ser Series de pandas
    if not V.index.equals(F.index):
        raise ValueError("V and F must have the same index.")  # V y F deben tener el mismo índice

    # Inicialización de listas para almacenar los valores calculados para cada período
    mfas, biases, maes, rmses = [], [], [], []

    # Cálculo de MFA, BIAS, MAE y RMSE para cada período
    for actual, forecast in zip(V, F):
        # Manejo de casos especiales donde el valor actual o la predicción es cero
        if actual == 0 and forecast == 0:
            mfa = bias = 0  # Si ambos son cero, no hay error
        elif actual == 0 and forecast != 0:
            mfa = 1  # Error máximo si se predice algo cuando no debería haber nada
            bias = -1  # Bias máximo negativo
        else:
            # Cálculos normales de MFA y BIAS
            mfa = np.abs((actual - forecast) / np.maximum(np.abs(actual), np.abs(forecast)))
            bias = (actual - forecast) / actual
        
        # Cálculo del error absoluto medio (MAE) y raíz del error cuadrático medio (RMSE)
        mae = np.abs(actual - forecast)
        rmse = np.sqrt(np.mean((actual - forecast) ** 2)/actual)
        
        # Almacenamiento de los cálculos en sus respectivas listas
        mfas.append(mfa)
        biases.append(bias)
        maes.append(mae)
        rmses.append(rmse)

    # Cálculo de los valores medios para cada métrica de error
    average_mfa = np.mean(mfas)
    average_bias = np.mean(biases)
    average_mae = np.mean(maes)
    average_rmse = np.mean(rmses)

    # Creación de un DataFrame para almacenar los resultados por período
    results_df = pd.DataFrame({'MFA': mfas, 
                               'BIAS': biases, 
                               'MAE': maes,
                               'RMSE': rmses })

    # Diccionario para los valores promedio de los errores
    average_results = {'Average MFA': average_mfa, 
                       'Average BIAS': average_bias, 
                       'Average MAE': average_mae,
                       'Average RMSE': average_rmse}

    return results_df, average_results

def Holt_Double_Opt(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División de datos en conjuntos de entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    # Evaluación y aplicación de parámetros óptimos si están disponibles
    if opt_parameters is not None and 'alpha' in opt_parameters and 'beta' in opt_parameters:
        # Uso de parámetros óptimos proporcionados para la configuración inicial del modelo
        best_alpha = opt_parameters['alpha']
        best_beta = opt_parameters['beta']
    else:
        # Búsqueda de grilla con validación cruzada para encontrar los mejores parámetros alpha y beta
        alpha_values = np.linspace(0.15, 0.85, 10)
        beta_values = np.linspace(0.15, 0.85, 10)
        tscv = TimeSeriesSplit(n_splits=5)
        best_alpha, best_beta = None, None
        best_score = float('inf')

        for alpha in alpha_values:
            for beta in beta_values:
                scores = []
                for train_index, test_index in tscv.split(TrainingSet):
                    train, test = TrainingSet.iloc[train_index], TrainingSet.iloc[test_index]
                    model = Holt(train['y'], initialization_method='estimated').fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
                    predictions = model.forecast(steps=len(test))
                    score = mean_squared_error(test['y'], predictions)
                    scores.append(score)

                average_score = np.mean(scores)
                # Actualización de los mejores parámetros si se encuentra un mejor puntaje promedio
                if average_score < best_score:
                    best_score = average_score
                    best_alpha, best_beta = alpha, beta

    # Construcción del modelo final con los mejores parámetros encontrados
    model = Holt(TrainingSet['y'], initialization_method='estimated').fit(smoothing_level=best_alpha, smoothing_trend=best_beta, optimized=False)

    # Generación de predicciones para el conjunto de prueba
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('Holt')
    Predictions.index = TestSet.index
    
    # Ajuste de predicciones negativas a cero, si es necesario
    Predictions[Predictions < 0] = 0

    # Cálculo y evaluación de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)
    
    # Devolución de los parámetros óptimos encontrados
    opt_parameters = {'alpha': best_alpha, 'beta': best_beta}

    return Predictions, average_results, opt_parameters

def HoltWinters_Opt(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División de datos en conjuntos de entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    TrainingSet['y'] = TrainingSet['y'].clip(lower=0.01)  # Reemplazar valores <= 0 con un valor pequeño

    # Evaluación y aplicación de parámetros óptimos si están disponibles
    if opt_parameters is not None and 'alpha' in opt_parameters and 'beta' in opt_parameters and 'gamma' in opt_parameters:
        # Utilización de parámetros óptimos proporcionados para la configuración inicial del modelo
        best_alpha = opt_parameters['alpha']
        best_beta = opt_parameters['beta']
        best_gamma = opt_parameters['gamma']
    else:
        # Búsqueda de grilla para encontrar los mejores parámetros alpha, beta y gamma
        alpha_values = np.linspace(0.15, 0.85, 10)
        beta_values = np.linspace(0.15, 0.85, 10)
        gamma_values = np.linspace(0.15, 0.85, 10)
        best_alpha, best_beta, best_gamma = None, None, None
        best_score = float('inf')

        for alpha in alpha_values:
            for beta in beta_values:
                for gamma in gamma_values:
                    model = ExponentialSmoothing(TrainingSet['y'], trend='add', seasonal='add', seasonal_periods=SP, 
                                                 initialization_method='heuristic').fit(smoothing_level=alpha, smoothing_trend=beta, 
                                                                                        smoothing_seasonal=gamma, optimized=False)
                    predictions = model.forecast(steps=len(TestSet))
                    score = mean_squared_error(TestSet['y'], predictions)
                    # Actualización de los mejores parámetros si se encuentra un mejor puntaje
                    if score < best_score:
                        best_score = score
                        best_alpha, best_beta, best_gamma = alpha, beta, gamma

    # Construcción del modelo final con los mejores parámetros encontrados
    model = ExponentialSmoothing(TrainingSet['y'], trend='add', seasonal='add', seasonal_periods=SP, 
                                 initialization_method='heuristic').fit(smoothing_level=best_alpha, smoothing_trend=best_beta, 
                                                                        smoothing_seasonal=best_gamma, optimized=False)

    # Generación de predicciones para el conjunto de prueba
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('HW')
    Predictions.index = TestSet.index
    
    # Ajuste de predicciones negativas a cero, si es necesario
    Predictions[Predictions < 0] = 0
    
    # Cálculo y evaluación de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Devolución de los parámetros óptimos, si no fueron inicialmente proporcionados
    opt_parameters = {'alpha': best_alpha, 'beta': best_beta, 'gamma': best_gamma} if not opt_parameters else opt_parameters

    return Predictions, average_results, opt_parameters


#-----#
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_squared_error
import pandas as pd

def AUTO_THETA_Opt(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División del conjunto de datos en entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    # Comprobación y aplicación de parámetros óptimos si existen
    if opt_parameters is not None and 'method' in opt_parameters:
        best_method = opt_parameters['method']
    else:
        # Inicialización de valores predeterminados para la optimización
        best_method = 'add'
        best_score = float('inf')

        # Bucle para encontrar el mejor método de desestacionalización
        for method in ['add', 'mul']:
            try:
                model = ThetaModel(TrainingSet['y'], deseasonalize=True, period=SP, method=method)
                model_fit = model.fit()
                predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoTheta')
                score = mean_squared_error(TestSet['y'], predictions)
                # Actualización del mejor método si se encuentra un mejor puntaje
                if score < best_score:
                    best_score = score
                    best_method = method
            except Exception as e:
                print(f"Error al ajustar el modelo Theta con método {method}: {e}")

        # Almacenar el mejor método encontrado
        opt_parameters = {'method': best_method}

    # Construcción del modelo final con el mejor método encontrado
    model = ThetaModel(TrainingSet['y'], deseasonalize=True, period=SP, method=best_method)
    model_fit = model.fit()

    # Generación de predicciones con el modelo final
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoTheta')
    Predictions.index = TestSet.index
    # Ajuste de predicciones negativas a cero
    Predictions[Predictions < 0] = 0

    # Cálculo de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Devolución de los resultados del pronóstico, los errores promedio y los parámetros óptimos
    return Predictions, average_results, opt_parameters


#-----#
#-----#
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import mean_squared_error
# SP no se va a usar en la funcion, y se reemplaza en la funcion con 12. Esto por que no fui capaz de que SP entrara como 12 
# En la funcion

def AUTO_ETS_Opt(V, TS, SP, DF, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División del conjunto de datos en entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")


    # Optimización de parámetros del modelo ETS
    if isinstance(opt_parameters, dict) and 'error' in opt_parameters and 'trend' in opt_parameters and 'seasonal' in opt_parameters:
        best_error = opt_parameters['error']
        best_trend = opt_parameters['trend']
        best_seasonal = opt_parameters['seasonal']
    else:
        # Inicializar valores predeterminados si no hay parámetros óptimos proporcionados
        best_error, best_trend, best_seasonal = 'add', 'add', 'add'
        best_score = float('inf')

        # Bucle para encontrar la mejor configuración de parámetros
        for error in ['add', 'mul']:
            for trend in ['add', 'mul', None]:
                for seasonal in ['add', 'mul', None]:
                    try:
                        model = ETSModel(TrainingSet['y'], error=error, trend=trend, seasonal=seasonal, seasonal_periods=12, 
                                         initialization_method='heuristic').fit()
                        predictions = model.forecast(steps=len(TestSet))
                        score = mean_squared_error(TestSet['y'], predictions)
                        # Actualización de los mejores parámetros si se encuentra un mejor puntaje
                        if score < best_score:
                            best_score = score
                            best_error, best_trend, best_seasonal = error, trend, seasonal
                    except Exception as e:
                        print(f"Error al ajustar el modelo ETS: {e}")

        # Almacenar los mejores parámetros encontrados
        opt_parameters = {'error': best_error, 'trend': best_trend, 'seasonal': best_seasonal}

    # Construcción del modelo ETS con los mejores parámetros encontrados
    model = ETSModel(TrainingSet['y'], error=best_error, trend=best_trend, seasonal=best_seasonal, seasonal_periods=12, 
                     initialization_method='heuristic').fit()
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('ETS')
    Predictions.index = TestSet.index
    # Ajuste de las predicciones negativas a cero
    Predictions[Predictions < 0] = 0

    # Cálculo de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Devolución de los resultados del pronóstico y los parámetros óptimos
    return Predictions, average_results, opt_parameters


#-----#
#-----#

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pandas as pd

def AUTO_CES_Opt(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División del conjunto de datos en entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    TrainingSet['y'] = TrainingSet['y'].clip(lower=0.01)  # Reemplazar valores <= 0 con un valor pequeño

    # Comprobación y aplicación de parámetros óptimos si existen
    if opt_parameters is not None and 'trend' in opt_parameters and 'seasonal' in opt_parameters:
        best_trend = opt_parameters['trend']
        best_seasonal = opt_parameters['seasonal']
    else:
        # Inicialización de valores predeterminados para la optimización
        best_trend, best_seasonal = 'add', 'add'
        best_score = float('inf')

        # Bucle para encontrar la mejor configuración de parámetros de tendencia y estacionalidad
        for trend in ['add', 'mul', None]:
            for seasonal in ['add', 'mul', None]:
                try:
                    model = ExponentialSmoothing(TrainingSet['y'], trend=trend, seasonal=seasonal, seasonal_periods=SP, use_boxcox=True)
                    # Optimización del modelo con basinhopping
                    model_fit = model.fit(optimized=True, remove_bias=True, method='basinhopping')
                    predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoCES')
                    score = mean_squared_error(TestSet['y'], predictions)
                    # Actualización de los mejores parámetros si se encuentra un mejor puntaje
                    if score < best_score:
                        best_score = score
                        best_trend, best_seasonal = trend, seasonal
                except Exception as e:
                    print(f"Error al ajustar el modelo CES con trend={trend}, seasonal={seasonal}: {e}")

        # Almacenar los mejores parámetros encontrados
        opt_parameters = {'trend': best_trend, 'seasonal': best_seasonal}
    TrainingSet['y'] = TrainingSet['y'].clip(lower=0.01)  # Reemplazar valores <= 0 con un valor pequeño
    # Construcción del modelo final con los mejores parámetros
    model = ExponentialSmoothing(TrainingSet['y'], trend=best_trend, seasonal=best_seasonal, seasonal_periods=SP, use_boxcox=True)
    model_fit = model.fit(optimized=True, remove_bias=True, method='basinhopping')

    # Generación de predicciones con el modelo final
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoCES')
    Predictions.index = TestSet.index
    # Ajuste de predicciones negativas a cero
    Predictions[Predictions < 0] = 0

    # Cálculo de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Devolución de los resultados del pronóstico, los errores promedio y los parámetros óptimos
    return Predictions, average_results, opt_parameters




import pandas as pd


import time

def Simple_Exp_Opt(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División de datos en conjuntos de entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")


    # Evaluación y aplicación de parámetros óptimos si están disponibles
    if opt_parameters is not None and 'alpha' in opt_parameters:
        # Uso de alpha óptimo proporcionado para la configuración inicial del modelo
        best_alpha = opt_parameters['alpha']
    else:
        # Búsqueda de grilla con validación cruzada para encontrar el mejor alpha
        alpha_values = np.linspace(0.15, 0.85, 20)  # Corregir error en 0,85 a 0.85
        tscv = TimeSeriesSplit(n_splits=5)
        best_alpha = None
        best_score = float('inf')

        for alpha in alpha_values:
            scores = []
            for train_index, test_index in tscv.split(TrainingSet):
                train, test = TrainingSet.iloc[train_index], TrainingSet.iloc[test_index]
                model = SimpleExpSmoothing(train['y'], initialization_method='estimated').fit(smoothing_level=alpha, optimized=False)
                predictions = model.forecast(steps=len(test))
                score = mean_squared_error(test['y'], predictions)
                scores.append(score)

            average_score = np.mean(scores)
            # Actualización de alpha óptimo si se encuentra un mejor puntaje promedio
            if average_score < best_score:
                best_score = average_score
                best_alpha = alpha

    # Construcción del modelo final con el mejor alpha encontrado
    model = SimpleExpSmoothing(endog=TrainingSet['y'], initialization_method='estimated').fit(smoothing_level=best_alpha, optimized=False)

    # Generación de predicciones para el conjunto de prueba
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('Holt')
    Predictions.index = TestSet.index
    
    # Ajuste de predicciones negativas a cero, si es necesario
    Predictions[Predictions < 0] = 0

    # Cálculo y evaluación de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)
    
    # Devolución de los parámetros óptimos encontrados
    opt_parameters = {'alpha': best_alpha}

    return Predictions, average_results, opt_parameters




import time
import traceback

def log_message(message):
    """Append log messages to a file."""
    with open("prophet_log.txt", "a") as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")





from dateutil.relativedelta import relativedelta
#from pmdarima.arima import auto_arima
def SARIMA(V,TS,DF,SP, FP, opt_parameters = None, mode='entrenamiento'):
    
        # Validación de parámetros
    if TS <= 0 or TS >= len(V):
        raise ValueError("El tamaño del conjunto de prueba TS debe ser mayor que 0 y menor que la longitud de V.")

    if SP <= 0:
        raise ValueError("La periodicidad estacional SP debe ser mayor que 0.")
    
    # Entrenamieto y Seteo
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    
    # Modelo de Forecast (En caso de que se este tomando mucho tiempo, se puede llevar stepwise=True, para una busqueda
    # mas eficiente de la configuracion del modelo AR=# , I=#, MA=#)
    model=auto_arima(y=TrainingSet['y'],m=SP,seasonal=True,stepwise=True,maxlag= 24, error_action='warn', missing='mean')
    
    # Predicciones
    Predictions=pd.Series(model.predict(n_periods=TS)).rename('Sarima')
    Predictions.index=TestSet.index
    
    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0
    
    #Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'],Predictions)
    return Predictions, average_results, None

from statsmodels.tsa.api import SimpleExpSmoothing

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
def Simple_Exp(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # División de datos en conjuntos de entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")


    # Evaluación y aplicación de parámetros óptimos si están disponibles
    if opt_parameters is not None and 'alpha' in opt_parameters:
        # Uso de alpha óptimo proporcionado para la configuración inicial del modelo
        best_alpha = opt_parameters['alpha']
    else:
        # Búsqueda de grilla con validación cruzada para encontrar el mejor alpha
        alpha_values = np.linspace(0.15, 0.85, 20)  # Corregir error en 0,85 a 0.85
        tscv = TimeSeriesSplit(n_splits=5)
        best_alpha = None
        best_score = float('inf')

        for alpha in alpha_values:
            scores = []
            for train_index, test_index in tscv.split(TrainingSet):
                train, test = TrainingSet.iloc[train_index], TrainingSet.iloc[test_index]
                model = SimpleExpSmoothing(train['y'], initialization_method='estimated').fit(smoothing_level=alpha, optimized=False)
                predictions = model.forecast(steps=len(test))
                score = mean_squared_error(test['y'], predictions)
                scores.append(score)

            average_score = np.mean(scores)
            # Actualización de alpha óptimo si se encuentra un mejor puntaje promedio
            if average_score < best_score:
                best_score = average_score
                best_alpha = alpha

    # Construcción del modelo final con el mejor alpha encontrado
    model = SimpleExpSmoothing(endog=TrainingSet['y'], initialization_method='estimated').fit(smoothing_level=best_alpha, optimized=False)

    # Generación de predicciones para el conjunto de prueba
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('Holt')
    Predictions.index = TestSet.index
    
    # Ajuste de predicciones negativas a cero, si es necesario
    Predictions[Predictions < 0] = 0

    # Cálculo y evaluación de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)
    
    # Devolución de los parámetros óptimos encontrados
    opt_parameters = {'alpha': best_alpha}

    return Predictions, average_results, opt_parameters


def NAIVE(V, TS, DF, SP, FP=None, opt_parameters=None, mode='entrenamiento'):
    # Verificación y configuración del índice como fecha en el DataFrame
    if not pd.api.types.is_datetime64_any_dtype(V.index):
        V['Fecha'] = pd.to_datetime(V['Fecha'])  # Convertir la columna 'Fecha' a datetime
        V.set_index('Fecha', inplace=True)  # Establecer la columna 'Fecha' como índice
    V.sort_index(inplace=True)  # Ordenar el DataFrame por índice de fecha
    if V.index.name == 'Fecha':
        V = V.reset_index()
    # División de datos en conjuntos de entrenamiento y prueba
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")
    # Inicialización de lista para almacenar las predicciones
    Predictions = []

    # Generación de predicciones usando valores pasados ponderados
    for i in range(TS):
        # Verificación de suficientes datos para predicciones basadas en meses anteriores
        if len(TrainingSet) >= 3:
            # Uso de los últimos tres meses con pesos específicos
            previous_month_value = TrainingSet['y'].iloc[-1] * 0.3  # Peso para el mes anterior
            two_months_ago_value = TrainingSet['y'].iloc[-2] * 0.2  # Peso para hace dos meses
            three_months_ago_value = TrainingSet['y'].iloc[-3] * 0.1  # Peso para hace tres meses
        else:
            # Establecer a cero si no hay suficientes datos
            previous_month_value = two_months_ago_value = three_months_ago_value = 0

        # Cálculo de la contribución del mismo mes del año anterior
        same_month_last_year = TrainingSet.index[-1] - pd.DateOffset(months=12) + relativedelta(months=i)
        same_month_last_year_value = V['y'].get(same_month_last_year, 0) * 0.4  # Peso para el mismo mes del año pasado

        # Suma ponderada de las contribuciones para la predicción del mes actual
        prediction = previous_month_value + two_months_ago_value + three_months_ago_value + same_month_last_year_value
        Predictions.append(prediction)

    # Conversión de la lista de predicciones a una Serie de pandas y ajuste del índice
    Predictions = pd.Series(Predictions, index=TestSet.index).rename('NAIVE')

    # Ajuste de predicciones negativas a cero
    Predictions[Predictions < 0] = 0

    # Cálculo y evaluación de errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    return Predictions, average_results, None




from statsmodels.tsa.api import Holt
def Holt_Double(V, TS, DF, SP, FP, opt_parameters=None, mode='entrenamiento'):

    # Entrenamieto y Seteo
    
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    # Modelo de Forecast (Ante problemas de convergencia a una solucion optima, 
    # se puede cambiar initialization_method='heuristic')
    model = Holt(endog = TrainingSet['y'],initialization_method='estimated').fit()

    # Predicciones
    Predictions = pd.Series(model.forecast(steps=len(TestSet))).rename('Holt')
    Predictions.index = TestSet.index
    
    # Ajustar pronósticos negativos a cero
    Predictions[Predictions < 0] = 0
    
    # Errores de pronóstico    
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    return Predictions, average_results , None

from statsmodels.tsa.holtwinters import ExponentialSmoothing

def HoltWinters(V, TS, DF, SP, FP, opt_parameters = None, mode='entrenamiento'):
    
    # Entrenamiento y Seteo
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")
    TrainingSet['y'] = TrainingSet['y'].clip(lower=0.01)  # Reemplazar valores <= 0 con un valor pequeño

    # Modelo de Forecast (Ante problemas de convergencia a una solucion optima, 
    # se puede cambiar initialization_method='heuristic')
    model = ExponentialSmoothing(endog=TrainingSet['y'], trend='add', seasonal='add', seasonal_periods=SP, 
                                    initialization_method='estimated').fit()
    # Predicciones
    Predictions = model.forecast(steps=len(TestSet)).rename('HW')
    Predictions.index = TestSet.index
    
    # Ajustar pronósticos negativos a cero
    Predictions[Predictions < 0] = 0
    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)
    return Predictions, None, None


def ARIMA(V,TS,DF,SP, FP, opt_parameters = None, mode='entrenamiento'):
    
     # Validación de parámetros
    if TS <= 0 or TS >= len(V):
        raise ValueError("El tamaño del conjunto de prueba TS debe ser mayor que 0 y menor que la longitud de V.")
    
    # Entrenamieto y Seteo
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    # Modelo de Forecast (En caso de que se este tomando mucho tiempo, se puede llevar stepwise=True, para una busqueda
    # mas eficiente de la configuracion del modelo AR=# , I=#, MA=#)
    model=auto_arima(y=TrainingSet['y'], seasonal=False,stepwise=True,maxlag= 24, error_action='warn', missing='mean')
    
    # Predicciones
    Predictions=pd.Series(model.predict(n_periods=TS)).rename('Arima')
    Predictions.index=TestSet.index
    
    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0
    
    #Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'],Predictions)
    return Predictions, average_results, None

from statsmodels.tsa.forecasting.theta import ThetaModel
def AUTO_THETA(V, TS, DF, SP, FP, opt_parameters=None, mode='entrenamiento'):
    # Entrenamiento y Seteo
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    # Modelo de Forecast
    model = ThetaModel(TrainingSet['y'], deseasonalize=True, period=SP, method='auto')
    model_fit = model.fit()

    # Predicciones
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoTheta')
    Predictions.index = TestSet.index

    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0
    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Retorno de información
    return Predictions, average_results, None

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
def AUTO_ETS(V, TS, DF, SP, FP, opt_parameters=None, mode='entrenamiento'):
    # Entrenamiento y Seteo
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")

    # Modelo de Forecast
    model = ETSModel(TrainingSet['y'], error='add', trend='add', seasonal='add', damped_trend=True, seasonal_periods=SP)
    model_fit = model.fit()

    # Predicciones
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoETS')
    Predictions.index = TestSet.index

    # Ajustar pronósticos negativos a cero, si es necesario
    Predictions[Predictions < 0] = 0

    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Retorno de información
    return Predictions, average_results, None


from statsmodels.tsa.holtwinters import ExponentialSmoothing
def AUTO_CES(V, TS, DF, SP, FP, opt_parameters = None, mode='entrenamiento'):
    # Entrenamiento y Seteo
    if mode == 'entrenamiento':
        # División de datos en conjuntos de entrenamiento y prueba
        TrainingSet = V.iloc[:-TS, :]  # Datos de entrenamiento: todo menos los últimos TS registros
        TestSet = V.iloc[-TS:, :]  # Datos de prueba: los últimos TS registros
    elif mode == 'forecast':
        # En modo forecast, se usa todo el conjunto de datos para la predicción
        TrainingSet = V
        TestSet = V.iloc[-TS:, :]  # Se mantiene TestSet para calcular las métricas de error
    else:
        raise ValueError("El modo debe ser 'entrenamiento' o 'forecast'")
    TrainingSet['y'] = TrainingSet['y'].clip(lower=0.01)  # Reemplazar valores <= 0 con un valor pequeño

    # Modelo de Forecast
    # Nota: No existe una implementación directa de AutoCES, así que usaremos ExponentialSmoothing con ajuste automático
    model = ExponentialSmoothing(TrainingSet['y'], trend='add', seasonal='add', seasonal_periods=SP)
    model_fit = model.fit(optimized=True)

    # Predicciones
    Predictions = pd.Series(model_fit.forecast(steps=len(TestSet))).rename('AutoCES')
    Predictions.index = TestSet.index

    # Errores de pronóstico
    results_df, average_results = ForecastErrors(TestSet['y'], Predictions)

    # Retorno de información
    return Predictions, average_results, None


#@app.get("/")
#async def read_test():
#    return {"message": "Deployed succesfully!"}


from typing import List

# Define Pydantic models for validation
class SeriesItem(BaseModel):
    Fecha: str  # Keeping it as str to convert later
    value: float

class Parameters(BaseModel):
    ts: int
    df: str
    sp: int
    fp: int

class InputData(BaseModel):
    parameters: Parameters
    series: List[SeriesItem]

from fastapi import BackgroundTasks
from fastapi.concurrency import run_in_threadpool
import time


#@app.post("/forecastjd/")
async def process_forecastJD(request: InputData):
    # Extract the series data from the request
    parameters = request.parameters
    series = request.series
    filtered_series = pd.DataFrame(series)
    filtered_series['value']=filtered_series['value'].fillna(0)
    #filtered_series = pd.DataFrame([s.dict() for s in request.series])
    filtered_series["Fecha"] = pd.to_datetime(filtered_series["Fecha"])  # Convert Fecha to datetime
    #filtered_series = filtered_series.asfreq('MS')
    # Apply models (assuming the models are accessible here)        print(filtered_series)
    print(filtered_series)

    print(filtered_series)
    Modelos_Univariables = [
        NAIVE,
        Simple_Exp_Opt,
        Holt_Double_Opt,
        HoltWinters_Opt,
        #ARIMA,
        #SARIMA,
        AUTO_THETA_Opt,
        AUTO_ETS_Opt,
        AUTO_CES_Opt
                            ]

    detailed_results = pd.DataFrame(columns=['Fecha', 'Modelo', 'V', 'F'])
    results_df = pd.DataFrame(columns=['Modelo', 'Average MFA', 'Average BIAS', 'Average MAE', 'Average RMSE'])
    opt_params_df = pd.DataFrame(columns=['Modelo', 'Opt_parameters'])
    ts = request.parameters.ts
    df = request.parameters.df
    sp = request.parameters.sp
    fp = request.parameters.fp

    filtered_series['REG']='pdp'
    try:
        filtered_series = filtered_series.set_index('Fecha')
        filtered_series = filtered_series.groupby(filtered_series.index).sum()
        # Process each model in a sequential manner, waiting for each to finish
        for model_func in Modelos_Univariables:
            try:
                #print(filtered_series)
                Variable_objetivo = 'value'
                v = filtered_series[[Variable_objetivo]].rename(columns={Variable_objetivo: 'y'})
                predictions, average_results, opt_parameters = model_func(v, ts, df, sp, fp,mode='entrenamiento')

                #log_message(model_func.__name__)

                new_row = pd.DataFrame({
                    'Modelo': [model_func.__name__],
                    'REG': [filtered_series['REG'].unique()[0]],
                    'Average MFA': [average_results['Average MFA']],
                    'Average BIAS': [average_results['Average BIAS']],
                    'Average MAE': [average_results['Average MAE']],
                    'Average RMSE': [average_results['Average RMSE']]
                })
                if not new_row.empty:  # Ensure new_row has data before concatenation
                    results_df = pd.concat([results_df, new_row], ignore_index=True)
                new_opt_params_row = pd.DataFrame({
                    'Modelo': [model_func.__name__],
                    'Opt_parameters': [opt_parameters]
                })
                #log_message(new_row)
                if not new_opt_params_row.empty and not new_opt_params_row.isna().all().all():
                    opt_params_df = pd.concat([opt_params_df, new_opt_params_row], ignore_index=True)
                detailed_df = pd.DataFrame({
                    'Fecha': filtered_series.index,
                    'Modelo': [model_func.__name__] * len(filtered_series),
                    'V': filtered_series[Variable_objetivo],
                    'F': None
                })
                
                detailed_df.loc[detailed_df.index[-len(predictions):], 'F'] = predictions.values
                #log_message(detailed_df)
                if not detailed_df.empty and not detailed_df.isna().all().all():
                    detailed_results = pd.concat([detailed_results, detailed_df], ignore_index=True)
                #log_message(detailed_results)
                
            except Exception as e:
                # Log the error and continue with the next model
                print(f"Error processing model {model_func.__name__}: {str(e)}")

        best_models_report = select_best_model(results_df)
        #print(results_df)
        #print(best_models_report)
        #print(detailed_results)
        #detailed_results.to_csv('resultadosfinales.csv')
        #results_df.to_csv('results_df.csv')
        future_fcst = Forecast (best_models_report, opt_params_df, detailed_results, ts, df, sp, fp)
        #plot_forecast_report(detailed_results)
        future_fcst['F'] = future_fcst['F'].clip(lower=0)
        future_fcst['F']=future_fcst['F'].fillna(0)
        print(filtered_series)
        return {
            'series': future_fcst.to_dict(orient="records")
            #,'detailed_results': detailed_results.to_dict()
            #,'results_df': results_df.to_dict()
        }

    except Exception as e:
        # Print error if any
        print(f"Error: {str(e)}")
        return {"error": str(e)}


# %%
#if __name__ == '__main__':
#    uvicorn.run(app, port=8000)