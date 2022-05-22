# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:07:27 2022

@author: 110469
"""

import pandas as pd
import joblib
import os 
import math

path = os.getcwd()


# pre-processing

df = pd.read_excel(path+'\\'+'modelo_v13.xlsx',sheet_name='INPUT',
                   skiprows=1).drop(columns='Unnamed: 0').rename(columns={'INDICADORES':'Micromercado'})

df = df.T
df.columns = df.iloc[0]
df = df.iloc[1:2,:]

df_mckinsey =  pd.read_excel(path+'\\'+'modelo_v13.xlsx',sheet_name='Base Mckinsey')

df_final = pd.merge(df, df_mckinsey, on = 'Micromercado')

df_incremento =  pd.read_excel(path+'\\'+'modelo_v13.xlsx',sheet_name='Base_Incremento')

df_final = pd.merge(df_final, df_incremento, on = 'Micromercado')


rua_ou_shop = df_final['Rua ou Shopping?'].values.tolist()
metros_pmenos = df_final['Quantos metros da loja mais próxima?'].values[0]
tem_canibalização_pmenos = df_final['Canibalização Pmenos? (até 2km)'].values[0]
metros_concorrente = df_final['Quantos metros do concorrente a loja mais próxima?'].values[0]
tem_canibalização_concorrente = df_final['Canibalização Concorrente? (até 600m)'].values[0]


df_final_rua = df_final[['Competitive_Advantage','Sortimento_Jan',
                         'Ebitda 2022', 'Share_PDV_Pague_Menos', 
                         'Market_Share_Estimado', 
                         'Faixa_Desconto_RX_Faixa_01', 'Faixa_Desconto_RX_Faixa_02',
                         'Faixa_Desconto_RX_Faixa_03','Faixa_Desconto_RX_Faixa_04',
                         'Faixa_Desconto_RX_Faixa_05', 'Faixa_Desconto_RX_Faixa_06',
                         'Faixa_Desconto_RX_Faixa_07', 'Faixa_Desconto_Sem_Faixa_Definida',
                         'Faixa_Desconto_RX_Fora_Politica','Sortimento RX']]

df_final_shopping = df_final[['Sortimento RX',
                      'Política de Desconto Genérico',
                      'Política de Desconto de Não Genérico',
                      'Score']]

# peso score 
df_score = pd.read_excel(path+'\\'+'modelo_v13.xlsx',sheet_name='Base Score')
df_acres_score = pd.merge(df[['Estado','Região','Score']],df_score,on=['Estado','Região'])
df_acres_score['Diferença'] = df_acres_score['Score'] - df_acres_score['Média de Score']
fator_score = 26048.32

score_incremento = df_acres_score['Diferença'].values[0]*fator_score


# modelo
if rua_ou_shop[0] == 'Rua':
    
    X = df_final_rua.values
    sc = joblib.load('street_sc_v13.joblib')
    
    X = sc.transform(X)
    regressor = joblib.load('street_xgboost_v13.joblib')

    y_pred = regressor.predict(X)
    Resultado = pd.DataFrame({'Potencial de Venda':y_pred})
    Resultado = Resultado + score_incremento
    
else:
            
    X = df_final_shopping.values
    regressor = joblib.load('shopping_model_regessor.joblib')

    y_pred = regressor.predict(X)
    Resultado = pd.DataFrame({'Potencial de Venda':y_pred})

# criando a canibalização
if  metros_pmenos <=2000 and tem_canibalização_pmenos=='Sim' :
    canibalização_pmenos = 0.0422*math.log(metros_pmenos) - 0.3222
else:
    canibalização_pmenos = 0
    
if  metros_concorrente <=600 and tem_canibalização_concorrente=='Sim' :
    canibalização_concorrente = 0.0145*math.log(metros_concorrente) - 0.0922
else:
    canibalização_concorrente = 0
    
canibalização_total =  canibalização_pmenos + canibalização_concorrente


# definindo as bandas
Resultado['Potencial de Venda'] = Resultado['Potencial de Venda']*(1+canibalização_total)
Resultado['Venda Potencial Mínima'] = Resultado['Potencial de Venda']*0.95
Resultado['Venda Potencial Máxima'] = Resultado['Potencial de Venda']*1.25
Resultado = Resultado.drop(columns='Potencial de Venda')

Resultado.to_excel('Potencial_de_venda.xlsx',index=False)
