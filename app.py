import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

data_cities = pd.read_csv("data/cities.csv")
data_providers = pd.read_csv('data/providers.csv')
data_stations = pd.read_csv("data/stations.csv")
data_ticket = pd.read_csv('data/ticket_data.csv')

# Création d'une fonction pour calculer le temps de trajet

def time(departure, arrival):
  t = arrival - departure
  return int(t.total_seconds()/3600)

# Création d'un dataframe avec le prix, les trajets, et la durée
df_1 = data_ticket[['departure_ts', 'arrival_ts', 'price_in_cents', 'o_city', 'd_city']]

# Changement des IDs des villes par les noms  
df_1 = pd.merge(df_1, data_cities[['id','unique_name']], left_on="o_city", right_on='id')
df_1 = df_1.drop(["o_city","id"], axis=1)
df_1 = df_1.rename(columns={"unique_name": "o_city"})
df_1 = pd.merge(df_1, data_cities[['id','unique_name']], left_on="d_city", right_on='id')
df_1 = df_1.drop(["d_city","id"], axis=1)
df_1 = df_1.rename(columns={"unique_name": "d_city"})

# Calul du temps de trajet en heur
# Convertion des colonnes departure_ts et arrival_ts en format date
df_1['departure_ts'] = pd.to_datetime(df_1['departure_ts'])
df_1['arrival_ts'] = pd.to_datetime(df_1['arrival_ts'])


  
df_1['time'] = df_1.apply(lambda x: time(x['departure_ts'], x['arrival_ts']), axis=1)

df_1 = df_1.drop(['departure_ts','arrival_ts'], axis=1)

# Création d'un pivot table avec en index la ville de départ et celle d'arrivé 
df_pt_1 = pd.pivot_table(df_1, index=['o_city','d_city'],values=['price_in_cents','time'],
aggfunc=[np.mean, np.max, np.min, len])

### STREAMLIT 

st.title("Test Tictactrip")

# récupération de la liste des villes d'origines
li_o_city = np.unique(df_1['o_city'])

op_o_city = st.selectbox('Sélectionnez une ville de départ ',
li_o_city)

st.write('Vous avez choisi', op_o_city)

#st.dataframe(df_pt_1.loc[option])
op_d_city = st.selectbox("Sélectionnez une ville d'arrivé ", df_pt_1.loc[op_o_city].index)


df_f = df_pt_1.loc[op_o_city].loc[op_d_city]

li_metric = ['amin', 'mean', 'amax', ]

st.text("Sommes des trajets entre "+ op_o_city + " et " + op_d_city + " : " + str(int(df_f.loc["len"].loc["time"])))

title_time = "Statistique entre " + op_o_city + " et " + op_d_city

st.text(title_time)

col1, col2, col3 = st.columns(3)
col1.metric("Min", str(round(df_f.loc[li_metric[0]].loc["time"],1)) + " H")
col2.metric("Moyenne", str(round(df_f.loc[li_metric[1]].loc["time"],1)) + " H")
col3.metric("Max", str(round(df_f.loc[li_metric[2]].loc["time"],1)) + " H")

#title_price = "Statistique sur le prix entre " + op_o_city + " et " + op_d_city
#st.text(title_price)

col1, col2, col3 = st.columns(3)
col1.metric("Min", str(round(df_f.loc[li_metric[0]].loc["price_in_cents"]/100,2)) + " €")
col2.metric("Moyenne", str(round(df_f.loc[li_metric[1]].loc["price_in_cents"]/100,2)) + " €")
col3.metric("Max", str(round(df_f.loc[li_metric[2]].loc["price_in_cents"]/100,2)) + " €")




