import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from math import radians, cos, sin, asin, sqrt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

data_cities = pd.read_csv("data/cities.csv")
data_providers = pd.read_csv('data/providers.csv')
data_stations = pd.read_csv("data/stations.csv")
data_ticket = pd.read_csv('data/ticket_data.csv')

# Création d'une fonction pour calculer le temps de trajet

def time(departure, arrival):
  t = arrival - departure
  return int(t.total_seconds()/3600)

# calucl distance entre deux cordonnées GPS 
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return int(km)

# traduction des moyens de transports

def trad_transport(transport_type):
  dico = {"carpooling": 'Covoiturage', "bus":"Bus", "train":"Train"}
  return dico[transport_type]    

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

#st.text(str("Sommes des trajets entre "+ op_o_city + " et " + op_d_city + " : " + str(int(df_f.loc["len"].loc["time"]))))

title_time = "Statistique entre " + op_o_city + " et " + op_d_city

#st.text(str(title_time))

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


# ************************************************************************************************************

# deuxième question : différence de prix moyen et durée selon le train, 
# le bus et le covoit selon la distance du trajet (0-200km, 201-800km, 800-2000km, 2000+km) 

df_2 = data_ticket[['departure_ts', 'arrival_ts', 'price_in_cents', 'o_city', 'd_city','company']]

#Création d'un DF regroupant nom des villes, les longitudes et latitudes des villes
# Ainsi que le prix et les temps de trajets

# Merge pour les villes d'origines (o_city)
df_2 = pd.merge(df_2, data_cities[['id','unique_name','latitude', 'longitude']], left_on="o_city", right_on='id')
df_2 = df_2.drop(["o_city","id"], axis=1)
df_2 = df_2.rename(columns={"unique_name": "o_city","latitude":"o_latitude","longitude":"o_longitude"})

#Merge pour les villes d'arrivés (d_city)
df_2 = pd.merge(df_2, data_cities[['id','unique_name','latitude', 'longitude']], left_on="d_city", right_on='id')
df_2 = df_2.drop(["d_city","id"], axis=1)
df_2 = df_2.rename(columns={"unique_name": "d_city","latitude":"d_latitude","longitude":"d_longitude"})

# Création d'une nouvelle colonne distance qui sera la distance en km entre chaque ville
# En utilisant la méthode Harversine

df_2['distance'] = df_2.apply(lambda x: haversine(x['o_longitude'], x['o_latitude'],x['d_longitude'], x['d_latitude']), axis=1)

# Création de la colonne du type de transport
df_2 = pd.merge(df_2, data_providers[['id','transport_type']], left_on="company", right_on='id')

# Création colonne label des distances en utilisat la méthode cut de pandas
# label des distances
label = ["0-200km", "201-800km", "800-2000km"]

# liste des points de coupe 
cut_points = [0,200,800,2000]

df_2['label_distance'] = pd.cut(x=df_2['distance'], bins=cut_points, labels=label, include_lowest=False)

# Calul du temps de trajet en heur
# Convertion des colonnes departure_ts et arrival_ts en format date
df_2['departure_ts'] = pd.to_datetime(df_2['departure_ts'])
df_2['arrival_ts'] = pd.to_datetime(df_2['arrival_ts'])
df_2['time'] = df_2.apply(lambda x: time(x['departure_ts'], x['arrival_ts']), axis=1)

# traduction des moyens de transports
df_2['transport_type'] = df_2['transport_type'].apply(trad_transport)

df_pt_3 = pd.pivot_table(df_2, index=['transport_type','label_distance'],values=['price_in_cents','time'],aggfunc=[np.mean])

#st.dataframe(df_pt_3)

option_transport = st.selectbox("Sélectionnez un moyen de transport ", df_2['transport_type'].unique())

option_label_distance = st.selectbox("Sélectionnez une fourchette de distances ", label)

st.text = "Statistique de voyage en " + option_transport + "selon la distance" + option_label_distance

st.text = "Différence de temps moyen selon le Covoiturage et le Train"

col1, col2 = st.columns(2)

if option_transport == "Bus":

  dif_mean_1 = str(round(df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'] - df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'],2)) + " H"
  dif_mean_2 = str(round(df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'] - df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'],2)) + " H"

  delta_mean_1 = round(((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'] / df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'])*100),2)
  delta_mean_2 = round(((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'] / df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'])*100),2)

  dif_mean_2 = str(dif_mean_2) + "%"
  delta_mean_2 = str(delta_mean_2) + "%"
  col1.metric("Différence de temps moyen avec le Covoiturage", dif_mean_1, delta_mean_1)
  col2.metric("Différence de temps moyen avec le Train", dif_mean_2, delta_mean_2)

  dif_mean_1 = str(round((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] - df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])/100,2)) + " H"
  dif_mean_2 = str(round((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] - df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])/100,2)) + " H"

  delta_mean_1 = round(((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] / df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])*100),2)
  delta_mean_2 = round(((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] / df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])*100),2)

  dif_mean_2 = str(dif_mean_2) + "%"
  delta_mean_2 = str(delta_mean_2) + "%"
  col1.metric("Différence de prix moyen avec le Covoiturage", dif_mean_1, delta_mean_1)
  col2.metric("Différence de prix moyen avec le Train", dif_mean_2, delta_mean_2)

if option_transport == "Covoiturage":

  dif_mean_1 = str(round(df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'] - df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'],2)) + " H"
  dif_mean_2 = str(round(df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'] - df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'],2)) + " H"

  delta_mean_1 = round(((df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'] / df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'])*100),2)
  delta_mean_2 = round(((df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'] / df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'])*100),2)

  dif_mean_2 = str(dif_mean_2) + "%"
  delta_mean_2 = str(delta_mean_2) + "%"
  col1.metric("Différence de temps moyen avec le Bus", dif_mean_1, delta_mean_1)
  col2.metric("Différence de temps moyen avec le Train", dif_mean_2, delta_mean_2)

  dif_mean_1 = str(round((df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] - df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])/100,2)) + " H"
  dif_mean_2 = str(round((df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] - df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])/100,2)) + " H"

  delta_mean_1 = round(((df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] / df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])*100),2)
  delta_mean_2 = round(((df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] / df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])*100),2)

  dif_mean_2 = str(dif_mean_2) + "%"
  delta_mean_2 = str(delta_mean_2) + "%"
  col1.metric("Différence de prix moyen avec le Bus", dif_mean_1, delta_mean_1)
  col2.metric("Différence de prix moyen avec le Train", dif_mean_2, delta_mean_2)

if option_transport == "Train":

  dif_mean_1 = str(round(df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'] - df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'],2)) + " H"
  dif_mean_2 = str(round(df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'] - df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'],2)) + " H"

  delta_mean_1 = round(((df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'] / df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['time'])*100),2)
  delta_mean_2 = round(((df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['time'] / df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['time'])*100),2)

  dif_mean_2 = str(dif_mean_2) + "%"
  delta_mean_2 = str(delta_mean_2) + "%"
  col1.metric("Différence de temps moyen avec le Bus", dif_mean_1, delta_mean_1)
  col2.metric("Différence de temps moyen avec le Train", dif_mean_2, delta_mean_2)

  dif_mean_1 = str(round((df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] - df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])/100,2)) + " H"
  dif_mean_2 = str(round((df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] - df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])/100,2)) + " H"

  delta_mean_1 = round(((df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] / df_pt_3.loc['Bus'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])*100),2)
  delta_mean_2 = round(((df_pt_3.loc['Train'].loc[option_label_distance].loc['mean'].loc['price_in_cents'] / df_pt_3.loc['Covoiturage'].loc[option_label_distance].loc['mean'].loc['price_in_cents'])*100),2)

  dif_mean_2 = str(dif_mean_2) + "%"
  delta_mean_2 = str(delta_mean_2) + "%"
  col1.metric("Différence de prix moyen avec le Bus", dif_mean_1, delta_mean_1)
  col2.metric("Différence de prix moyen avec le Covoiturage", dif_mean_2, delta_mean_2)

###################################################################################################
#  Machine Learning

# Création du modele de 'arbre de décision
model = DecisionTreeRegressor()

# Création du jeu d'entrainement (dataset)
df_3 = df_2[['transport_type','o_city','d_city','price_in_cents']]

# on factorize les colonnes non numériques
df_3['transport_type_f'] = df_3['transport_type'].factorize()[0]
df_3['o_city_f'] = df_3['o_city'].factorize()[0]
df_3['d_city_f'] = df_3['d_city'].factorize()[0]

# Création du dataframe du jeu d'entrainement
X = df_3.drop(['o_city','d_city','price_in_cents','transport_type'], axis = 1)
y = df_3['price_in_cents']
X_train, X_test, y_train, y_test = train_test_split(X,y)

model_DTR = model.fit(X_train, y_train)

# Création des dictionnaires pour les entrées et la prédiction

df_3_to_dico = df_3[['transport_type','transport_type_f']]
df_3_to_dico = df_3_to_dico.drop_duplicates()
df_3_to_dico.reset_index(drop=True, inplace=True)
dico_transport_type = {}
for i in df_3_to_dico.index :
  dico_transport_type[df_3_to_dico['transport_type'][i]] = df_3_to_dico['transport_type_f'][i]

df_3_to_dico = df_3[['o_city','o_city_f']]
df_3_to_dico = df_3_to_dico.drop_duplicates()
df_3_to_dico.reset_index(drop=True, inplace=True)
dico_o_city = {}
for i in df_3_to_dico.index :
  dico_o_city[df_3_to_dico['o_city'][i]] = df_3_to_dico['o_city_f'][i]

df_3_to_dico = df_3[['d_city','d_city_f']]
df_3_to_dico = df_3_to_dico.drop_duplicates()
df_3_to_dico.reset_index(drop=True, inplace=True)
dico_d_city = {}
for i in df_3_to_dico.index :
  dico_d_city[df_3_to_dico['d_city'][i]] = df_3_to_dico['d_city_f'][i]

#Création d'une side bar

st.sidebar.header("Les paramètres d'entrée pour la prédiction du prix")


def user_input():
  option_ml_transport = st.sidebar.selectbox("Choix du moyen de transport", dico_transport_type.keys())
  option_ml_o_city = st.sidebar.selectbox("Choix de la ville de départ", dico_o_city.keys())
  option_ml_o_city = st.sidebar.selectbox("Choix de la ville d'arrivée", dico_d_city.keys())

  data = {'transport':dico_transport_type[option_ml_transport],
  'o_city': dico_o_city[option_ml_o_city],
  'd_city':dico_d_city[option_ml_o_city]}

  return pd.DataFrame(data, index=[0])  

df = user_input()

prediction = model_DTR.predict(df)

st.subheader("Prediction du prix d'un voyage ")
st.subheader("Le prix du voyage serait de : ")
st.write(str(str(round(prediction[0]/100,2)) + " €"))
