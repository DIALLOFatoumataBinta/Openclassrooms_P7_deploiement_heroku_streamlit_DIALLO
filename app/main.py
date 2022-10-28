#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt

from PIL import Image
from lime.lime_tabular import LimeTabularExplainer



# In[2]:


st.set_page_config(layout="wide")


def main():
    
    #cl1,cl2,cl3,cl4,cl5 = st.beta_columns(5)
    
    image = Image.open("logo.png")
    
    st.sidebar.image(image,use_column_width=True)
    
    st.markdown("<h1 style='text-align: center; color: black;'>Bienvenue sur votre portail de scoring client</h1>", unsafe_allow_html=True)   

    #@st.cache
    def load_data(path,filename):
        data = pd.read_csv(path+filename)
        dd = pd.read_csv("DATA/app_train.csv")
        list_id = dd['SK_ID_CURR'].tolist()
        return list_id,data
        #return data
    
    def amount_formatter(amount):

        x = round(amount)
        x = "{:,.2f}".format(x)
        x = x.split(".")[0]
        x = x.replace(","," ")

        return x

    def write_client_info(ID_client,app_train):

        list_infos = ['SK_ID_CURR','NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR',
                      'FLAG_OWN_REALTY','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY',
                      'NAME_INCOME_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','DAYS_BIRTH','DAYS_EMPLOYED',
                      'CNT_FAM_MEMBERS','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','Age']

        X = app_train[list_infos]
        client_infos = X[X['SK_ID_CURR'] == int(ID_client)]

        st.text("")
        st.header("Informations descriptives du client:")
        st.text("")

        cls1,cls2 = st.columns(2)

        #age = int(np.abs(client_infos['DAYS_BIRTH'].tolist()[0])/365)
        age =  client_infos['Age'].tolist()[0]
        genre = str(client_infos['CODE_GENDER'].tolist()[0])
        situation_familiale = str(client_infos['NAME_FAMILY_STATUS'].tolist()[0])
        nb_enfants = client_infos['CNT_CHILDREN'].tolist()[0]
        revenus_total = client_infos['AMT_INCOME_TOTAL'].tolist()[0]
        montant_credit = client_infos['AMT_CREDIT'].tolist()[0]
        cnt_fam_members = int(client_infos['CNT_FAM_MEMBERS'].tolist()[0])
        amt_annuity = client_infos['AMT_ANNUITY'].tolist()[0]
        name_contract_type = client_infos['NAME_CONTRACT_TYPE'].tolist()[0]

        with cls1:
            st.write("**IDentifiant**: "+str(ID_client))
            st.write("**Age**: "+str(age)+" ans")
            st.write("**Genre**: "+genre)
            st.write("**Situation familiale**: "+situation_familiale)
            st.write("**Nombre d'enfants**: "+str(nb_enfants))

        with cls2:
            st.write("**Composition de la famille:** "+str(cnt_fam_members))
            st.write("**Révenus total:** "+str(amount_formatter(revenus_total))+" $")
            st.write("**Montant du crédit:** "+str(amount_formatter(montant_credit))+" $")
            st.write("**Annuité**: "+str(amount_formatter(amt_annuity))+" $")
            st.write("**Type de prêt**: "+str(name_contract_type))

        return age,genre,situation_familiale,nb_enfants,revenus_total, montant_credit



    def plot_features_importances(dataframe):

        model = pickle.load(open("LightGBM_imb.pkl",'rb'))

        cols = dataframe.drop(['SK_ID_CURR','TARGET'],axis=1).columns
        importances = model.best_estimator_.feature_importances_
        features_importances = pd.concat((pd.DataFrame(cols, columns = ['Variable']), 
                                         pd.DataFrame(importances, columns = ['Importance'])), 
                                        axis = 1).sort_values(by='Importance', ascending = False)

        cols = features_importances[["Variable", "Importance"]].groupby("Variable").mean().sort_values(by="Importance", ascending=False)[:40].index
        best_features = features_importances.loc[features_importances.Variable.isin(cols)]
        fig=plt.figure(figsize=(8, 10))
        sns.barplot(x="Importance", y="Variable", data=best_features.sort_values(by="Importance", ascending=False))

        #plt.title("Importance des variables",fontsize=40)
        plt.xlabel("Importance",fontsize=18)
        plt.ylabel("Variable",fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=14)
        st.pyplot(fig)

        return features_importances
    
    def regroupement(ID_client,app_train, var,types):
    
        cols = ["SK_ID_CURR","DAYS_BIRTH","CODE_GENDER","CNT_CHILDREN","AMT_INCOME_TOTAL",
                "AMT_CREDIT","AMT_ANNUITY", "CNT_FAM_MEMBERS", "NAME_CONTRACT_TYPE", "Age"]

        data = app_train[cols]
        X = data[data['SK_ID_CURR'] == int(ID_client)]
        new_data = data[data['SK_ID_CURR'] != int(ID_client)]

        fam_status = new_data[new_data['CNT_FAM_MEMBERS'] == X['CNT_FAM_MEMBERS'].values[0]]

        if types == "ménages":
            df_create = pd.concat((pd.DataFrame(["Client","Médiane/dataset"], columns = ['labels']), 
                              pd.DataFrame([X[var].values[0],
                                            new_data[var].median(),
                                            ],
                                           columns = ['Values'])),axis = 1)
        else:
            df_create = pd.concat((pd.DataFrame(["Client","Médiane/dataset","Médiane/ménages"], columns = ['labels']), 
                              pd.DataFrame([X[var].values[0],
                                            new_data[var].median(),
                                            fam_status[var].median(),],
                                           columns = ['Values'])),axis = 1)

        return df_create
    def plot_client_stats(ID_client,app_train):

        amt_income_total = regroupement(ID_client,app_train, "AMT_INCOME_TOTAL", "revenu")
        amt_credit = regroupement(ID_client,app_train, "AMT_CREDIT", "crédit")
        cnt_children = regroupement(ID_client,app_train, "CNT_FAM_MEMBERS", "ménages")
        amt_annuite = regroupement(ID_client,app_train, "AMT_ANNUITY", "annuité")
        Age = regroupement(ID_client,app_train, "Age", "annuité")    

        fig = plt.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')

        ax1 = plt.subplot(2,3,1)
        sns.barplot(x="labels", y="Values", data=amt_income_total,ax=ax1)
        plt.xticks(rotation=40)
        plt.title("Revenu total")

        ax2 = plt.subplot(2,3,2)
        sns.barplot(x="labels", y="Values", data=amt_credit,ax=ax2)
        plt.xticks(rotation=45)
        plt.title("Montant du crédit")


        ax3 = plt.subplot(2,3,3)
        sns.barplot(x="labels", y="Values", data=cnt_children,ax=ax3)
        plt.xticks(rotation=40)
        plt.title("Nombres ménages")

        ax4 = plt.subplot(2,3,5)
        sns.barplot(x="labels", y="Values", data=amt_annuite,ax=ax4)
        plt.xticks(rotation=40)
        plt.title("Nombres annuité")    

        ax5 = plt.subplot(2,3,6)
        sns.barplot(x="labels", y="Values", data=Age,ax=ax5)
        plt.xticks(rotation=40)
        plt.title("Age")


        plt.subplots_adjust(wspace=0.9)


        fig.tight_layout()
        st.pyplot(fig)

        return True
    
    
    def gauge_plot(arrow_index, labels):

        list_colors = np.linspace(0,1,int(len(labels)/2))
        size_of_groups=np.ones(len(labels))

        white_half = np.ones(len(list_colors))*.5
        color_half = list_colors

        cs1 = cm.RdYlGn_r(color_half)
        cs2 = cm.seismic(white_half)
        cs = np.concatenate([cs1,cs2])

        fig, ax = plt.subplots()

        ax.pie(size_of_groups, colors=cs, labels=labels)

        my_circle=plt.Circle( (0,0), 0.6, color='white')
        ax.add_artist(my_circle)

        arrow_angle = (arrow_index/float(len(list_colors)))*3.14159
        arrow_x = 0.8*math.cos(arrow_angle)
        arrow_y = 0.8*math.sin(arrow_angle)
        arr = plt.arrow(0,0,-arrow_x,arrow_y, width=.02, head_width=.05, head_length=.1, fc='k', ec='k')

        ax.add_artist(arr)
        ax.add_artist(plt.Circle((0, 0), radius=0.04, facecolor='k'))
        ax.add_artist(plt.Circle((0, 0), radius=0.03, facecolor='w', zorder=11))

        ax.set_aspect('equal')

        st.pyplot(fig)

        return True
    
    def predict_target(ID_client):
        model = pickle.load(open("LightGBM_imb.pkl",'rb'))

        path_read = 'DATA/'
        filename = "df_current_clients.csv"
        data = pd.read_csv(path_read+filename)    

        X = data[data['SK_ID_CURR'] == int(ID_client)]
        X.drop(['SK_ID_CURR','TARGET'],axis=1,inplace=True)

        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0][prediction]

        return prediction, proba
    
    #@st.cache
    def get_client_score(ID_client,prediction,proba):

        T = pd.DataFrame(columns=["A","B","Note"])

        j = 0
        for i in range(1,21):
            T.loc[i-1,"A"] = j
            T.loc[i-1,"B"] = j + 0.05
            T.loc[i-1,"Note"] = i
            j = j + 0.05

        if prediction == 1:
            proba = 1 - proba

        prob_data = pd.DataFrame(columns=["Proba","Note","arrow_idx"])

        X_val = T[(proba >=T["A"]) & (proba <T["B"])]

        prob_data = prob_data.append(pd.DataFrame(
                {'Proba' : [proba],'Note' : [X_val["Note"].values[0]],
                 'arrow_idx' : [X_val["Note"].values[0] * 50]}),
                ignore_index=True)

        prob_data['SK_ID_CURR'] = int(ID_client)
        prob_data['TARGET'] = prediction

        return prob_data
    
    
    def get_local_interpretation(ID_client,dataframe,features_importances):

        model = pickle.load(open("LightGBM_imb.pkl",'rb'))
        X = dataframe[dataframe['SK_ID_CURR'] == int(ID_client)]
        X = X.drop(['SK_ID_CURR', 'TARGET'], axis=1)
        dataframe = dataframe.drop(['SK_ID_CURR', 'TARGET'], axis=1)

        X_train = dataframe.sample(frac=0.1,random_state=42).values

        explainer = LimeTabularExplainer(training_data = X_train,
                                         mode='classification',
                                         feature_names = dataframe.columns,
                                         training_labels = dataframe.columns.tolist(),
                                         verbose=1,
                                         random_state=42)
        #st.write(np.array(X))
        #st.write(type(np.array(X)))
        explanation = explainer.explain_instance(np.ravel(np.array(X)), 
                                                 predict_fn = model.predict_proba,
                                                 labels=[0,1],
                                                 num_features = len(dataframe.columns))

        #fig = explanation.as_pyplot_figure(label=label)
        #st.pyplot(fig)

        return explanation
    def pyplot_lime_explanation(explanation, features_importances):
        
        features_list = features_importances["Variable"].values[:31]
        
        exp = explanation.as_list()
        
        fig = plt.figure()
        vals = []
        names = []
        
        import re
        
        for x in exp:
            i = re.split('<|>|=',x[0])
        
            try:
                float(i[0].strip(" "))
                i = i[1].strip(" ")
            except:
                i = i[0].strip(" ")
        
            if i in features_list:
                vals.append(x[1])
                names.append(x[0])
      
        vals.reverse()
        names.reverse()
        
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(vals)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        
        plt.title("Interprétabilité locale",size=22, fontweight="bold")
        plt.yticks(pos, names)
        plt.xlabel("Importance",fontsize=18)
        plt.ylabel("Variable",fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=14)
        #plt.savefig("fig/interpretabilite_locale.png")
        
        st.pyplot(fig)

        return
    
    #code pour streamlit
    path_read = 'DATA/'
    filename="app_train.csv"
    list_id,df = load_data(path_read, "df_current_clients.csv")
    #df = load_data(path_read, "df_current_clients.csv")
    df2 = pd.read_csv(path_read+filename)
    modelname="XGBoost"
    

    st.text("")
    st.text("")            

    id_client = st.sidebar.text_input("Entrez l'identifiant d'un client:",)

    if id_client == '':
        st.write("S'il vous plait entrez un identifiant correct.")
    elif int(id_client) in list_id:

        age,genre,situation_familiale,nb_enfants,revenus_total, montant_credit = write_client_info(int(id_client), df2)
        
        select = st.sidebar.selectbox("Afficher la comparaison avec d'autres clients:", 
                                      ["Types données","Revenu Total", "Montant du crédit","Nombres annuité","Age"],key='1')
        
        st.markdown("<h3 style='text-align: center; color: black;'>Afficher la comparaison avec d'autres clients</h3>", unsafe_allow_html=True)

        plot_client_stats(id_client,df2)

        
        prediction_her, proba_her= predict_target(id_client)

        with st.spinner('Chargement du score du client...'):

            score_list = get_client_score(id_client,prediction_her,proba_her)
            X = score_list[score_list['SK_ID_CURR'] == int(id_client)]

            new_data = df2[df2['SK_ID_CURR'] == int(id_client)]

            st.text("")
            st.header("Prédiction:")
            st.text("")

            cols1,cols2 = st.columns(2)
            with cols1:     
                clt_score = X['Note'].values[0]
                if clt_score > 14:
                    st.markdown(f"Ce client a une note de **{clt_score:02d}/20** pour rembourser son crédit. Le risque de défaut de paiement de ce client est **faible**.")
                else:
                    st.markdown(f"Ce client a une note de **{clt_score:02d}/20** pour rembourser son crédit. Le risque de défaut de paiement de ce client est **élevé**.")

                if not new_data['TARGET'].isnull().values.any():
                    if int(new_data['TARGET'] == 1):
                        st.markdown("Pour rappel, ce client a été en défaut de paiement auparavant.")
                    else:
                        st.markdown("Pour rappel, ce client n'a pas été en défaut de paiement auparavant.")

                st.text("")
                st.text("")
                st.markdown("**NB:** Le seuil de **14/20** a été défini pour évaluer le niveau du risque de défaut de paiement d'un client: pour une note **inférieure à 14/20**, le risque est **élevé** et pour une note **supérieure à 14/20** le risque est **faible**. Plus la note se rapproche de **20/20** plus le risque est faible et plus la note est faible plus le client est risqué.")

            with cols2:
                values = np.linspace(0,1,1000)
                labels = [' ']*len(values)*2
                labels[25] = '20'
                labels[250] = '15'
                labels[500] = '10'
                labels[750] = '5'
                labels[975] = '0'

                arrow_index = X['arrow_idx'].values[0]
                gauge_plot(arrow_index,labels=labels)





        st.markdown("<h3 style='text-align: center; color: black;'>Explication poussée (feature importance)</h3>", unsafe_allow_html=True)
        features_importances = plot_features_importances(df)
        st.markdown("<h3 style='text-align: center; color: black;'>Explication locale de la prédiction</h3>", unsafe_allow_html=True)
        exp = get_local_interpretation(id_client,df,features_importances)
        pyplot_lime_explanation(exp, features_importances)

        

    else:
        st.warning("Identifiant incorrect. Veuillez saisir un identifiant correct")
if __name__ == '__main__':
    main()
