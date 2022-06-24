from pathlib import Path
import os
import numpy as np
import pandas as pd
import time
import datetime
import sklearn
import geopy.geocoders
import joblib
from geopy.geocoders import Nominatim
BASE_DIR = Path(__file__).resolve().parent.parent
# reading the models
logistic_01=joblib.load(os.path.join(BASE_DIR,'ml_utils\ML_models\Logistic_regression_01.pkl'))
scaler_logistic_01 = joblib.load(os.path.join(BASE_DIR,"ml_utils\ML_models\logistic_regression_scaler_01.pkl"))

running_model = joblib.load(os.path.join(BASE_DIR,'ml_utils\ML_models\BestModel_RunningStartups.pkl'))
running_scaler = joblib.load(os.path.join(BASE_DIR,'ml_utils\ML_models\RunningStartups_scaler.pkl'))

colsed_model = joblib.load(os.path.join(BASE_DIR,'ml_utils\ML_models\BestModel_closed_startups.pkl'))
closed_scaler = joblib.load(os.path.join(BASE_DIR,'ml_utils\ML_models\Closed_startups_scaler.pkl'))

def pre_process(data):
    print(data)
    df_columns = ['funding_rounds', 'funding_total_usd', 'milestones', 'relationships',
       'lat', 'lng', 'founded_at_year', 'first_funding_at_year',
       'last_funding_at_year', 'first_milestone_at_year',
       'last_milestone_at_year', 'ADVERTISING', 'BIOTECH', 'CONSULTING',
       'ECOMMERCE', 'EDUCATION', 'ENTERPRISE', 'GAMES_VIDEO', 'HARDWARE',
       'MOBILE', 'NETWORK_HOSTING', 'OTHER', 'PUBLIC_RELATIONS', 'SEARCH',
       'SOFTWARE', 'WEB', 'AUS', 'CAN', 'DEU', 'ESP', 'FRA', 'GBR', 'IND',
       'ISR', 'NLD', 'Other', 'USA', 'active_days']
    
    ## creating dataset to pass to the model
    # make a data frame of all X columns and initilize it to 0
    df = pd.DataFrame(0 ,index=[0],columns=df_columns)
    df.loc[0,"funding_rounds"] = float(data.get('funding_rounds'))
    df.loc[0,"funding_total_usd"] = float(data.get("funding_total_usd"))
    df.loc[0,"milestones"] = float(data.get("milestones"))
    df.loc[0,"relationships"] = float(data.get("relationships"))
    df.loc[0,"founded_at_year"] = float(data.get("founded_at_year"))
    df.loc[0,"first_funding_at_year"] = float(data.get("first_funding_at_year"))
    df.loc[0,"last_funding_at_year"] = float(data.get("last_funding_at_year"))
    df.loc[0,"first_milestone_at_year"] = float(data.get("first_milestone_at_year"))
    df.loc[0,"last_milestone_at_year"] = float(data.get("last_milestone_at_year"))
    df.loc[0,str(data.get("country_code"))]=1.0
    df.loc[0,str(data.get('catogery_code'))]=1.0
    df.loc[0,'active_days'] = float(data.get('active_days'))
    
    ## getting lattitude and longitude information from state
    geolocator = Nominatim(user_agent="my_request")
    location = geolocator.geocode(data.get('state_code').lower())

    df.loc[0,'lat'] = float(location.latitude)
    df.loc[0,'lng'] = float(location.longitude)

    df1 = scaler_logistic_01.transform(df)
    out = str(logistic_01.predict(df1)[0])

    # ipo operating 1
    # 
    if(out=="1"):
        df2 = running_scaler.transform(df)
        out = str(running_model.predict(df2)[0])
    else:
        df3 = closed_scaler.transform(df)
        out = str(colsed_model.predict(df3)[0])   
    # aquired and closed 0 

    return out

#pre_process("hi");
