import pandas as pd
import numpy as np
#from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import catboost as cat_
import seaborn as sns
import lightgbm as lgb
import streamlit as st
from sklearn.metrics import mean_squared_error
import plotly.figure_factory as ff
import altair as alt
import base64
import csv
from sklearn.metrics import confusion_matrix
from streamlit import metrics
#@st.cache
st.set_page_config(page_title="Predicting model to determine if a building will have an insurance claim",
                     layout='wide')
st.write("""# Predicting model to determine if a building will have an insurance claim""")
st.sidebar.header("Data variables")
# Import the train data
train = pd.read_csv('train_data.csv')

# Import the test data
test = pd.read_csv('test_data.csv')
#Submission = pd.read_csv('SUPCOM_SampleSubmission.csv')
st.subheader("Convert the dummy variables to Categorical values")
ntrain = train.shape[0]
ntest = test.shape[0]
data = pd.concat((train, test)).reset_index(drop = True)

#st.sidebar.header('Upload your CSV data')
#uploaded_file = st.sidebar.file_uploader("User_dataset.csv", type=["csv"])
#st.sidebar.markdown("""
#[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
#""")
# Combine train and test data for easy preprocessing

# Observing the dataset 
st.subheader("Observing the dataset")
describe = data.describe()
information = data.info()

st.write(describe)
st.subheader('Class labels and their corresponding index numbers')
st.write(data.columns)
data.drop(['Customer Id','Geo_Code'], inplace=True, axis=1)
data['Building_Fenced'].unique()
data['Building_Painted'].unique()
data['Garden'].unique()
data['NumberOfWindows'].unique()
data['Settlement'].unique()
data.NumberOfWindows = data.NumberOfWindows.replace('   .', '0')
data.NumberOfWindows = data.NumberOfWindows.replace('>=10', '10')
data.info()
data['NumberOfWindows'].unique()
data.NumberOfWindows = pd.to_numeric(data.NumberOfWindows, downcast='float')
data.info()

# converting Dummy variables to categorical variables
data = pd.get_dummies(data)

st.write("Row ", data.shape,  "Column" )

#Identifying the number of years in the column
data.YearOfObservation = data.YearOfObservation.astype('str')
unique_year = []
for i in range(len(data.YearOfObservation)):
  if data.YearOfObservation[i] not in unique_year:
    unique_year.append(data.YearOfObservation[i])

#Replacing the year values
data.YearOfObservation = data.YearOfObservation.replace('2016', 1)
data.YearOfObservation = data.YearOfObservation.replace('2015', 2)
data.YearOfObservation = data.YearOfObservation.replace('2014', 3)
data.YearOfObservation = data.YearOfObservation.replace('2013', 4)
data.YearOfObservation = data.YearOfObservation.replace('2012', 5)
st.write("""***note that the years have been converted below*** 
            \n2016  - 1 
            \n2015 - 2 
            \n2014 - 3 
            \n2013 - 4 
            \n2012 - 5
            """)
data.head()
    
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
data_sc_raw = scaler.fit_transform(data)
data_sc_raw = pd.DataFrame(data_sc_raw, columns = data.columns)
data = data_sc_raw

st.write(data.head())
st.subheader("Scaling the dataset for a better modelling")
st.write("Note that the data has been scaled down using **Maximum Absolute scaler**.")
st.markdown("i.e this means that every value in a specific column was divided by the absolute maximum number in the column.")
#matrix = np.array([[2.0, -4.0, 1], [1.0, 3.2, 0.0], [10.2, 22.0, 11]], column=)
st.write("""e.g, A=[1,2,3]; Now the ***maximum absolute number is 3***. divide every number in A by 3.Now A=[0.333, 0.75, 1]  
Also, B = [-4,2,1], the ***maximum absolute value is 4***. now B=[-1,0.5,0.25]""")
st.markdown("Now with the values you have, check the maximum numbers in the above ***description*** and divide the maximum by your number ")
def user_input_features():
    Building_Dimension = st.sidebar.slider('Building Dimension', 0.000033, 1.0, 0.059137)
    Building_Type = st.sidebar.slider('Building_Type', 0.250000, 1.0, 0.559512)
    Date_of_Occupancy = st.sidebar.slider('Date_of_Occupancy', 0.766369, 1.0,0.974733)
    Insured_Period = st.sidebar.slider('Insured_Period', 0.000000, 1.0, 0.913672)
    NumberOfWindows = st.sidebar.slider('NumberOfWindows', 0.000, 1.0, 0.187799)
    Residential = st.sidebar.slider('Residential', 0.000, 1.0, 0.281064)
    YearOfObservation = st.sidebar.slider('YearOfObservation', 0.2, 1.0, 0.670642)
    Building_Fenced_N = st.sidebar.slider('Building_Fenced_N', 0.0, 1.0, 0.433767)
    Building_Fenced_V = st.sidebar.slider('Building_Fenced_V', 0.0, 1.0, 0.566233)
    Building_Painted_N = st.sidebar.slider('Building_Painted_N', 0.0, 1.0, 0.309219)
    Building_Painted_V = st.sidebar.slider('Building_Painted_V', 0.0, 1.0, 0.690781)
    Garden_O = st.sidebar.slider('Garden_O', 0.0, 1.0,	0.432789)
    Garden_V = st.sidebar.slider('Garden_V', 0.0, 1.0, 0.566135)
    Settlement_R = st.sidebar.slider('Settlement_R', 0.0, 1.0, 0.433962)
    Settlement_U = st.sidebar.slider('Settlement_U', 0.0, 1.0, 0.566038)
    data = {'Building Dimension': Building_Dimension,
            'Building_Type': Building_Type,
            'Date_of_Occupancy': Date_of_Occupancy,
            'Insured_Period': Insured_Period,
            'NumberOfWindows': NumberOfWindows,
            'Residential': Residential,
            'YearOfObservation': YearOfObservation,
            'Building_Fenced_N': Building_Fenced_N,
            'Building_Fenced_V': Building_Fenced_V,
            'Building_Painted_V': Building_Painted_V,
            'Building_Painted_N': Building_Painted_N,
            'Garden_O': Garden_O,
            'Garden_V': Garden_V,
            'Settlement_R': Settlement_R,
            'Settlement_U': Settlement_U}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
describe = data.describe()


st.write(describe)
st.write('''
   **Variable Description**
Customer Id: Identification number for the Policy holder
YearOfObservation: year of observation for the insured policy
Insured_Period: "duration of insurance policy in Olusola Insurance. (Ex: Full year insurance, Policy Duration = 1; 6 months = 0.5"
Residential: is the building a residential building or not
Building_Painted: "is the building painted or not (N-Painted, V-Not Painted, N=1, v=0)"
Building_Fence: "is the building fence or not (N-Fenced, V-Not Fenced, n=1, V=0)"
Garden: "building has garden or not (V-has garden,  O-no garden, V=1, O=0)"
Settlement: "Area where the building is located. (R- rural area,  U- urban area, R= 1, U=0)"
Building Dimension: Size of the insured building in m2
Building_Type: "The type of building (Type 1, 2, 3, 4)"
Date_of_Occupancy: date building was first occupied
NumberOfWindows: number of windows in the building
Geo Code: Geographical Code of the Insured building
Claim: "target variable. (0: no claim, 1: at least one claim over insured period)."
''')

st.subheader("User Input Parameter")
st.write(df)

from sklearn.model_selection import train_test_split
X = data.drop(['Claim'], axis = 1)
y = data.Claim.values

train = data[:ntrain].copy()
test = data[ntrain:].copy()
test = test.reset_index(drop=True)

X_train = train.drop("Claim", axis=1)
y_train = train.Claim.values
X_test = test.drop("Claim", axis=1)
y_test = test.Claim.values

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#data.columns
# Visualisation
#st.subheader("VISUALISATION")


st.subheader("1. Modeling the data with Catboost")
# Model 1 - CatBoost
from catboost import CatBoostClassifier
classifier = CatBoostClassifier()
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)
#prediction = pd.DataFrame(prediction, columns='target')
#cm = confusion_matrix(y_test, prediction)
#st.write('Confusion Matrix is  ', cm)
prediction_prob = classifier.predict_proba(df)



st.subheader('Prediction Probability')
st.write(prediction_prob)
prob = classifier.predict_log_proba(X_test)
prob = pd.DataFrame(prob, columns=['Not_claimed', 'Claim'])

prob1 = prob['Claim']


prediction_df = pd.DataFrame(prediction, columns=['Predict'])
def filesdownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode() #string <-> convention
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction_df.csv">Download Predicted CSV File</a>'
    return href
st.markdown(filesdownload(prediction_df), unsafe_allow_html=True)