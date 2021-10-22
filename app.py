# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:00:38 2021

@author: Elijah_Nkuah
"""
# Core Pkgs
import streamlit as st 
st.set_page_config(page_title = "Bank Account", layout = 'wide', initial_sidebar_state = 'auto')

# EDA Pkgs
import pandas as pd 
import numpy as np 
from PIL import Image


# Utils
import os
import joblib 
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
matplotlib.use('Agg')

# DB
from database_acc import *
#from managed_db import *
# Password 
def generate_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password,hashed_text):
	if generate_hashes(password) == hashed_text:
		return hashed_text
	return False


feature_names_best = ['country', 'year', 'location_type',
       'cellphone_access', 'household_size', 'age_of_respondent',
       'gender_of_respondent', 'relationship_with_head', 'marital_status',
       'education_level', 'job_type']


gender_dict = {"Male":1,"Female":2}
country_dict = {'Kenya':1, 'Rwanda':2, 'Tanzania':3, 'Uganda':4}
year_dict = {'2018':1, '2017':2, '2016':3}
feature_dict = {"No":0,"Yes":1}
location_dict = {'Rural':1, 'Urban':2}
relationship_dict = {'Head of Household':1, 'Spouse':2, 'Child':3, 'Parent':4, 'Other relative':5, 'Other non-relatives':6}
marital_dict = {'Married/Living together':1, 'Single/Never Married':2, 'Widowed':3,'Divorced/Seperated':4, 'Dont know':5}
education_dict = {'No formal education':0,'Primary education':1,'Secondary education':2,'Vocational/Specialised training':3,
                  'Tertiary education':4, 'Other/Dont know/RTA':5}
job_dict = {'Dont Know/Refuse to answer':0,'No Income':1,'Other Income':2,'Remittance Dependent':3,'Government Dependent':4,
            'Self employed':5,'Farming and Fishing':6,'Informally employed':7,'Formally employed Private':8, 
       'Formally employed Government':9}


def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key
def get_cvalue(val):
	country_dict = {'Kenya':1, 'Rwanda':2, 'Tanzania':3, 'Uganda':4}
	for key,value in year_dict.items():
		if val == key:
			return value
def get_yvalue(val):
	year = {'2018':1, '2017':2, '2016':3}
	for key,value in year.items():
		if val == key:
			return value
def get_fvalue(val):
	feature_dict = {"No":1,"Yes":2}
	for key,value in feature_dict.items():
		if val == key:
			return value
def get_lvalue(val):
    location_dict = {'Rural':1, 'Urban':2}
    for key,value in location_dict.items():
            if val == key:
                    return value
def get_rvalue(val):
	relationship_dict = {'Head of Household':1, 'Spouse':2, 'Child':3, 'Parent':4, 'Other relative':5, 'Other non-relatives':6}
	for key,value in relationship_dict.items():
		if val == key:
			return value
def get_mvalue(val):
	marital_dict = {'Married/Living together':1, 'Single/Never Married':2, 'Widowed':3,'Divorced/Seperated':4, 'Dont know':5}
	for key,value in marital_dict.items():
		if val == key:
			return value
def get_evalue(val):
	education_dict = {'No formal education':0,'Primary education':1,'Secondary education':2,'Vocational/Specialised training':3,
                  'Tertiary education':4, 'Other/Dont know/RTA':5}
	for key,value in education_dict.items():
		if val == key:
			return value
def get_jvalue(val):
	job_dict = {'Dont Know/Refuse to answer':0,'No Income':1,'Other Income':2,'Remittance Dependent':3,'Government Dependent':4,
            'Self employed':5,'Farming and Fishing':6,'Informally employed':7,'Formally employed Private':8, 
       'Formally employed Government':9}
	for key,value in job_dict.items():
		if val == key:
			return value

# Load ML Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# ML Interpretation
#pip install lime
#import lime
#import lime.lime_tabular


html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Predicting those who may have Bank Account </h1>
		<h5 style="color:white;text-align:center;">BANK ACCOUNT </h5>
		</div>
		"""

# Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""


descriptive_message_temp ="""
	<div style="background-color:silver;"overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:#FFA500;padding:10px">Bank Account Summary</h3>
		<p>It is clear to see that having a bank account in your name give you greater independence and 
        allows you to organise your money and access it easily. UNICEF did research on Who in Africa is most likely to have a bank account! This research was done in Kenya, Rwanda, Tanzania and Uganda </p>
	</div>
	"""
Steps_to_follow ="""
	<div style="background-color:silver;"overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:#FFA500;padding:10px">Follow The Steps Below:</h3>
		<ul>
		<li style="text-align:justify;color:white;padding:10px">Signup if not already having account from the sidebar</li>
		<li style="text-align:justify;color:white;padding:10px">After signing up;</li>
		<li style="text-align:justify;color:white;padding:10px">Log in with your details</li>
		<li style="text-align:justify;color:white;padding:10px">i.e Username & Password</li>
		<ul>
	</div>
	"""
#@st.cache
#def load_image(img):
#image =Image.open(os.path.join(img))
#	return im
	
#st.set_option('deprecation.showPyplotGlobalUse', False)

#def change_avatar(sex):
#	if sex == "male":
#		avatar_img = 'img_avatar.png'
#	else:
#		avatar_img = 'img_avatar2.png'
#	return avatar_img


def main():
	"""Prediction App for persons having bank account or not"""
	st.image(Image.open(os.path.join("images/bank_banner.png")))
	menu = ["Home","Login","Signup"]
	submenu = ["Plot","Prediction","Metrics"]

	choice = st.sidebar.selectbox("Menu",menu)
	if choice == "Home":
		st.header("Home")
		st.markdown(descriptive_message_temp,unsafe_allow_html=True)
		st.markdown(Steps_to_follow, unsafe_allow_html=True)
		st.sidebar.image(Image.open(os.path.join('images/LOGO.png')))

	elif choice == "Login":
		st.sidebar.image(Image.open(os.path.join('images/LOGO.png')))
		username = st.sidebar.text_input("Username")
		password = st.sidebar.text_input("Password",type='password')
		if st.sidebar.checkbox("Login"):
			create_usertable()
			hashed_pswd = generate_hashes(password)
			result = login_user(username,verify_hashes(password,hashed_pswd))
			# if password == "12345":
			if result:
				st.success("Welcome {} to Bank Account Prediction App".format(username))
				st.image(Image.open(os.path.join('images/welcome.png')))
				activity = st.selectbox("Activity",submenu)
				st.sidebar.image(Image.open(os.path.join('images/LOGO.png')))
				if activity == "Plot":
					st.subheader("Data Visualisation")
					df = pd.read_csv("data/Train.csv")
					st.markdown("Total dataset used is {}".format(df.shape))
					st.write(df.head())
					fig4 = plt.figure(figsize=(20, 8))
					plt.title("Job type of Respondents", fontsize=20)
					sns.countplot(x = "job_type", data = df)
					st.pyplot(fig4)
					fig4a, ax = plt.subplots()
					df['job_type'].value_counts().plot(kind='pie', title="Job type of Respondents")
					st.pyplot(fig4a)
					fig5, ax = plt.subplots()
					df['gender_of_respondent'].value_counts().plot(kind='pie', title="Gender of Respondent")
					st.pyplot(fig5)
					fig, ax = plt.subplots()
					df['bank_account'].value_counts().plot(kind='bar', color="#ADD8E6", title="Not Bank Holder:0, Bank Holder: 1")
					st.pyplot(fig)
					fig1, ax = plt.subplots()
					df['education_level'].value_counts().plot(kind='bar', title="Education Level",color="#D2691E")
					st.pyplot(fig1)
					fig2, ax = plt.subplots()
					df['marital_status'].value_counts().plot(kind='bar', title="Marital Status")
					st.pyplot(fig2)
					fig3 = plt.figure(figsize=(15, 8))
					plt.title("Gender of Respondent", fontsize=20)
					sns.countplot(x = "gender_of_respondent", data = df)
					st.pyplot(fig3)
					
					

				elif activity == "Prediction":
					st.subheader("Predictive Analytics")

					country = st.selectbox("Country of Residence",tuple(country_dict.keys()))
					year = st.radio("Year",year_dict.keys())
					location = st.radio("Location Type",tuple(location_dict.keys()))
					cellphone = st.selectbox("Do You have access to phone? ", tuple(feature_dict.keys()))
					household = st.number_input("How many people do you leave together as one household?",1,25)
					age = st.number_input("What is your Age?", 5,110)
					sex = st.radio("Sex",tuple(gender_dict.keys()))
					relationship = st.selectbox("What is your relationship to the head of your household?", tuple(relationship_dict.keys()))
					marital = st.selectbox("What is your Marital Status?", tuple(marital_dict.keys()))
					education = st.selectbox("What is your level of Education?", tuple(education_dict.keys()))
					job = st.selectbox("What is your Employment Status?", tuple(job_dict.keys()))

					feature_list = [get_cvalue(country),get_yvalue(year),get_lvalue(location),get_fvalue(cellphone),household,age,get_value(sex,gender_dict),get_rvalue(relationship),get_mvalue(marital), get_evalue(education),get_jvalue(job)]
					st.write("The Number of independent varaiables is {}".format(len(feature_list)))
					pretty_result = {"Country":country,"year":year,"location":location,"cellphone":cellphone,"household":household,"Age":age,"Sex":sex,"Relationship":relationship,"Marital Status":marital,"Education Level":education,"Job Type":job}
					st.json(pretty_result)
					single_sample = np.array(feature_list).reshape(1,-1)

					# ML
					model_choice = st.selectbox("Select Model",["Lightgbm","Catboost","Xgboost"])
					if st.button("Predict"):
						if model_choice == "Lightgbm":
							loaded_model = load_model("models/lgb_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						elif model_choice == "Catboost":
							loaded_model = load_model("models/cat_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)
						else:
							st.info("Note: The prediction by this model is opposite. So reverse it when you see the result. Still working on it. Sorry for an inconvenient")
							loaded_model = load_model("models/xgb_model_2.pkl")
							prediction = loaded_model.predict(single_sample)
							pred_prob = loaded_model.predict_proba(single_sample)

						
						if prediction == 1:
							st.success("Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							#st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						elif prediction == 2:
							st.success("Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							#st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						elif prediction == 3:
							st.success("Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)
							st.subheader("Prescriptive Analytics")
							#st.markdown(prescriptive_message_temp,unsafe_allow_html=True)
						else:
							st.warning("Not Bank Account Holder")
							pred_probability_score = {"Probability of not having Bank account":pred_prob[0][0]*100,"Probability of having Bank account":pred_prob[0][1]*100}
							st.subheader("Prediction Probability Score using {}".format(model_choice))
							st.json(pred_probability_score)	


			else:
				st.warning("Incorrect Username/Password")

    
	elif choice == "Signup":
		st.sidebar.image(Image.open(os.path.join('images/LOGO.png')))
		new_username = st.text_input("User Name")
		new_password = st.text_input("Password", type='password')
		confirmed_password = st.text_input("Confirm Password", type='password')
		if new_password == confirmed_password:
			st.success("Password Confirmed")
		else:
			st.warning("Passwords not the same")
		if st.button("Submit"):
			create_usertable()
			hashed_new_password = generate_hashes(new_password)
			add_userdata(new_username, hashed_new_password)
			st.success("You have successfully created a new account")
			st.info("Login To Get Started")


if __name__ == '__main__':
	main()
