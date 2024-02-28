import streamlit as st
import prediction
import eda


page = st.sidebar.selectbox(label= 'Select Menu: ', options=['Home','Data Analysis','Sentiment Prediction'])
    
if page == 'Home':
    st.header("Welcome to the Amazon's Customer Feedback Analysis Web")
    st.write("\n")
    st.write("This is a web app to illustrate the results of EDA (Exploratory Data Analysis) as well as sentiment predictions based on given review texts. If you're curious about the features available in this web app, please check the menu on the left sidebar.")
elif page == 'Data Analysis':
    eda.run()
else:
    prediction.run()