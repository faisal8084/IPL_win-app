import pandas as pd
import numpy as np
import joblib
import streamlit as st

pipe = joblib.load('iplpipe.pkl')
final_df = joblib.load('final_df.pkl')

st.title("IPL WIN PREDICTION")

batting_team =st.selectbox('Select the batting team',sorted(final_df['batting_team'].unique()))
bowling_team =st.selectbox('Select the bowling team',sorted(final_df['bowling_team'].unique()))
city =st.selectbox('Select the city',sorted(final_df['city'].unique()))
over = st.number_input("Overs completed")
wicket =st.number_input("Wickets out")
target_runs = st.number_input("Enter the target runs")
score = st.number_input("Current Score")

if st.button("Predict Probability"):
    runs_left = target_runs - score
    balls_left = 120 - (over*6)
    wicket_left = 10 - wicket
    crr = score/over
    rr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'over':[over],'wicket':[wicket],'target_runs':[target_runs],'score':[score],'runs_left':[runs_left],'balls_left':[balls_left],'wicket_left':[wicket_left],'crr':[crr],'rr':[rr]})
    result = pipe.predict_proba(input_df)


    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")