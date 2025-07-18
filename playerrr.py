import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title('Player Performance Prediction App')

df = pd.read_csv('player.csv')
df.drop('Player', axis='columns', inplace=True)
df.drop('Team', axis='columns', inplace=True)
df.drop('Role', axis='columns', inplace=True)
df['Fitness'] = df['Fitness'].apply(lambda x: 1 if x == 'Fit' else 0)

x = df.drop('FantasyScore', axis='columns')
y = df['FantasyScore']

model = LinearRegression()
model.fit(x, y)

lr = st.number_input('Enter average number of runs in last 5 matches')
lw = st.number_input('Enter average number of wickets in last 5 matches')
ar = st.number_input('Enter average number of runs in last match')
aw = st.number_input('Enter average number of wickets in last match')
f = st.number_input('Enter fitness score (if fit then 1 else 0)')
rf = st.number_input('Enter recent form score')
r = st.number_input('Enter number of runs in the match played')
w = st.number_input('Enter number of wickets in the match played')

input_data = [[lr, lw, ar, aw, f, rf, r, w]]
prediction = model.predict(input_data)

if st.button('Predict'):
    st.write(f'The predicted fantasy score is {prediction[0]:.2f}')
    st.balloons()
   
