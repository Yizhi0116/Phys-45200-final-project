import streamlit as st

pg = st.navigation([
    st.Page("pages/Final_5_19_2.py",   title="OAT / TAT Squeezing",  icon="🌀"),
    st.Page("pages/dipolar.py",   title="Dipolar XY Spin Squeezing", icon="🧲"),
])

pg.run()
