import streamlit as st

st.title('🥳 🔗 My Web App')  

with st.sidebar:
    text = st.text_input('Paste something here:')
    if text:
        st.success(text)
        st.balloons()

    # Retrieve your secrets via the st.secrets dict:
    #st.write("DB username:", st.secrets["db_username"])
    #st.write("DB password:", st.secrets["db_password"])
    #st.write("My cool secrets:", st.secrets["my_cool_secrets"]["things_i_like"])


with st.form('my_form'):
  
    text2 = st.text_area('Enter text prompt:', 'What is the meaning of life the universe and everything?')
    submitted = st.form_submit_button('Submit')

    if submitted :
        st.info("42")
        st.snow()

# streamlit run streamlit_app.py