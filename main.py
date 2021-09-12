import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set()

from PIL import Image
from helper import *
st.title('Dog Breed Classifier')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploaded',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except:
        return 0        

uploaded_file = st.file_uploader("Upload Image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        display_image = display_image.resize((500,300))
        st.image(display_image)
        prediction = predictor(os.path.join('uploaded',uploaded_file.name))
        print(prediction)
        os.remove('uploaded/'+uploaded_file.name)
        # drawing graphs
        st.text('Predictions :-')
        fig, ax = plt.subplots()
        ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)
        ax.set(xlabel='Confidence %', ylabel='Breed')

        st.pyplot(fig)
        



    #     indices = recommend()
    #     st.text('Recommended Products')
    #     os.remove('uploaded/'+uploaded_file.name)
    #     col1, col2, col3, col4,col5 = st.beta_columns(5)
    #     try:
    #         with col1:
    #             st.image('images/'+images_name[indices[0][1]])
    #     except:
    #         pass
    #     try:
    #         with col2:
    #             st.image('images/'+images_name[indices[0][2]])
    #     except:
    #         pass
    #     try:
    #         with col3:
    #             st.image('images/'+images_name[indices[0][3]])
    #     except:
    #         pass
    #     try:
    #         with col4:
    #             st.image('images/'+images_name[indices[0][4]])
    #     except:
    #         pass
    #     try:
    #         with col5:
    #             st.image('images/'+images_name[indices[0][5]])        
    #     except:
    #         pass                
    #     #file has uploaded
        
    # else:
    #     pass    
