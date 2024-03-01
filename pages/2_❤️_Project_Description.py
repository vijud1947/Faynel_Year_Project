import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt

st.title('Project Details')
# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Approach')
st.write('We used _supervised learning_ to classify phishing and legitimate websites. '
         'We benefit from content-based approach and focus on html of the websites. '
         'Also, We used scikit-learn for the ML models.'
         )
st.write('For this educational project, '
         'We created my own data set and defined features, some from the literature and some based on manual analysis. '
         'We used requests library to collect data, BeautifulSoup module to parse and extract features. ')
st.write('The source code and data sets are available in the below Github link:')
st.write('_https://github.com/AdarshVajpayee19/Phishing-Website-Detection-ML_')
# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Data set')
st.write('We used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')

# ----- FOR THE PIE CHART ----- #
labels = 'phishing', 'legitimate'
phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
legitimate_rate = 100 - phishing_rate
sizes = [phishing_rate, legitimate_rate]
explode = (0.1, 0)
fig, ax = plt.subplots()
ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
ax.axis('equal')
st.pyplot(fig)
# ----- !!!!! ----- #
# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)
st.write('Features + URL + Label ==> Dataframe')
st.markdown('label is 1 for phishing, 0 for legitimate')
number = st.slider("Select row number to display", 0, 100)
st.dataframe(ml.legitimate_df.head(number))


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(ml.df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='phishing_legitimate_structured_data.csv',
    mime='text/csv',
)
# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader('Features')
st.write('We used only content-based features.We didn\'t use url-based faetures like length of url etc.'
         'Most of the features extracted using find_all() method of BeautifulSoup module after parsing html.')

st.subheader('Results')
st.write('We used 7 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
         'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
         'Comparison table is below:')
st.table(ml.df_results)
# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)
st.write('NB --> Gaussian Naive Bayes')
st.write('SVM --> Support Vector Machine')
st.write('DT --> Decision Tree')
st.write('RF --> Random Forest')
st.write('AB --> AdaBoost')
st.write('NN --> Neural Network')
st.write('KN --> K-Neighbours')
# Add a horizontal line
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Phishing Demo.")
# st.image("static\phising.gif", use_column_width=True)
st.markdown(
    '<img src="./app/static/phising.gif">',
    unsafe_allow_html=True,
)
st.markdown("<hr>", unsafe_allow_html=True)

