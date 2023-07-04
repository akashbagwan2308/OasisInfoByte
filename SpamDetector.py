import pickle
import streamlit as st
import pandas as pd

data= pd.read_csv('mail_data.csv')
data.head(5)

data['Category']= data['Category'].map({'ham':0,'spam':1})

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x= data['Message']
y= data['Category']

x= cv.fit_transform(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train,y_train)
result = model.score(x_test,y_test)*100
f"Accuracy : {result.round(2)} %"

import pickle
pickle.dump(model,open("spam.pkl","wb"))
pickle.dump(model,open("vectorizer.pkl","wb"))
clf = pickle.load(open("spam.pkl",'rb'))
# clf


msg = "Hello"
data =[msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
print(result)

msg = "You win 10 dollors"
data =[msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
print(result)


# model = pickle.load(open("spam1.pkl",'rb'))
# cv = pickle.load(open('vectorizer1.pkl','rb'))
#
def main():
    st.title("Email Spam Classification Apps")
    st.subheader("Build with Streamlit & Python")
    msg = st.text_input("Enter a text :")
    if st.button('Predict'):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]

        if result == 1:
            st.error("This is a spam mail")
        else:
            st.success("This is a ham mail")

main()