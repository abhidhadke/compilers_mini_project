#EDA packages
import pandas as pd
import numpy as np
import seaborn as sns
import neattext.functions as nfx

from sklearn.linear_model import LogisticRegression


#transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'emotion_dataset_2.csv')

df.head()

df['Emotion'].value_counts()

sns.countplot(x='Emotion', data=df)

#user handles
df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

#stopwords
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)

df['Clean_Text'] = df['Text'].apply(nfx.remove_special_characters)
df

#features and lables
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

x_train, x_test, y_train, y_test = train_test_split(Xfeatures,ylabels,test_size = 0.3, random_state=42)


from sklearn.pipeline import Pipeline

pipe_lr = Pipeline(steps=[('cv', CountVectorizer()),('lr', LogisticRegression(solver='lbfgs', max_iter=3000))])

pipe_lr.fit(x_train, y_train)

#check accuracy
pipe_lr.score(x_test, y_test)
#print(x_test)

#prediction
ex1 = "This is a sad sentence"
pipe_lr.predict([ex1])

pipe_lr.classes_

def analyze_text():
    output_text = ""
    text_input = input_path_entry.get()
    output = pipe_lr.predict([text_input])
    output_text = f"Text Emotion: {output}"
    output_label.config(text=output_text)
    root.update_idletasks()


import tkinter as tk
from tkinter import filedialog
from tkinter import font

root = tk.Tk()
root.title("Emotional Text Analysis")
root.geometry("800x600")
custom_font = font.Font(family="Helvetica", size=25)

input_label = tk.Label(root, text="Enter the text: ",font=custom_font)
input_label.pack()
input_path_entry = tk.Entry(root,font=custom_font)
input_path_entry.pack()

analyze_button = tk.Button(root, text="Analyse", command=analyze_text,font=custom_font)
analyze_button.pack()

output_label = tk.Label(root, text="")
output_label.pack()

root.mainloop()