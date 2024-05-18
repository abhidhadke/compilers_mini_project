import tkinter as tk
from tkinter import ttk, filedialog, font
from tkinter.scrolledtext import ScrolledText
from textblob import TextBlob
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import neattext as nfx

# Load your dataset and perform necessary data preprocessing
df = pd.read_csv(r'emotion_dataset_2.csv')
df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords).apply(nfx.remove_special_characters).apply(nfx.remove_userhandles)
Xfeatures = df['Clean_Text']
ylabels = df['Emotion']
x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.3, random_state=42)

# Create and train your model pipeline
pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression(solver='lbfgs', max_iter=3000))])
pipe_lr.fit(x_train, y_train)
print("Accuracy of model: ",pipe_lr.score(x_test,y_test))

# Function to perform sentiment analysis using the trained model
def analyze_text():
    text = input_text.get("1.0", tk.END)
    output = pipe_lr.predict([text])
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Text Emotion: {output}")
    output_text.config(state=tk.DISABLED)

# Function to load text from a file
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            text = file.read()
            input_text.delete("1.0", tk.END)
            input_text.insert(tk.END, text)

# GUI setup
root = tk.Tk()
root.title("Emotional Text Analysis")
root.geometry("800x600")

# Define custom fonts and styles
heading_font = font.Font(family="Helvetica", size=30, weight="bold")
button_font = font.Font(family="Helvetica", size=20)
output_font = font.Font(family="Helvetica", size=16)

style = ttk.Style()
style.theme_use("clam")  # You can change the theme (e.g., 'clam', 'default', 'alt')

# Configure colors
style.configure('TLabel', background='#e6e6e6', foreground='black')
style.configure('TButton', background='#3399ff', foreground='white', font=button_font)  # Apply font to the button
style.configure('TEntry', background='white', foreground='black')
style.configure('TText', background='white', foreground='black')

# Create input text area
input_label = ttk.Label(root, text="Enter the text:", font=heading_font)
input_label.pack()
input_text = ScrolledText(root, font=heading_font, height=8, wrap=tk.WORD)
input_text.pack()

# Create Load File button
load_button = ttk.Button(root, text="Load File", command=load_file)
load_button.pack()

# Create Analyze button
analyze_button = ttk.Button(root, text="Analyze", command=analyze_text)
analyze_button.pack()

# Create output text area
output_label = ttk.Label(root, text="Analysis Result:", font=heading_font)
output_label.pack()
output_text = ScrolledText(root, font=output_font, height=4, state=tk.DISABLED, wrap=tk.WORD)
output_text.pack()

root.mainloop()

