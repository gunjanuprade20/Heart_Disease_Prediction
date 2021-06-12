from tkinter import*
from tkinter import ttk,messagebox
from PIL import Image,ImageTk # pip install pillow
import tkinter as tk
from Prediction import Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow

# %matplotlib inline

# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
def main():
    main_window=Tk()
    app=MainWindow(main_window)
    main_window.mainloop()

class MainWindow:
    def __init__(self,root):

        heart = pd.read_csv(r"C:\Users\AJ\Downloads\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\Files\Dataset.csv")
        # we have unknown values '?'
        # change unrecognized value '?' into mean value through the column
        min_max = MinMaxScaler()
        columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        heart[columns_to_scale ] = min_max.fit_transform(heart[columns_to_scale])
        y = heart['target']
        X = heart.drop(['target'], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
        # heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


        # print(len(X_train))
        # len(X_test)
        knn_classifier = KNeighborsClassifier(n_neighbors = 4)
        knn_classifier.fit(X_train, y_train)

        self.root=root
        self.root.title("Main Window")
        #self.root.geometry("1000x400+0+0")
        self.root.geometry("1530x710+0+0")
        self.root.config(bg="black")

        #bg image
        img=Image.open(r"C:\Users\AJ\Downloads\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\Files\img2.png")
        img=img.resize((1530,710),Image.ANTIALIAS)
        self.user_img=ImageTk.PhotoImage(img)
        user_bgimage=Label(self.root,image=self.user_img)
        user_bgimage.place(x=0,y=0,width=1350,height=710)
        face_recogn_title=Label(user_bgimage, text="HEART DISEASE PREDICTION MODEL", font=("times new Roman", 25, "bold"), bg="black", fg="Yellow")
        face_recogn_title.place(x=0,y=40,width=1530,height=50)  

        #frame for enter details:
        title=Label(user_bgimage,text="Check Your Heart!!!",font=("times new roman",20,"bold"),bg="black",fg="white")
        title.place(x=550,y=100)

        # -----------Row1
        age=Label(user_bgimage,text=" Enter age(yrs) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        age.place(x=50,y=200)
        self.txt_age=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_age.place(x=260,y=200,width=150,height=30)
        
        rbp=Label(user_bgimage,text=" Enter RBP(94-200) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        rbp.place(x=50,y=250)
        self.txt_rbp=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_rbp.place(x=260,y=250,width=150,height=30)
    
       
        ecg=Label(user_bgimage,text=" Enter ECG(0,1,2) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        ecg.place(x=50,y=300)
        self.txt_ecg=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_ecg.place(x=260,y=300,width=150,height=30)

        old_peak=Label(user_bgimage,text="Enter Old Peak(0-6.2)",font=("times new roman",15,"bold"),bg="black",fg="white")
        old_peak.place(x=50,y=350)
        self.txt_old_peak=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_old_peak.place(x=260,y=350,width=150,height=30)
     
        sex=Label(user_bgimage,text=" Enter Sex ",font=("times new roman",15,"bold"),bg="black",fg="white")
        sex.place(x=500,y=200)
        self.txt_sex=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_sex.place(x=710,y=200,width=150,height=30)
       
        chol=Label(user_bgimage,text=" Enter Serum Chol ",font=("times new roman",15,"bold"),bg="black",fg="white")
        chol.place(x=500,y=250)
        self.txt_chol=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_chol.place(x=710,y=250,width=150,height=30)
      
        thalach=Label(user_bgimage,text=" Enter thalach(71-202) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        thalach.place(x=500,y=300)
        self.txt_thalach=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_thalach.place(x=710,y=300,width=150,height=30)

        slope=Label(user_bgimage,text="Enter slope(0,1,2)",font=("times new roman",15,"bold"),bg="black",fg="white")
        slope.place(x=500,y=350)
        self.txt_slope=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_slope.place(x=710,y=350,width=150,height=30)
     
        thal=Label(user_bgimage,text="Enter thal(0,1,2,3)",font=("times new roman",15,"bold"),bg="black",fg="white")
        thal.place(x=500,y=400)
        self.txt_thal=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_thal.place(x=710,y=400,width=150,height=30)
        
        cp=Label(user_bgimage,text=" Enter CP(0-4) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        cp.place(x=960,y=200)
        self.txt_cp=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_cp.place(x=1180,y=200,width=150,height=30)

        fbp=Label(user_bgimage,text=" Enter Fasting BP(0-4) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        fbp.place(x=960,y=250)
        self.txt_fbp=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_fbp.place(x=1180,y=250,width=150,height=30)
        
        exang=Label(user_bgimage,text=" Enter exAngina(0/1) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        exang.place(x=960,y=300)
        self.txt_exang=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_exang.place(x=1180,y=300,width=150,height=30)

        ca=Label(user_bgimage,text=" Enter C.A(0-3) ",font=("times new roman",15,"bold"),bg="black",fg="white")
        ca.place(x=960,y=350)
        self.txt_ca=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
        self.txt_ca.place(x=1180,y=350,width=150,height=30)
        
        img1=Image.open(r"C:\Users\AJ\Downloads\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\Files\click_me1jpg.jpg")
        img1=img1.resize((250,60),Image.ANTIALIAS)
        self.btn_img=ImageTk.PhotoImage(img1)
        #img_label= Label(image=self.btn_img)
        btn_image=Button(user_bgimage,image=self.btn_img,cursor="hand2",command=self.prediction_window,borderwidth=2)
        btn_image.place(x=530, y=500)
        
        #click_btn= PhotoImage(file=r"C:\Users\AJ\Downloads\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\Files\click_me1jpg.jpg")

        #Let us create a label for button event
        #img_label= Label(image=click_btn)

        #Let us create a dummy button and pass the image
        #button= Button(user_bgimage, image=click_btn,command=self.prediction_window,borderwidth=0)
        #button.place(x=500, y=500)

    def prediction_window(self):
        #self.root.destroy()
        new_win=Toplevel(self.root)
        app=Prediction(new_win)
main()