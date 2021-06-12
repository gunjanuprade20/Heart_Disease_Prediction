from tkinter import*
from tkinter import ttk,messagebox
from PIL import Image,ImageTk # pip install pillow
import tkinter as tk
#import tkinter
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



def takeInput():
    inputValues = []

    age1 = ((float(txt_age.get()) - 29) / (77-29 ))
    print(age1)
    trestbps1 = ((int(txt_rbp.get()) - 94)/(200-94))
    chol1 = ((int (txt_chol.get()) - 126)/(564-126))
    thalach1 = ((int(txt_thalach.get()) - 71)/(202-71))
    oldpeak1 = (float(txt_old_peak.get())/(6.2))
        
    inputValues.append(age1)
    inputValues.append(txt_sex.get())
    inputValues.append(txt_cp.get())
    inputValues.append(trestbps1)
    inputValues.append(chol1)
    inputValues.append(txt_fbp.get())
    inputValues.append(txt_ecg.get())
    inputValues.append(thalach1)
    inputValues.append(trestbps1)
    inputValues.append(oldpeak1)
    inputValues.append(txt_slope.get())
    inputValues.append(txt_ca.get())
    inputValues.append(txt_thal.get()) 
        
    print(inputValues)


    print("\n") 
    final_Result = knn_classifier.predict([inputValues])
    print(final_Result)

    #substituteWindow = tk.Tk()
    substituteWindow = Toplevel(mainWindow)
    substituteWindow.title("RESULT PREDICTION")
    substituteWindow.geometry("1530x710+0+0")
    substituteWindow.config(bg="black")
    
    img=Image.open(r"C:\Users\admin\Desktop\Gunjan\Files\image1.jpg")
    img=img.resize((1530,710),Image.ANTIALIAS)
    back_img=ImageTk.PhotoImage(img)
    back_bgimage=Label(substituteWindow,image=back_img)
    back_bgimage.place(x=0,y=0,width=1350,height=710)
      
    if final_Result[0] == 1:
            l1=Label(back_bgimage,text="     HEART DISEASE DETECTED      ",font=("Arial",25,"bold"),bg="black",fg="Orange")
            l1.place(x=400,y=200)
            l2 =Label(back_bgimage, text="PLEASE VISIT NEAREST CARDIOLOGIST AT THE EARLIEST", font=('Impact', -20),fg='red')
            l2.place(x=450,y=300)
            l3 =Label(back_bgimage, text=".......THANK YOU.......", font=('Impact', -20), fg='red')
            l3.place(x=550,y=400)
    else: 
            ll1=Label(back_bgimage,text="    HEART DISEASE NOT DETECTED     ",font=("times new roman",25,"bold"),bg="black",fg="white")
            ll1.place(x=400,y=200)
            ll2 =Label(back_bgimage, text="   DO NOT FORGOT TO EXERCISE DAILY  ", font=('Impact', -20), fg='red')
            ll2.place(x=450,y=300)  
            ll3 =Label(back_bgimage, text=".......THANK YOU.......", font=('Impact', -20), fg='red')
            ll3.place(x=550,y=400)
    substituteWindow.mainloop()
        

heart = pd.read_csv(r"C:\Users\admin\Desktop\Gunjan\Files\Dataset.csv")
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

    
mainWindow = tk.Tk()
mainWindow.geometry("1530x710+0+0")
mainWindow.config(bg="black")

mainWindow.title("HEART DISEASE PREDICTION")
img=Image.open(r"C:\Users\admin\Desktop\Gunjan\Files\img2.png")
img=img.resize((1530,710),Image.ANTIALIAS)
user_img=ImageTk.PhotoImage(img)
user_bgimage=Label(mainWindow,image=user_img)
user_bgimage.place(x=0,y=0,width=1350,height=710)

face_recogn_title=Label(user_bgimage, text="HEART DISEASE PREDICTION MODEL", font=("times new Roman", 25, "bold"), bg="black", fg="Yellow")
face_recogn_title.place(x=0,y=40,width=1530,height=50)  

#frame for enter details:
title=Label(user_bgimage,text="Check Your Heart!!!",font=("times new roman",20,"bold"),bg="black",fg="white")
title.place(x=550,y=100)

        
age=Label(user_bgimage,text=" Enter age(yrs) ",font=("times new roman",15,"bold"),bg="black",fg="white")
age.place(x=50,y=200)
txt_age=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_age.place(x=260,y=200,width=150,height=30)
        
rbp=Label(user_bgimage,text=" Enter RBP(94-200) ",font=("times new roman",15,"bold"),bg="black",fg="white")
rbp.place(x=50,y=250)
txt_rbp=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_rbp.place(x=260,y=250,width=150,height=30)
    
       
ecg=Label(user_bgimage,text=" Enter ECG(0,1,2) ",font=("times new roman",15,"bold"),bg="black",fg="white")
ecg.place(x=50,y=300)
txt_ecg=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_ecg.place(x=260,y=300,width=150,height=30)

old_peak=Label(user_bgimage,text="Enter Old Peak(0-6.2)",font=("times new roman",15,"bold"),bg="black",fg="white")
old_peak.place(x=50,y=350)
txt_old_peak=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_old_peak.place(x=260,y=350,width=150,height=30)
     
sex=Label(user_bgimage,text=" Enter Sex ",font=("times new roman",15,"bold"),bg="black",fg="white")
sex.place(x=500,y=200)
txt_sex=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_sex.place(x=710,y=200,width=150,height=30)
       
chol=Label(user_bgimage,text=" Enter Serum Chol ",font=("times new roman",15,"bold"),bg="black",fg="white")
chol.place(x=500,y=250)
txt_chol=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_chol.place(x=710,y=250,width=150,height=30)
      
thalach=Label(user_bgimage,text=" Enter thalach(71-202) ",font=("times new roman",15,"bold"),bg="black",fg="white")
thalach.place(x=500,y=300)
txt_thalach=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_thalach.place(x=710,y=300,width=150,height=30)

slope=Label(user_bgimage,text="Enter slope(0,1,2)",font=("times new roman",15,"bold"),bg="black",fg="white")
slope.place(x=500,y=350)
txt_slope=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_slope.place(x=710,y=350,width=150,height=30)
     
thal=Label(user_bgimage,text="Enter thal(0,1,2,3)",font=("times new roman",15,"bold"),bg="black",fg="white")
thal.place(x=500,y=400)
txt_thal=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_thal.place(x=710,y=400,width=150,height=30)
        
cp=Label(user_bgimage,text=" Enter CP(0-4) ",font=("times new roman",15,"bold"),bg="black",fg="white")
cp.place(x=960,y=200)
txt_cp=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_cp.place(x=1180,y=200,width=150,height=30)

fbp=Label(user_bgimage,text=" Enter Fasting BP(0-4) ",font=("times new roman",15,"bold"),bg="black",fg="white")
fbp.place(x=960,y=250)
txt_fbp=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_fbp.place(x=1180,y=250,width=150,height=30)
        
exang=Label(user_bgimage,text=" Enter exAngina(0/1) ",font=("times new roman",15,"bold"),bg="black",fg="white")
exang.place(x=960,y=300)
txt_exang=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_exang.place(x=1180,y=300,width=150,height=30)

ca=Label(user_bgimage,text=" Enter C.A(0-3) ",font=("times new roman",15,"bold"),bg="black",fg="white")
ca.place(x=960,y=350)
txt_ca=Entry(user_bgimage,font=("times new  roman",15),bg="black",fg="white")
txt_ca.place(x=1180,y=350,width=150,height=30)
        
img1=Image.open(r"C:\Users\admin\Desktop\Gunjan\Files\click_me1jpg.jpg")
img1=img1.resize((250,60),Image.ANTIALIAS)
btn_img=ImageTk.PhotoImage(img1)
#img_label= Label(image=self.btn_img)
btn_image=Button(user_bgimage,image=btn_img,cursor="hand2",command=takeInput,borderwidth=2)
btn_image.place(x=530, y=500)


mainWindow.mainloop()


