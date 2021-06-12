from tkinter import*
from tkinter import ttk,messagebox
from PIL import Image,ImageTk # pip install pillow
import tkinter as tk

class Prediction:
    def __init__(self,root):
      
        inputValues = []

        age1 = ((int(self.txt_age.get()) - 29)  / (77-29 ))
        print(age1)
        trestbps1 = ((int(self.txt_rbp.get()) - 94)/(200-94))
        chol1 = ((int (self.txt_chol.get()) - 126)/(564-126))
        thalach1 = ((int(self.txt_thalach.get()) - 71)/(202-71))
        oldpeak1 = (float(self.txt_old_peak.get()))/(6.2)
        
        inputValues.append(age1)
        inputValues.append(self.txt_sex.get())
        inputValues.append(self.txt_cp.get())
        inputValues.append(trestbps1)
        inputValues.append(chol1)
        inputValues.append(self.txt_fbp.get())
        inputValues.append(self.txt_ecg.get())
        inputValues.append(thalach1)
        inputValues.append(trestbps1)
        inputValues.append(oldpeak1)
        inputValues.append(self.txt_slope.get())
        inputValues.append(self.txt_ca.get())
        inputValues.append(self.txt_thal.get()) 
        
        print(inputValues)


        print("\n") 
        final_Result = knn_classifier.predict([inputValues])
        print(final_Result)
        
        self.root=root
        self.root.title("Result")
        #self.root.geometry("1000x400+0+0")
        self.root.geometry("1530x710+0+0")
        self.root.config(bg="black")
        
        img=Image.open(r"C:\Users\AJ\Downloads\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\Files\image1.jpg")
        img=img.resize((1530,710),Image.ANTIALIAS)
        self.back_img=ImageTk.PhotoImage(img)
        back_bgimage=Label(self.root,image=self.back_img)
        back_bgimage.place(x=0,y=0,width=1350,height=710)
      
        if final_Result[0] == 1:
                l1=Label(back_bgimage,text="HEART DISEASE DETECTED",font=("times new roman",20,"bold"),bg="black",fg="white")
                l1.place(x=550,y=100)
                l2 =Label(back_bgimage, text="PLEASE VISIT NEAREST CARDIOLOGIST AT THE EARLIEST", font=('Impact', -20), fg='red')
                l2.place(x=550,y=200)
        else: 
                ll1=Label(back_bgimage,text="HEART DISEASE NOT DETECTED",font=("times new roman",20,"bold"),bg="black",fg="white")
                ll1.place(x=550,y=100)
                ll2 =Label(back_bgimage, text="DO NOT FORGOT TO EXERCISE DAILY.", font=('Impact', -20), fg='red')
                ll2.place(x=550,y=200)  
   
