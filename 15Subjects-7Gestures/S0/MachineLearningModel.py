# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 02:23:04 2022

@author: Mohamed Essam
"""
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#import os 

#create master file
master_df = pd.DataFrame()

#read files
data = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-tap-S0.csv")
data2 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-twodwn-S0.csv")
data3 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-opendwn-S0.csv")
data4 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-neut-S0.csv")
data5 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-fistout-S0.csv")
data6 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-fistdwn-S0.csv")
data7 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-twout-S0.csv")
data8 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-right-S0.csv")
data9 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-openout-S0.csv")
data10 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-left-S0.csv")
data11 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-tap-S1.csv")
data12 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-twodwn-S1.csv")
data13 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-opendwn-S1.csv")
data14 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-neut-S1.csv")
#data15 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-fistout-S1.csv")
data16 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-fistdwn-S1.csv")
data17 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-twout-S1.csv")
data18 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-right-S1.csv")
#data19 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-openout-S1.csv")
data20 = pd.read_csv("C:/Users/Mohamed Essam/Downloads/15Subjects-7Gestures/15Subjects-7Gestures/S0/emg-left-S1.csv")

#add columns
data["movement"] = "TAP"
data2["movement"] = "victory"
data3["movement"] = "open"
data4["movement"] = "neutral"
data5["movement"] = "closed"
data6["movement"] = "closed"
data7["movement"] = "victory"
data8["movement"] = "extension"
data9["movement"] = "open"
data10["movement"] = "flexion"
data11["movement"] = "TAP"
data12["movement"] = "victory"
data13["movement"] = "open"
data14["movement"] = "neutral"
#data15["movement"] = "closed"
data16["movement"] = "closed"
data17["movement"] = "victory"
data18["movement"] = "extension"
#data19["movement"] = "open"
data20["movement"] = "flexion"

#remove columns
data = data.drop(["timestamp"], axis=1)
data2 = data2.drop(["timestamp"], axis=1)
data3 = data3.drop(["timestamp"], axis=1)
data4 = data4.drop(["timestamp"], axis=1)
data5 = data5.drop(["timestamp"], axis=1)
data6 = data6.drop(["timestamp"], axis=1)
data7 = data7.drop(["timestamp"], axis=1)
data8 = data8.drop(["timestamp"], axis=1)
data9 = data9.drop(["timestamp"], axis=1)
data10 = data10.drop(["timestamp"], axis=1)
data11 = data11.drop(["timestamp"], axis=1)
data12 = data12.drop(["timestamp"], axis=1)
data13 = data13.drop(["timestamp"], axis=1)
data14 = data14.drop(["timestamp"], axis=1)
#data15 = data5.drop(["timestamp"], axis=1)
data16 = data16.drop(["timestamp"], axis=1)
data17 = data17.drop(["timestamp"], axis=1)
data18 = data18.drop(["timestamp"], axis=1)
#data19 = data9.drop(["timestamp"], axis=1)
data20 = data20.drop(["timestamp"], axis=1)


#drop null readings
data = data.dropna()
data2 = data2.dropna()
data3 = data3.dropna()
data4 = data4.dropna()
data5 = data5.dropna()
data6 = data6.dropna()
data7 = data7.dropna()
data8 = data8.dropna()
data9 = data9.dropna()
data10 = data10.dropna()
data11 = data.dropna()
data12 = data2.dropna()
data13 = data3.dropna()
data14 = data4.dropna()
#data15 = data5.dropna()
data16 = data6.dropna()
data17 = data7.dropna()
data18 = data8.dropna()
#data19 = data9.dropna()
data20 = data10.dropna()

num_data= data.select_dtypes(exclude="object")
obj_data= data.select_dtypes(include="object")
num_data2= data2.select_dtypes(exclude="object")
obj_data2= data2.select_dtypes(include="object")
num_data3= data3.select_dtypes(exclude="object")
obj_data3= data3.select_dtypes(include="object")
num_data4= data4.select_dtypes(exclude="object")
obj_data4= data4.select_dtypes(include="object")
num_data5= data5.select_dtypes(exclude="object")
obj_data5= data5.select_dtypes(include="object")
num_data6= data6.select_dtypes(exclude="object")
obj_data6= data6.select_dtypes(include="object")
num_data7= data7.select_dtypes(exclude="object")
obj_data7= data7.select_dtypes(include="object")
num_data8= data8.select_dtypes(exclude="object")
obj_data8= data8.select_dtypes(include="object")
num_data9= data9.select_dtypes(exclude="object")
obj_data9= data9.select_dtypes(include="object")
num_data10= data10.select_dtypes(exclude="object")
obj_data10= data10.select_dtypes(include="object")

num_data11= data11.select_dtypes(exclude="object")
obj_data11= data11.select_dtypes(include="object")
num_data12= data12.select_dtypes(exclude="object")
obj_data12= data12.select_dtypes(include="object")
num_data13= data13.select_dtypes(exclude="object")
obj_data13= data13.select_dtypes(include="object")
num_data14= data14.select_dtypes(exclude="object")
obj_data14= data14.select_dtypes(include="object")
# num_data15= data15.select_dtypes(exclude="object")
# obj_data15= data15.select_dtypes(include="object")
num_data16= data16.select_dtypes(exclude="object")
obj_data16= data16.select_dtypes(include="object")
num_data17= data17.select_dtypes(exclude="object")
obj_data17= data17.select_dtypes(include="object")
num_data18= data8.select_dtypes(exclude="object")
obj_data18= data18.select_dtypes(include="object")
# num_data9= data9.select_dtypes(exclude="object")
# obj_data9= data9.select_dtypes(include="object")
num_data20= data20.select_dtypes(exclude="object")
obj_data20= data20.select_dtypes(include="object")

# enc = LabelEncoder()
# for i in range (0,obj_data.shape[1]):
#       obj_data.iloc[:,i] = enc.fit_transform(obj_data.iloc[:,i])
# for i in range (0,obj_data2.shape[1]):
#       obj_data2.iloc[:,i] = enc.fit_transform(obj_data2.iloc[:,i])
# for i in range (0,obj_data3.shape[1]):
#       obj_data3.iloc[:,i] = enc.fit_transform(obj_data3.iloc[:,i])
# for i in range (0,obj_data4.shape[1]):
#       obj_data4.iloc[:,i] = enc.fit_transform(obj_data4.iloc[:,i])
# for i in range (0,obj_data5.shape[1]):
#       obj_data5.iloc[:,i] = enc.fit_transform(obj_data5.iloc[:,i])
# for i in range (0,obj_data6.shape[1]):
#       obj_data6.iloc[:,i] = enc.fit_transform(obj_data6.iloc[:,i])
# for i in range (0,obj_data7.shape[1]):
#       obj_data7.iloc[:,i] = enc.fit_transform(obj_data7.iloc[:,i])
# for i in range (0,obj_data8.shape[1]):
#       obj_data8.iloc[:,i] = enc.fit_transform(obj_data8.iloc[:,i])
# for i in range (0,obj_data9.shape[1]):
#       obj_data9.iloc[:,i] = enc.fit_transform(obj_data9.iloc[:,i])
# for i in range (0,obj_data10.shape[1]):
#       obj_data10.iloc[:,i] = enc.fit_transform(obj_data10.iloc[:,i])
     
new_data =  pd.concat([num_data,obj_data], axis=1)
new_data2 =  pd.concat([num_data2,obj_data2], axis=1)
new_data3 =  pd.concat([num_data3,obj_data3], axis=1)
new_data4 =  pd.concat([num_data4,obj_data4], axis=1)
new_data5 =  pd.concat([num_data5,obj_data5], axis=1)
new_data6 =  pd.concat([num_data6,obj_data6], axis=1)
new_data7 =  pd.concat([num_data7,obj_data7], axis=1)
new_data8 =  pd.concat([num_data8,obj_data8], axis=1)
new_data9 =  pd.concat([num_data9,obj_data9], axis=1)
new_data10 =  pd.concat([num_data10,obj_data10], axis=1)
new_data11 =  pd.concat([num_data11,obj_data11], axis=1)
new_data12 =  pd.concat([num_data12,obj_data12], axis=1)
new_data13 =  pd.concat([num_data13,obj_data13], axis=1)
new_data14 =  pd.concat([num_data14,obj_data14], axis=1)
# new_data15 =  pd.concat([num_data15,obj_data15], axis=1)
new_data16 =  pd.concat([num_data16,obj_data16], axis=1)
new_data17 =  pd.concat([num_data17,obj_data17], axis=1)
new_data18 =  pd.concat([num_data18,obj_data18], axis=1)
# new_data9 =  pd.concat([num_data9,obj_data9], axis=1)
new_data20 =  pd.concat([num_data20,obj_data20], axis=1)

master_df = pd.concat([new_data,new_data2,new_data3,new_data4,new_data5,new_data6,new_data7,new_data8,new_data9,new_data10,new_data11,new_data12,new_data13,new_data14,new_data16,new_data17,new_data18,new_data20],axis=0)
x = master_df.drop(["movement"], axis = 1)
y = master_df["movement"]


x_train  , x_test , y_train , y_test = train_test_split(x,y,train_size=0.99)
model= RandomForestClassifier(n_estimators=1000,max_depth=(40))
#model = MLPClassifier()
model.fit(x_train , y_train)
print (model.predict([[-8	,3	,-4	,0	,6	,-2	,-1	,-1]]))


print (model.score(x_train, y_train))
print (model.score (x_test, y_test))

plt.plot(x,y)
