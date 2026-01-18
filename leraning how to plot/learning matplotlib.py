import matplotlib.pyplot as plt
import pandas as pd

#set the fig varible
data = pd.read_csv(r"C:\Users\user\Downloads\Capitec_Stock_with_Dividends.csv")



#pyplot is a collection of functions that make matplotlib function like matlab 
#each pyplot makes some cange to the figure object

#basic plot

#plt.plot([1,2,3,20],[1,2,4,25])
#plt.ylabel("values")
#plt.xlabel("words")
#plt.show()

Plot  = plt.figure()
#line,= plt.plot([data["Price"].min(),data["Price"].max()],[data["High"].min(),data["High"].max()])
Plot.add_subplot(2,2,1).scatter(x=data["Price"],y=data["Low"],c="red")
Plot.add_subplot(2,2,2).scatter(x=data["Price"],y=data["Low"],c="red")

            
plt.show()

