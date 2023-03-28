#%%

/**
* @author David Nilsson - Prime Fitness Studio AB
* www.primefitness.se
* 2023-03-28
*/

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from datetime import timedelta
from dateutil.parser import parse
import datetime
import statistics
from wordcloud import WordCloud, STOPWORDS
import sys
from os import path
from PIL import Image


#%%
# read the data file into a dataframe
df = pd.read_csv('openDamagesinTrains.csv')
print(df)

print(df.shape)

#%% 
"""
Drop the index
"""

#%%
"""
Extract data for a given vehicle
"""
grpByVehicle = df.groupby(['Vehicle'])    #grpByManu = df.groupby(['manufacturef'])
# Initializing a list df_[] to store the data frame into
vehicles_List = (['F26001', 'F26002', 'F26003', 'F26004', 'F26005', 'F26006', 'F26007', 'F26008', 'F26009', 'F26010', 'F26011', 'F26012'])

df_list = []
for i in range(12): 
    df_list.append(grpByVehicle.get_group(vehicles_List[i]))



#%%
# Initializing lists to store the data frames of each vechicles
# column-data of Damage reporting date and Damage closing date


# Appending the data from each column in the data frame of each vechicles
# column-data of Damage reporting date and Damage closing date to the lists
Damagereportingdate_list = []
Damageclosingdate_list = []
for i in df:
    Damagereportingdate_list.append(pd.to_datetime(df["Damage reporting date"]))
    Damageclosingdate_list.append(df["Damage closing date"])

"""
# Printing out the Damage reporting date
for i in Damagereportingdate_list:
    print("Print of Damagereportingdate_list[{}]: {}".format(i, Damagereportingdate_list[i]))
"""

for i in range(len(Damagereportingdate_list)):
    # Using the index to access the Pandas Series object from the list
    date = Damagereportingdate_list[i]
    print("Date: {}".format(date)) 
    
    

# Converting format and printing out the Damege closing date
for i in range(len(df_list)):
    mask = df_list[i]["Damage closing date"].notna()
    df_list[i].loc[mask, "Damage closing date"] = pd.to_datetime(df_list[i].loc[mask, "Damage closing date"], errors='coerce').dt.strftime('%Y-%m-%d')
    print("Print of Damage closing date[" + str(i) + "]: ", df_list[i]["Damage closing date"])



# Calculating and printing out the time to repair
for i in range(len(df_list)):
    mask = df_list[i]["Damage closing date"].notna() & df_list[i]["Damage reporting date"].notna()
    df_list[i].loc[mask, "Time to failure"] = pd.to_datetime(df_list[i].loc[mask, "Damage closing date"], errors="coerce") - \
    pd.to_datetime(df_list[i].loc[mask, "Damage reporting date"], errors="coerce")
    print("Day´s to repair[" + str(i) + "]: ", df_list[i]["Time to failure"])
  

# Initializing lists   
df_list_mean_list = []
df_list_median_list = [] 
df_list_std_list = []
  
# Calculating and printing out the mean, median and SD of time to repair for each vehicle
for i in range(len(df_list)):
    mask = df_list[i]["Time to failure"].notna()
    print("Mean of time to repair for vehicle " + str(i) + ": ", df_list[i].loc[mask, "Time to failure"].mean())
    df_list_mean_list.append((str(df_list[i].loc[mask, "Time to failure"].mean())))

    mask = df_list[i]["Time to failure"].notna()
    print("Median of time to repair for vehicle " + str(i) + ": ", df_list[i].loc[mask, "Time to failure"].median())
    df_list_median_list.append((str(df_list[i].loc[mask, "Time to failure"].median())))

    mask = df_list[i]["Time to failure"].notna()
    print("Standard deviation of time to repair for vehicle " + str(i) + ": ", np.std(df_list[i].loc[mask, "Time to failure"]))
    df_list_std_list.append((str(np.std(df_list[i].loc[mask, "Time to failure"]))))


# Storing the lists for plotting into new list variables
var_df_list_mean_list = df_list_mean_list
var_df_list_median_list = df_list_median_list
var_df_list_std_list = df_list_std_list


# Histogram of Time to failure for vehicles
# Creating separate plots of every vehicle
fig, axs = plt.subplots(nrows=len(df_list), ncols=1, figsize=(8, 6*len(df_list)))

# Iterate over each dataframe
for i, df in enumerate(df_list):
    # Loading the data for current vehicle
    mask = df["Time to failure"].notna()
    # converting timedelta64 to float64
    time_to_failure = df.loc[mask, "Time to failure"] / np.timedelta64(1, 'D')

    # Create histogram
    axs[i].hist(time_to_failure, bins=[0, 100, 200, 300, 400, 500])
    axs[i].set_title(f"Histogram of Time to failure for vehicle {i+1}")
    axs[i].set_xlabel("Time to failure (days)")
    axs[i].set_ylabel("Count")
    
    


# Plotting a histogram with mathplotlib
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

colors = ['blue']
ax0.hist(var_df_list_mean_list, bins=[0, 100, 200, 300, 400, 500], density=True, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('Mean time to repair')

ax1.hist(var_df_list_median_list, bins=[0, 100, 200, 300, 400, 500], density=True, histtype='bar', stacked=False)
ax1.set_title('Median time to repair')

ax2.hist(var_df_list_std_list, bins=[0, 100, 200, 300, 400, 500], histtype='bar', stacked=True, fill=True)
ax2.set_title('SD of time to repair')

#ax3.hist(time_to_failure, bins=[0, 250, 500, 750, 1000], histtype='bar')
#ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()




# Descriptive analysis with wordcloud

# Iterate over each dataframe and creating a wordcloud of field Damage area
for i, df in enumerate(df_list):
    # Loading the data for current vehicle
    mask = df["Damage area"].notna()
    # converting timedelta64 to float64
    text_to_wordcloud = str(df.loc[mask, "Damage area"])
    wordcloud = WordCloud(stopwords='stopwords.txt').generate(text_to_wordcloud)

# make figure to plot
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
# Removing axes
plt.axis("off")
# Printing out the picture of wordcloud
plt.show()


#%%
'''
Is there a relationship between Damagereportingdate and Damagereportingdate
'''
"""
# Scatter plot of manufacturer A
plt.scatter(Damagereportingdate_list, Damageclosingdate_list)
plt.title("Relation between Damagereportingdate and Damegeclosingdate - Vehicles")
plt.xlabel("Damagereportingdate")
plt.ylabel("Damageclosingdate")
plt.show()
"""

"""
# Converting time format to YYYY-MM-DD for column Date reporting date

df = pd.DataFrame({'Damage_reporting_date': {1:Damagereportingdate_list}})
print (df)


n=1000
ch = ['2011/02/22', '13/07/2022']
df = pd.DataFrame({"date": np.random.choice(ch,n)})
df["date"] = df["date"].str.replace("/","-").astype("M8[us]")
print(df)
"""

"""
df['Damage_reporting_date'] = pd.to_datetime(df.Damage_reporting_date)
print (df)


df['Damage_closing_date'] = pd.to_datetime(df.Damage_closing_date)
print (df)


df['Damage_reporting_date'] = df['Damage_reporting_date'].dt.strftime('%m/%d/%Y')
print (df)
"""




"""
df = pd.DataFrame({'Date': ['04/10/2017','2013-07-13']})
df['Clean_Date'] = df.Date.apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))
print(df)
"""


"""
n=1000
ch = ['2011/02/22', '13/07/2022']
df = pd.DataFrame({"date": np.random.choice(ch,n)})

df["date"] = df["date"].str.replace("/","-").astype("M8[us]")
print(df)
"""

"""
ch = ['2011/02/22', '13/07/2022']
df = pd.DataFrame({"date": np.random.choice(ch,n)})

df["date"] = df["date"].str.replace("/","-").astype("M8[us]")
print(df)
"""

"""
pd.interval_range(start=pd.Timestamp(Damagereportingdate_list[0]),
                  end=pd.Timestamp(Damagereportingdate_list[11]))
"""



"""

# Scatter plot of manufacturer B
plt.scatter(loadb, timeb)
plt.title("Relation between load and time - Manufacturer B")
plt.xlabel("Load")
plt.ylabel("Time")
plt.show()




# Scatter plot of manufacturer C
plt.scatter(loadc, timec)
plt.title("Relation between load and time - Manufacturer C")
plt.xlabel("Load")
plt.ylabel("Time")
plt.show()


#%%
'''
Characteristics of data
mean, median, mode
'''
# Computing statistical values of manufacturer A
meana = dfa['load'].mean()
print("Mean of load for manufacturer A: ", meana)

mediana = dfa['load'].median()
print("Median of load for manufacturer A: ", mediana)

modea = dfa['load'].mode()
print("Mode of load for manufacturer A: ", modea)





# Computing statistical values of manufacturer B
meanb = dfb['load'].mean()
print("Mean of load for manufacturer B: ", meanb)

medianb = dfb['load'].median()
print("Median of load for manufacturer B: ", medianb)

modeb = dfb['load'].mode()
print("Mode of load for manufacturer B: ", modeb)





# Computing statistical values of manufacturer C
meanc = dfc['load'].mean()
print("Mean of load for manufacturer C: ", meanc)

medianc = dfc['load'].median()
print("Median of load for manufacturer C: ", medianc)

modec = dfc['load'].mode()
print("Mode of load for manufacturer C: ", modec)





#%%
'''
How is load distributed
Why does it matter
uniform, normal, exponential, weibull
'''
# Histogram of manufacturer A
histogram_loada = dfa[['load']].plot(kind='hist', bins=10)
print("Histogram of load for manufacturer A: ", histogram_loada)

histogram_timea = dfa[['time']].plot(kind='hist', bins=10)
print("Histogram of time for manufacturer A: ", histogram_timea)




# Histogram of manufacturer B
histogram_loadb = dfb[['load']].plot(kind='hist', bins=10)
print("Histogram of load for manufacturer B: ", histogram_loadb)

histogram_timeb = dfb[['time']].plot(kind='hist', bins=10)
print("Histogram of time for manufacturer B: ", histogram_timeb)




# Histogram of manufacturer C
histogram_loadc = dfc[['load']].plot(kind='hist', bins=10)
print("Histogram of load for manufacturer C: ", histogram_loadc)

histogram_timec = dfc[['time']].plot(kind='hist', bins=10)
print("Histogram of time for manufacturer C: ", histogram_timec)

#%%
'''
variance, standard deviation
What is the meaning of 6sigma
'''
# standard deviation of load for manufacturer A
SDloada = dfa['load'].std()
print("Standard deviation of load of manufacturer A: ", SDloada)

# standard deviation of time for manufacturer A
SDtimea = dfa['time'].std()
print("Standard deviation of time of manufacturer A: ", SDtimea)




# standard deviation of load for manufacturer B
SDloadb = dfb['load'].std()
print("Standard deviation of load of manufacturer B: ", SDloadb)

# standard deviation of time for manufacturer B
SDtimeb = dfb['time'].std()
print("Standard deviation of time of manufacturer B: ", SDtimeb)




# standard deviation of load for manufacturer C
SDloadc = dfc['load'].std()
print("Standard deviation of load of manufacturer C: ", SDloadc)

# standard deviation of time for manufacturer C
SDtimec = dfc['time'].std()
print("Standard deviation of time of manufacturer C: ", SDtimec)
#%%
'''
Other plots that can be useful 
boxplot
'''

# Making a boxplot for loada for manufacturer A
fig, ax = plt.subplots()

# Printing the boxplot of loads
ax.boxplot([loada])

# Naming the axe of the plot
# ax.set_xticklabels(['Time A'])
ax.set_ylabel('Manufacturer A - Load A')

plt.show()




# Making a boxplot for time for manufacturer A
fig, ax = plt.subplots()

# Printing the boxplot of times
ax.boxplot([timea])

# Naming the axe of the plot
# ax.set_xticklabels(['Load A'])
ax.set_ylabel('Manufacturer A - Time A')

plt.show()




# Making a boxplot for loada for manufacturer B
fig, ax = plt.subplots()

# Printing the boxplot of loads
ax.boxplot([loadb])

# Naming the axe of the plot
# ax.set_xticklabels(['Time B'])
ax.set_ylabel('Manufacturer B - Load B')

plt.show()




# Making a boxplot for time for manufacturer B
fig, ax = plt.subplots()

# Printing the boxplot of times
ax.boxplot([timeb])

# Naming the axe of the plot
# ax.set_xticklabels(['Load B'])
ax.set_ylabel('Manufacturer B - Time B')

plt.show()




# Making a boxplot for loada for manufacturer C
fig, ax = plt.subplots()

# Printing the boxplot of loads
ax.boxplot([loadc])

# Naming the axe of the plot
# ax.set_xticklabels(['Time C'])
ax.set_ylabel('Manufacturer C - Load C')

plt.show()




# Making a boxplot for time for manufacturer C
fig, ax = plt.subplots()

# Printing the boxplot of times
ax.boxplot([timec])

# Naming the axe of the plot
# ax.set_xticklabels(['Load C'])
ax.set_ylabel('Manufacturer C - Time C')

plt.show()

"""