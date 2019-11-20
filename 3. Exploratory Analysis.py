

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd




#EXPLORATORY ANALYSIS

#Initial visualisation:

#First and last 5 rows:
data.head(5)
data.tail(5)

#info about datasets:
data.info(memory_usage='deep')
data.describe()


#First Plot:
plt.plot(data.energy_consumption, color='blue', linewidth=0.2)
plt.ylabel('Energy Consumption')
plt.xlabel('Date')
plt.title('Energy consumption 2011 - 2014')
plt.grid()
plt.show()



daily= data.resample('D').sum()
weekly= data.resample('W').sum()
monthly= data.resample('M').sum()
quarterly= data.resample('Q').sum()




# data for 3 days - every half an hour
small_df= data.loc['2012-02-06 00:00:00':'2012-02-12 23:30:00',:]
myFmt = DateFormatter('%Y-%m-%d  %H:%M:%S') 
# plot:
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 3 days worth of data (every 30 min)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()

#daily data
myFmt = DateFormatter('%Y-%m-%d') 
#plot:
fig, ax = plt.subplots()
ax.plot(daily.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - daily data")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()

#monthly data
myFmt = DateFormatter('%Y-%m-%d') 
# plot:
fig, ax = plt.subplots()
ax.plot(monthly.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.xaxis.set_major_locator(mdates.DayLocator (interval=30)) 
ax.set(title="Energy consumption - monthly data")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()

#quarterly data
myFmt = DateFormatter('%Y-%m-%d') 
# plot:
fig, ax = plt.subplots()
ax.plot(quarterly.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - quarterly data")
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()
ax.grid()


#TRENDS: 
myFmt = DateFormatter('%Y-%m-%d  %H:%M') 
data_mean = small_df['energy_consumption'].rolling(48).mean()
#plot: 
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.plot(data_mean, color='red', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 3 days worth of data (every 30 min)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()


#DIFFERENCING DATA:
data_mean = small_df.loc[:,'energy_consumption'].rolling(48).mean()
std= small_df.loc[:,'energy_consumption'].rolling(48).std()

# Before differencing:
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 , label='Energy consumption')
ax.plot(data_mean, color='black', marker='o', linewidth=1, markersize=2, label='Mean' )
ax.plot(std, color='purple', marker='o', linewidth=1, markersize=2, label='Std' )
ax.legend(loc='upper left')
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 7 days worth of data (frequency= 30 min)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()


# After differencing:
diff=pd.DataFrame(small_df['energy_consumption'].diff())
diff= diff[1:-1]
diff_data_mean = diff.rolling(48).mean()
diff_std= diff.rolling(48).std()
#
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2, label='Energy consumption' )
ax.plot(diff, color='green', marker='o', linewidth=1, markersize=2, label='Differenced energy consumption' )
ax.plot(diff_data_mean, color='black', marker='o', linewidth=1, markersize=2, label='Mean' )
ax.plot(diff_std, color='purple', marker='o', linewidth=1, markersize=2, label='Std'  )
ax.legend(loc='upper left')
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 7 days worth of data (frequency= 30 min)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()   


#ACF AND PACF PLOTS BEFEORE DIFFERENCING: 
plot_acf(small_df.loc[:,'energy_consumption'])
plt.show()

plot_pacf(small_df.loc[:,'energy_consumption'], lags=48)
plt.show()    

#ACF AND PACF PLOTS AFTER DIFFERENCING: 
plot_acf(diff.loc[:, 'energy_consumption'])
plt.show()

plot_pacf(diff, lags=48)
plt.show()    


#ADF TEST: 
print('')
dfuller_tets = adfuller(diff.energy_consumption)
print('Augmented Dickey-Fuller test')
print('Test Statistic              ' + str(dfuller_tets[0]))
print('p-value                      ' + str(dfuller_tets[1]))
print('#Lags Used                   ' + str(dfuller_tets[2]))
print('Number of Observations Used  ' + str(dfuller_tets[3]))
print('Critical Value 1%           ' + str(dfuller_tets[4]['1%']))
print('Critical Value 5%           ' + str(dfuller_tets[4]['5%']))
print('Critical Value 10%          ' + str(dfuller_tets[4]['10%']))



#TS DECOMPOSITION:
sub_df= pd.DataFrame(new_data.iloc[0:500,:], copy=True)

decomposition = seasonal_decompose(sub_df,
                                   freq=48,
                                   model='multiplicative')
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

decomposition.plot()
  











############       DATASET 2          ############

#data= data.iloc[:,3:]
#data= data.iloc[:, [0,1,2,3,7]]
data.head(5)
data.tail(5)

#info about datasets:
data.info(memory_usage='deep')
data.describe()


#First Plot:
plt.plot(data.energy_consumption, color='blue', linewidth=0.2)
plt.ylabel('Energy Consumption')
plt.xlabel('Date')
plt.title('Energy consumption 2016-01-12 to 2016-05-27')
plt.grid()
plt.show()

#DATA AGGREGATION:
daily= data.resample('D').sum()
weekly= data.resample('W').sum()
monthly= data.resample('M').sum()
quarterly= data.resample('Q').sum()


# data for 3 days - every half an hour
small_df= data.loc['2016-02-02 00:00:00':'2016-02-04 23:59:00',:]
myFmt = DateFormatter('%Y-%m-%d  %H:%M:%S') 
#plot:
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=0.5, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 3 days worth of data")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()

#daily data
myFmt = DateFormatter('%Y-%m-%d') 
# plot the data
fig, ax = plt.subplots()
ax.plot(daily.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - daily data")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()

#monthly data
myFmt = DateFormatter('%Y-%m-%d') 
# plot the data
fig, ax = plt.subplots()
ax.plot(monthly.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.xaxis.set_major_locator(mdates.DayLocator (interval=28)) 
ax.set(title="Energy consumption - monthly data")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()

#quarterly data
myFmt = DateFormatter('%Y-%m-%d') 
# plot the data
fig, ax = plt.subplots()
ax.plot(quarterly.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
#ax.xaxis.set_major_locator(mdates.DayLocator (interval=d_interval)) 
ax.set(title="Energy consumption - quarterly data")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()




#TRENDS: 
myFmt = DateFormatter('%Y-%m-%d  %H:%M') 
data_mean = small_df['energy_consumption'].rolling(48).mean()
# plot the data
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 )
ax.plot(data_mean, color='red', marker='o', linewidth=1, markersize=2 )
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 3 days worth of data (every 10 min)")
ax.xaxis.set_major_formatter(myFmt)
fig.autofmt_xdate()
fig.set_size_inches(16.5, 4.2)
ax.grid()





#Variation of Energy consumption in comparison to temperature: 

myFmt = DateFormatter('%Y-%m-%d  %H:%M') 
small_df= data.loc['2016-02-02 00:00:00':'2016-02-08 23:59:00',:]

# plot the data
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2, label='Energy_Consumption')
ax.set(xlabel="Date", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 7 days of data (Frequency=10 minutes)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()
plt.legend()
ax_two = ax.twinx()  #second  Y axis

ax_two.plot(small_df.iloc[:,0], color='red', marker='o', linewidth=1, markersize=2, label='Indoors_temp' )
ax_two.plot(small_df.iloc[:,2], color='green', marker='o', linewidth=1, markersize=2, label='Outside_temp' )
ax_two.tick_params(axis='y')
ax_two.set_ylabel('Tempreature (Celsius)')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.legend()
plt.show()


#Heatmap:
correlation=data.corr()#correlation matrix
fig = plt.figure()
ax1 = fig.add_subplot(111)
colour_map = correlation_matrix.get_cmap('rainbow', 90)
colour_ax = ax1.imshow(correlation, interpolation="nearest", cmap=colour_map)
labels=['','Indoors_T', 'Indoors_hum', 'Temp_outside', 'Windspeed', 'Energy Consumption']
ax1.set_xticklabels(labels)
ax1.set_yticklabels(labels)
fig.colorbar(colour_ax, ticks=[0,.5,.15,.50,.85,.9,1])
plt.title('Attribute Correlation')
ax1.grid(True)
plt.show()


#DIFFERENCING DATA:
data_mean = small_df.loc[:,'energy_consumption'].rolling(143).mean()
std= small_df.loc[:,'energy_consumption'].rolling(143).std()


# Before differencing:
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2 , label='Energy consumption')
ax.plot(data_mean, color='black', marker='o', linewidth=1, markersize=2, label='Mean' )
ax.plot(std, color='purple', marker='o', linewidth=1, markersize=2, label='Std' )
ax.legend(loc='upper left')
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 7 days of data (frequency= 10 min)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()


# After differencing:
diff=pd.DataFrame(small_df['energy_consumption'].diff())
diff= diff[1:-1]
diff_data_mean = diff.rolling(143).mean()
diff_std= diff.rolling(143).std()
#
fig, ax = plt.subplots()
ax.plot(small_df.energy_consumption, color='blue', marker='o', linewidth=1, markersize=2, label='Energy consumption' )
ax.plot(diff, color='green', marker='o', linewidth=1, markersize=2, label='Differenced energy consumption' )
ax.plot(diff_data_mean, color='black', marker='o', linewidth=1, markersize=2, label='Mean' )
ax.plot(diff_std, color='purple', marker='o', linewidth=1, markersize=2, label='Std'  )
ax.legend(loc='upper left')
ax.set(xlabel="Date-Time", ylabel='Energy consumption.')
ax.set(title="Energy consumption - 7 days of data (frequency= 30 min)")
ax.xaxis.set_major_formatter(myFmt) 
fig.autofmt_xdate()
ax.grid()   


#ACF AND PACF PLOTS BEFEORE DIFFERENCING: 
plot_acf(small_df.loc[:,'energy_consumption'])
plt.show()

plot_pacf(small_df.loc[:,'energy_consumption'], lags=143)
plt.show()    


#ACF AND PACF PLOTS AFTER DIFFERENCING: 

plot_acf(diff.loc[:, 'energy_consumption'])
plt.show()


plot_pacf(diff, lags=143)
plt.show()    


#ADF TEST: 
print('')
dfuller_tets = adfuller(diff.energy_consumption)
print('Augmented Dickey-Fuller test')
print('Test Statistic              ' + str(dfuller_tets[0]))
print('p-value                      ' + str(dfuller_tets[1]))
print('#Lags Used                   ' + str(dfuller_tets[2]))
print('Number of Observations Used  ' + str(dfuller_tets[3]))
print('Critical Value 1%           ' + str(dfuller_tets[4]['1%']))
print('Critical Value 5%           ' + str(dfuller_tets[4]['5%']))
print('Critical Value 10%          ' + str(dfuller_tets[4]['10%']))












#---  Cross validation for window size (SVR in this case) ---------
#The same approach can be generalized to other models 

increment=40
times=1

t_res=pd.DataFrame(columns=['window_size', 'training_MSE', 'test_MSE'], index=np.arange(50, 550, 50))
t_res.iloc[:,:]=0

for each in np.arange(1, 11):
    
    
    x_train, y_train, x_test, y_test= split_data(new_data, increment)#this increment will make sure we move the window size
    
    
    
    #This experiment is carried out AFTER we get the result from the Grid search function
    for j in t_res.index:
        print(j)
        training_pred= fit_gs.predict(np.array(x_train.iloc[j:j+horizon,:]))
        training_pred= pd.DataFrame(training_pred, index=x_train.iloc[j:j+horizon,:].index)
        training_mse=  np.sqrt(mean_squared_error(x_train.iloc[j:j+horizon,:], training_pred))
        
        fit_gs= search.fit(np.array(x_test.iloc[0:j,:]), np.array(y_test.iloc[0:j,:]).ravel())
        test_pred= fit_gs.predict(np.array(x_test.iloc[j:j+horizon,:]))
        test_pred= pd.DataFrame(test_pred, index=x_test.iloc[j:j+horizon,:].index)
        test_mse=  np.sqrt(mean_squared_error(x_test.iloc[j:j+horizon,:], test_pred))
    
            
        
        t_res.loc[j,'window_size']=j
        t_res.loc[j,'training_MSE']=t_res.loc[j,'training_MSE'] + training_mse
        t_res.loc[j,'test_MSE']=t_res.loc[j,'test_MSE'] + test_mse

    
    increment= increment+5
    print(str(times))
    times= times+1

#once I get the restuls, I simply divide the training MSE and the test MSE by the number of folds
#which will give me the average error per window size across all folds

       
#----------------------------   



