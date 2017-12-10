


import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cPickle
import warnings
warnings.filterwarnings('ignore')
#get_ipython().magic(u'matplotlib inline')

plt.rcParams["axes.labelsize"] = 30
sns.set(font_scale=1.5)




print("Loading Training Data" )
df = pd.read_csv('train.csv')


## Extract hour, day, month, year from datetime

df['hour'] = df['datetime'].apply(lambda x : int(x.split(' ')[1].split(':')[0]))
df['day'] = df['datetime'].apply(lambda x : parser.parse(x).weekday())

df['month'] = df['datetime'].apply(lambda x : int(x.split(' ')[0].split('-')[1]))
df['year'] = df['datetime'].apply(lambda x : int(x.split(' ')[0].split('-')[0]))






# ## Exploratory Data Analysis 


## Convert to datatype categorical 
df['season']=df['season'].astype('category')
df['weather']=df['weather'].astype('category')
df['holiday']=df['holiday'].astype('category')
df['workingday']=df['workingday'].astype('category')
df['year'] = df['year'].astype('category')
df['hour'] = df['hour'].astype('category')
df['day'] = df['day'].astype('category')
df['month'] = df['month'].astype('category')


# Now that we have the required datatypes, we can move forward with the analysis. For visualisation purpose, we will map the values of categorical variables corresponding to the entities they actually corresponds. For eg. _season_ = 1 corresponds to spring and likewise.  
# 
# Note that _weather_ = 1 is mapped to '_Clear_' and not to 'Clear,Partly cloudy,etc' and so on for ease of visualization.




# Defining dictionaries mapping the categorical variables values.
Season = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
Weather = {1:'Clear',2:'Mist',3:'Low Rain',4:'High Rain'}
holiday = {1:'Yes',0:'No'}
workingday = {1:'Yes',0:'No'}
day = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:"Sun"}
month = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:"July",8:'Aug',9:'Sep',10:'Oct',11:"Nov",12:"Dec"}
    





# Transform the categorical variables
categ_var = {'season':Season,'weather':Weather,'holiday':holiday,'workingday':workingday,'day':day,'month':month}
for key,val in categ_var.iteritems():
    df[key] = df[key].map(val)

# Inverse dictionay mapping ( Useful for converting back to numerical values for building model)    
def inverse_map(dic):
    res = {v:k for k,v in dic.iteritems()}
    return res    



## Plot distribution of variable count

fig,axs = plt.subplots(1,2,figsize=(15,4))
sns.boxplot(x='count',data=df,ax=axs[0])
sns.distplot(df['count'],ax=axs[1])
plt.suptitle('Distribution Of Count',fontsize = 20)


## Visualisation of season and weather relationship with count 
fig = plt.figure(figsize=(20, 10)) 
gs = gridspec.GridSpec(2, 3, width_ratios=[10,10,1]) 
ax0 = plt.subplot(gs[0,1])
sns.boxplot(x='season',y='count',data=df,ax=ax0)


ax1 = plt.subplot(gs[0,0])
sns.boxplot(x='weather',y='count',data=df,ax=ax1)



ax2 = plt.subplot(gs[1,:1])
sns.barplot(x='season',y='count',hue = 'weather',data=df,ax=ax2)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)

ax3 = plt.subplot(gs[1,1:])
sns.barplot(x='year',y='count',hue = 'weather',data=df,ax=ax3)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)


## Visualisation of hour,day relationship with count
fig = plt.figure(figsize=(20, 10)) 
gs = gridspec.GridSpec(2, 3, width_ratios=[10,10,1]) 
ax0 = plt.subplot(gs[0,0])
sns.pointplot(x='hour',y='count',data=df,ax=ax0)

ax1 = plt.subplot(gs[0,1])
sns.barplot(x='day',y='count',data=df,ax=ax1)



ax2 = plt.subplot(gs[1,:])
sns.barplot(x='day',y='count',hue = 'hour',data=df,ax=ax2,ci=None)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=12, fancybox=True, shadow=True)



## Visualisation of holiday and workingday relationship with count 

fig = plt.figure(figsize=(30, 20)) 
gs = gridspec.GridSpec(3, 3, width_ratios=[10,10,1]) 
ax0 = plt.subplot(gs[0,1])
sns.barplot(x='holiday',y='count',data=df,ax=ax0)

ax1 = plt.subplot(gs[0,0])
sns.barplot(x='workingday',y='count',data=df,ax=ax1)


ax2 = plt.subplot(gs[1,:])
sns.barplot(x='holiday',y='count',hue = 'hour',data=df,ax=ax2,ci=None)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=12, fancybox=True, shadow=True)

ax3 = plt.subplot(gs[2,:])
sns.barplot(x='workingday',y='count',hue = 'hour',data=df,ax=ax3,ci=None)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=12, fancybox=True, shadow=True)



## Visualisation of continuos variables 

fig, axs = plt.subplots(1,4,figsize=(20,5))
sns.distplot(df['temp'],ax=axs[0])
sns.distplot(df['humidity'],ax=axs[1])
sns.distplot(df['atemp'],ax=axs[2])
sns.distplot(df['windspeed'],ax=axs[3])



## Plotting the corroleation of continuous variables with count
corr = df.corr()
fig, axs = plt.subplots(figsize=(20,6))
ax = sns.heatmap(corr,annot=True)


print("Building Models")



## Function to calculate log mean square error
def logMLSE(y_tar,y_pred):
    n = np.size(y_tar)
    a = np.log(y_pred+1)
    b = np.log(y_tar+1)
    #print(np.sqrt(np.sum(np.square(a-b))/n))
    return np.sqrt(np.sum(np.square(a-b)/n))





## Remap the categorical values to numerical values
for key,val in categ_var.iteritems():
    df[key] = df[key].map(inverse_map(val))





## Linear Regression Model
from sklearn import linear_model
lr = linear_model.LinearRegression()
Y = np.log(df['count'])

var = ['season','holiday','workingday','weather','temp','humidity','windspeed','hour','year','day','month']
a =[col for col in df.columns if col in var]
X = df[a]
X
lr.fit(X,Y)
y_pred=lr.predict(X)
print("Linear Regression Error = "+str(logMLSE(np.exp(Y),np.exp(y_pred))))







## Random Forest Model
from sklearn.ensemble import RandomForestRegressor
Rf = RandomForestRegressor(n_estimators=100)
Rf.fit(X,Y)
y_pred=Rf.predict(X)
print("Random Forest Error = "+str(logMLSE(np.exp(Y),np.exp(y_pred))))






with open('Rf', 'wb') as f:
    cPickle.dump(Rf, f)


# ## Final Submission 
# 





print("Creating Final Submission")
dfTest = pd.read_csv('test.csv')
dfTest['hour'] = dfTest['datetime'].apply(lambda x : int(x.split(' ')[1].split(':')[0]))
dfTest['day'] = dfTest['datetime'].apply(lambda x : parser.parse(x).weekday())
#df['count']=df['count'].apply(lambda x : np.log(x))
dfTest['month'] = dfTest['datetime'].apply(lambda x : int(x.split(' ')[0].split('-')[1]))
dfTest['year'] = dfTest['datetime'].apply(lambda x : int(x.split(' ')[0].split('-')[0]))
#del df['datetime']


dfTest['season']=dfTest['season'].astype('category')
dfTest['weather']=dfTest['weather'].astype('category')
dfTest['holiday']=dfTest['holiday'].astype('category')
dfTest['workingday']=dfTest['workingday'].astype('category')
dfTest['year'] = dfTest['year'].astype('category')
dfTest['hour'] = dfTest['hour'].astype('category')
dfTest['day'] = dfTest['day'].astype('category')
dfTest['month'] = dfTest['month'].astype('category')






a =[col for col in df.columns if col in var]
X_test = dfTest[a]

with open('Rf', 'rb') as f:
    Rf = cPickle.load(f)
y_pred = Rf.predict(X_test)

res = pd.DataFrame({"datetime": dfTest['datetime'],"count":np.exp(y_pred)})
res.to_csv('bike_predictions.csv', index=False)

print("Done")
plt.show()
