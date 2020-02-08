#!/usr/bin/env python
# coding: utf-8

# In[242]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[ ]:





# In[321]:


df_nta = pd.read_csv('mta_clean.csv')
df_nta=df_nta.dropna()
df_nta=df_nta.reset_index(drop=True)
df_nta['count'] = df_nta['entries'] + df_nta['exits']
df_bike = pd.read_csv('bike_trip_2019.csv')
df_bike=df_bike.dropna()
df_bike=df_bike.reset_index(drop=True)

nta_sort_lat = df_nta.sort_values(['latitude']) # ascending
nta_sort_lat.reset_index(drop=True,inplace=True)
nta_sort_long = df_nta.sort_values(['longitude']) # ascending
nta_sort_long.reset_index(drop=True,inplace=True)


# In[322]:


df_bike.head()


# In[323]:


df_nta.head()


# In[324]:


def binary_search(value, lst):
    value = float(value)
    l = len(lst)
    if len(lst) == 2:
        if abs(value-lst[0])<abs(value-lst[1]):
            idx = 0
        else:
            idx = 1
        return idx
    if value < lst[int(l/2)]:
        idx = binary_search(value, lst[:int(l/2)+1])
    elif value > lst[int(l/2)]:
        idx = int(l/2) + binary_search(value, lst[int(l/2):])
    else:
        idx = int(l/2)
        return idx
    return idx

def distance(tup1, tup2):
    return abs(float(tup1[0])-float(tup2[0]))+abs(float(tup1[1])-float(tup2[1]))

def find_k_stations(station_id, k):
    lat = float(df_bike[df_bike['station_id']==station_id]['station_latitude'])
    long = float(df_bike[df_bike['station_id']==station_id]['station_longitude'])
    lat_idx = binary_search(lat, list(nta_sort_lat['latitude']))
    if lat_idx-2 < 0:
        stations_on_lat = [[nta_sort_lat['short_id'][i],distance((lat,long),(nta_sort_lat['latitude'][i],nta_sort_lat['longitude'][i])),(nta_sort_lat['count'][i],nta_sort_lat['entries'][i],nta_sort_lat['exits'][i])] for i in range(lat_idx, lat_idx+5)]
    elif lat_idx+2 >= len(nta_sort_lat['short_id']):
        stations_on_lat = [[nta_sort_lat['short_id'][i],distance((lat,long),(nta_sort_lat['latitude'][i],nta_sort_lat['longitude'][i])),(nta_sort_lat['count'][i],nta_sort_lat['entries'][i],nta_sort_lat['exits'][i])] for i in range(lat_idx-5, lat_idx)]
    else:
        stations_on_lat = [[nta_sort_lat['short_id'][i],distance((lat,long),(nta_sort_lat['latitude'][i],nta_sort_lat['longitude'][i])),(nta_sort_lat['count'][i],nta_sort_lat['entries'][i],nta_sort_lat['exits'][i])] for i in range(lat_idx-2, lat_idx+3)]
    long_idx = binary_search(long, list(nta_sort_long['longitude']))
    if long_idx-2 < 0:
        stations_on_long = [[nta_sort_long['short_id'][i],distance((lat,long),(nta_sort_long['latitude'][i],nta_sort_long['longitude'][i])),(nta_sort_long['count'][i],nta_sort_long['entries'][i],nta_sort_long['exits'][i])] for i in range(long_idx, long_idx+5)]
    elif long_idx+2 > len(nta_sort_lat['short_id']):
        stations_on_long = [[nta_sort_long['short_id'][i],distance((lat,long),(nta_sort_long['latitude'][i],nta_sort_long['longitude'][i])),(nta_sort_long['count'][i],nta_sort_long['entries'][i],nta_sort_long['exits'][i])] for i in range(long_idx-5, long_idx)]
    else:
        stations_on_long = [[nta_sort_long['short_id'][i],distance((lat,long),(nta_sort_long['latitude'][i],nta_sort_long['longitude'][i])),(nta_sort_long['count'][i],nta_sort_long['entries'][i],nta_sort_long['exits'][i])] for i in range(long_idx-2, long_idx+3)]

    stations = []
    stations.extend(stations_on_long)
    stations.extend(stations_on_lat)
    stations = sorted(stations, key=lambda x:x[1])

    return [stations[i][0] for i in range(k)], [stations[i][1] for i in range(k)], [stations[i][2] for i in range(k)]
        


# In[325]:


stations, distances, counts = find_k_stations(72,3)


# In[326]:


print(stations)
print(distances)
print(counts)


# In[357]:


df = df_bike[['station_id','count_x','count_y','total_counts']]

station1=[]
station2=[]
station3=[]
distance1=[]
distance2=[]
distance3=[]
count1=[]
entries1=[]
exits1=[]
count2=[]
entries2=[]
exits2=[]
count3=[]
entries3=[]
exits3=[]

for i in range(len(df['station_id'])):
    stations, distances, counts = find_k_stations(df['station_id'][i], 3)
    station1.append(stations[0])
    station2.append(stations[1])
    station3.append(stations[2])
    distance1.append(distances[0])
    distance2.append(distances[1])
    distance3.append(distances[2])
    count1.append(counts[0][0])
    entries1.append(counts[0][1])
    exits1.append(counts[0][2])
    count2.append(counts[1][0])
    entries2.append(counts[1][1])
    exits2.append(counts[1][2])
    count3.append(counts[2][0])
    entries3.append(counts[2][1])
    exits3.append(counts[2][2])

df['station1'] = station1
df['station2'] = station2
df['station3'] = station3
df['distance1'] = distance1
df['distance2'] = distance2
df['distance3'] = distance3
df['count1']=count1
df['entries1']=entries1
df['exits1']=exits1
df['count2']=count2
df['entries2']=entries2
df['exits2']=exits2
df['count3']=count3
df['entries3']=entries3
df['exits3']=exits3

df['feature'] = sum([df['count'+str(i)]/df['distance'+str(i)]/df['distance'+str(i)]/(1/df['distance1']+1/df['distance2']+1/df['distance3']) for i in range(1,4)])
#df['feature'] = sum([1/df['distance'+str(i)] for i in range(1,4)])
df['1/d1'] = 1/df['distance1']
df['1/d1^2'] = 1/df['distance1']/df['distance1']
df['d_exp'] = np.exp(-df['distance1'])
df['count1/d1'] = df['count1']/df['distance1']

df['d1'] = df['distance1']
df['d1^2'] = df['distance1']**2
df['d1^3'] = df['distance1']**3
df['d1^4'] = df['distance1']**4
df['logcount'] = np.log(df['total_counts'])
df['logcount1'] = np.log(df['count1'])


# In[345]:


df


# In[303]:


plt.scatter(df['d1'],df['total_counts'],alpha=0.3)
#plt.xlim((0,5000))
plt.show()
# 1/30


# In[237]:


type(df['distance1'][0])
df[['distance1','distance2','distance3','count1','count2','count3']]


# In[366]:


X = sm.add_constant(np.log(df[['logcount1','distance1']]))
Y = df['logcount']

model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 

summary = model.summary()
print(summary)


# In[288]:


model.params


# In[299]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[329]:


df.to_csv('df_analysis.csv')


# In[ ]:




