
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df_student = pd.read_csv('student-por.csv',sep=';',header=1,skipinitialspace=True)


# In[3]:


df_student.head()


# In[4]:


# Q1: How many columns are there?

# A1: There are 33 columns.


# In[5]:


# Q1: What types are the columms?

# A1: see below


# In[6]:


df_student.dtypes


# In[7]:


# Q1: Are there any missing values?

# A1: No


# In[8]:


df_student.isnull().values.any()


# In[9]:


df_student.describe()


# Q1: Are there an equal number of M and F?
# 
# A1: There are 382 M and 266 F.

# In[10]:


df_student.groupby('F').count()


# In[11]:


# Add column names for readability
names = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',           'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',            'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc',            'health','absences','G1','G2','G3']


# In[12]:


df_student.columns = names


# In[13]:


df_student.head(10)


# In[14]:


df_student.shape


# In[15]:


df_student_f = df_student[(df_student['sex']) == 'F']
print(df_student_f)


# In[16]:


fx = df_student_f['G3']
num_bins = 8
plt.hist(fx, num_bins, color = 'red', alpha = 0.5)
plt.show()


# In[17]:


df_student_m = df_student[(df_student['sex']) == 'M']
print (df_student_m)


# In[18]:


mx = df_student_m['G3']
num_bins = 8
plt.hist(mx, num_bins, color = 'blue', alpha = 0.5)
plt.show()


# In[19]:


df_student_f.mean()


# In[20]:


df_student_m.mean()


# In[21]:


# Is there a correlation between the amount a student studies (st) and their final grade (g)?

df_student_f_st = df_student_f['studytime']
df_student_f_g = df_student_f['G3']

df_student_m_st = df_student_m['studytime']
df_student_m_g = df_student_m['G3']

import seaborn as sns
sns.boxplot(df_student_f_st,df_student_f_g)
plt.show()
sns.boxplot(df_student_m_st,df_student_m_g)
plt.show()


# In[22]:


# Conclusion: increasing study time correlates positively with better final grades for female 
# students. For male students the study time correlates positively with better final grades up to 
# up to a study time of 5 to 10 hours. For male students, studying longer than 10 hours has a negative
# correlation with final grades.


# In[23]:


from scipy.stats import linregress
linregress(df_student_f_st, df_student_f_g)


# In[24]:


linregress(df_student_m_st, df_student_m_g)


# In[25]:


# Is there a correlation between the amount of time that the students go out (goout) and their
# grades (g)?

df_student_f_go = df_student_f['goout']

df_student_m_go = df_student_m['goout']

sns.boxplot(df_student_f_go, df_student_f_g)
plt.show()

sns.boxplot(df_student_m_go, df_student_m_g)
plt.show


# Conclusion: Going out with friends doesn't correlate with their final grade (G3)

# In[26]:


df_student


# In[27]:


# Is there a correlation between the student's family size and their grades?

df_student_fs = df_student['famsize']
df_student_g = df_student['G3']

sns.boxplot(df_student_fs, df_student_g)
plt.show()


#  Concluson: Family size has no correlation with final student grades

# In[28]:


df_student


# In[29]:


list(df_student)


# In[30]:


df_student_famsize = df_student['famsize']
df_student_g = df_student['G3']

sns.boxplot(df_student_famsize, df_student_g)
plt.show()


# In[31]:


# Is there a correlation between the student's parent's cohabitation and grades?

df_student_Ps = df_student['Pstatus']
df_student_g = df_student['G3']

sns.boxplot(df_student_Ps, df_student_g)
plt.show()


# In[32]:


df_student_mjob = df_student['Mjob']
df_student_g = df_student['G3']

sns.boxplot(df_student_mjob, df_student_g)
plt.show()


# In[33]:


df_student_fjob = df_student['Fjob']
df_student_g = df_student['G3']

sns.boxplot(df_student_fjob, df_student_g)
plt.show()


# In[34]:


df_student_reason = df_student['reason']
df_student_g = df_student['G3']

sns.boxplot(df_student_reason, df_student_g)
plt.show()


# In[35]:


df_student_guard = df_student['guardian']
df_student_g = df_student['G3']

sns.boxplot(df_student_guard, df_student_g)
plt.show()


# In[36]:


# Is there a correlation between the education of the student's mother and their grades?

df_student_medu = df_student['Medu']
df_student_g = df_student['G3']

sns.boxplot(df_student_medu, df_student_g)
plt.show()


# In[37]:


# Is there a correlation between the father's education level and the students grades?

df_student_fedu = df_student['Fedu']
df_student_g = df_student['G3']


# In[38]:


sns.boxplot(df_student_fedu, df_student_g)
plt.show() 


# In[39]:


# Is there a correlation between the travel time from home to school and their grades?

df_student_tt = df_student['traveltime']
df_student_g = df_student['G3']

sns.boxplot(df_student_tt, df_student_g)
plt.show()


# In[40]:


df_student_stime = df_student['studytime']
df_student_g = df_student['G3']

sns.boxplot(df_student_stime, df_student_g)
plt.show()


# In[41]:


# Is there a correlation between the student's previous failures and their grades?

df_student_fail = df_student['failures']
df_student_g = df_student['G3']

sns.boxplot(df_student_fail, df_student_g)
plt.show()


# In[42]:


# Is there a correlation between the student's support from the school and their grades?

df_student_sups = df_student['schoolsup']
df_student_g = df_student['G3']

sns.boxplot(df_student_sups, df_student_g)
plt.show()


# In[43]:


# Is there a correlation between the student's family education support and their grades?

df_student_fes = df_student['famsup']
df_student_g = df_student['G3']

sns.boxplot(df_student_fes, df_student_g)
plt.show()


# In[44]:


# Is there a correlation between the student's payment for extra classes and their grades?

df_student_exc = df_student['paid']
df_student_g = df_student['G3']

sns.boxplot(df_student_exc, df_student_g)
plt.show()


# In[45]:


# Is there a correlation between the student's attendance in extra curricular activities 
#  and their grades?

df_student_xca = df_student['activities']
df_student_g = df_student['G3']

sns.boxplot(df_student_xca, df_student_g)
plt.show()


# In[46]:


df_student_nurs = df_student['nursery']
df_student_g = df_student['G3']

sns.boxplot(df_student_nurs, df_student_g)
plt.show()


# In[47]:


# Is there a correlation between the student's who want to pursue higher education and their grades?

df_student_high = df_student['higher']
df_student_g = df_student['G3']

sns.boxplot(df_student_high, df_student_g)
plt.show()


# In[48]:


# Is there a correlation between the student's access to the internet at home and their grades?

df_student_net = df_student['internet']
df_student_g = df_student['G3']

sns.boxplot(df_student_net, df_student_g)
plt.show()


# In[49]:


df_student_rom = df_student['romantic']
df_student_g = df_student['G3']

sns.boxplot(df_student_rom, df_student_g)
plt.show()


# In[50]:


# Is there a correlation between the student's quality of family relationships and their grades?

df_student_famrel = df_student['famrel']
df_student_g = df_student['G3']

sns.boxplot(df_student_famrel, df_student_g)
plt.show()


# In[51]:


# Is there a correlation between the student's free time and their grades?

df_student_ft = df_student['freetime']
df_student_g = df_student['G3']

sns.boxplot(df_student_ft, df_student_g)
plt.show()


# In[52]:


df_student_goout = df_student['goout']
df_student_g = df_student['G3']

sns.boxplot(df_student_goout, df_student_g)
plt.show()


# In[53]:


# Is there a correlation between the student's daily comsumption of alcohol and their grades?

df_student_dalc = df_student['Dalc']
df_student_g = df_student['G3']

sns.boxplot(df_student_dalc, df_student_g)
plt.show()


# In[54]:


# Is there a correlation between the student's weekend comsumption of alcohol and their grades?

df_student_walc = df_student['Walc']
df_student_g = df_student['G3']

sns.boxplot(df_student_walc, df_student_g)
plt.show()


# In[55]:


# Is there a correlation between the student's health and their grades?

df_student_h = df_student['health']
df_student_g = df_student['G3']

sns.boxplot(df_student_h, df_student_g)
plt.show()


# In[56]:


# Is there a correlation between the student's absences and their grades?

df_student_ab = df_student['absences']
df_student_g = df_student['G3']

sns.boxplot(df_student_ab, df_student_g)
plt.show()


# Calculate the Pearson r for the correlation between female student study time and their grades.

# In[57]:


from scipy import stats

stats.pearsonr(df_student_f_st, df_student_f_g)


# There is linear correlation (r=0.25), and the p value is small 

# In[58]:


stats.pearsonr(df_student_fail, df_student_g)


# In[59]:


df_student_high_no = df_student['higher'] == 1
df_student_high_yes = df_student['higher']  == 0

df_student_high_tot = df_student_high_no | df_student_high_yes
df_student_high_tot = df_student_high_tot.astype(int)
#print(df_student_high_tot)
stats.pearsonr(df_student_high_tot, df_student_g)


# In[60]:


stats.pearsonr(df_student_dalc, df_student_g)


# In[61]:


stats.pearsonr(df_student_ab, df_student_g)


# In[62]:


import statsmodels.api as sm

x = df_student_f_st
y = df_student_f_g

model = sm.OLS(y,x).fit()
predict = model.predict(x)

print (model.summary)


# In[63]:


from scipy.stats import linregress
linregress(df_student_fail, df_student_g)


# Apply the sklearn preprocessing and one hot encoder to all of the attributes starting with reason.
# 

# In[64]:


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# In[65]:


data_rea = df_student['reason']
values_rea = array(data_rea)
print (values_rea)


# In[66]:


onehot_encoder_rea = OneHotEncoder(sparse = False)


# In[67]:


label_encoder = LabelEncoder()
integer_encoded_rea = label_encoder.fit_transform(values_rea)
print (integer_encoded_rea)

#integer_encoded_rea = integer_encoded_rea.reshape(len(integer_encoded_rea),1)


# In[68]:


onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_rea = integer_encoded_rea.reshape(len(integer_encoded_rea),1)
onehot_encoded_rea = onehot_encoder.fit_transform(integer_encoded_rea)
print(onehot_encoded_rea)


# Apply the sklearn preprocessing and one hot encoder to the categorical attribute: famsize.

# In[69]:


df_student['reason'] = onehot_encoded_rea


# In[70]:


data_famsize = df_student['famsize']
values_famsize = array(data_famsize)


# In[71]:


print (values_famsize)


# Integer encode

# In[72]:


label_encoder = LabelEncoder()
integer_encoded_famsize = label_encoder.fit_transform(values_famsize)
print (integer_encoded_famsize)


# Binary encode

# In[73]:


onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_famsize = integer_encoded_famsize.reshape(len(integer_encoded_famsize),1)
onehot_encoded_famsize = onehot_encoder.fit_transform(integer_encoded_famsize)
print(onehot_encoded_famsize)


# In[74]:


df_student['famsize'] = onehot_encoded_famsize


# Invert

# In[75]:


inverted_famsize = label_encoder.inverse_transform([argmax(onehot_encoded_famsize[0 , :])])
print(inverted_famsize)


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: Pstatus.

# In[76]:


data_Pstatus = df_student['Pstatus']
values_Pstatus = array(data_Pstatus)

print (values_Pstatus)


# In[77]:


label_encoder_Pstatus = LabelEncoder()
integer_encoded_Pstatus = label_encoder_Pstatus.fit_transform(values_Pstatus)
print (integer_encoded_Pstatus)


# In[78]:


onehot_encoder_Pstatus = OneHotEncoder(sparse = False)
integer_encoded_Pstatus = integer_encoded_Pstatus.reshape(len(integer_encoded_Pstatus),1)
onehot_encoded_Pstatus = onehot_encoder_Pstatus.fit_transform(integer_encoded_Pstatus)
print(onehot_encoded_Pstatus)


# In[79]:


df_student['Pstatus'] = onehot_encoded_Pstatus


# In[80]:


inverted_Pstatus = label_encoder_Pstatus.inverse_transform([argmax(onehot_encoded_Pstatus[0 , :])])
print(inverted_Pstatus)


# Apply the sklearn preprocessing and one hot encoder to the categorical attribute: Mjob.

# In[81]:


data_Mjob = df_student['Mjob']
values_Mjob = array(data_Mjob)

print (values_Mjob)


# In[82]:


label_encoder_Mjob = LabelEncoder()
integer_encoded_Mjob = label_encoder_Mjob.fit_transform(values_Mjob)
print (integer_encoded_Mjob)


# In[83]:


onehot_encoder_Mjob = OneHotEncoder(sparse = False)
integer_encoded_Mjob = integer_encoded_Mjob.reshape(len(integer_encoded_Mjob),1)
onehot_encoded_Mjob = onehot_encoder_Mjob.fit_transform(integer_encoded_Mjob)
print(onehot_encoded_Mjob)


# In[84]:


df_student['Mjob'] = onehot_encoded_Mjob


# Apply the sklearn preprocessing and one hot encoder to the categorical attribute: Fjob.

# In[85]:


data_Fjob = df_student['Fjob']
values_Fjob = array(data_Fjob)


# In[86]:


print (values_Fjob)


# In[87]:


label_encoder_Fjob = LabelEncoder()
integer_encoded_Fjob = label_encoder_Fjob.fit_transform(values_Fjob)
print (integer_encoded_Fjob)


# In[88]:


onehot_encoder_Fjob = OneHotEncoder(sparse = False)
integer_encoded_Fjob = integer_encoded_Fjob.reshape(len(integer_encoded_Fjob),1)
onehot_encoded_Fjob = onehot_encoder_Fjob.fit_transform(integer_encoded_Fjob)
print(onehot_encoded_Fjob)


# In[89]:


df_student['Fjob'] = onehot_encoded_Fjob


# Apply the sklearn preprocessing and one hot encoder to the nominal attribute: guard.

# In[90]:


data_guard = df_student['guardian']
values_guard = array(data_guard)

print (values_guard)


# In[91]:


onehot_encoder_guard = OneHotEncoder(sparse = False)


# In[92]:


label_encoder = LabelEncoder()
integer_encoded_guard = label_encoder.fit_transform(values_guard)
print (integer_encoded_guard)


# In[93]:


onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_guard = integer_encoded_guard.reshape(len(integer_encoded_guard),1)
onehot_encoded_guard = onehot_encoder.fit_transform(integer_encoded_guard)
print(onehot_encoded_guard)


# In[94]:


df_student['guardian'] = onehot_encoded_guard


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: schoolsup.

# In[95]:


data_schoolsup = df_student['schoolsup']
values_schoolsup = array (data_schoolsup)
print (values_schoolsup)


# In[96]:


onehot_encoder_schoolsup = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_schoolsup = label_encoder.fit_transform(values_schoolsup)
print (integer_encoded_schoolsup)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_schoolsup = integer_encoded_schoolsup.reshape(len(integer_encoded_schoolsup),1)
onehot_encoded_schoolsup = onehot_encoder.fit_transform(integer_encoded_schoolsup)
print(onehot_encoded_schoolsup)

df_student['schoolsup'] = onehot_encoded_schoolsup


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: paid.

# In[97]:


data_paid = df_student['paid']
values_paid = array(data_paid)
print (values_paid)


# In[98]:


onehot_encoder_paid = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_paid = label_encoder.fit_transform(values_paid)
print (integer_encoded_paid)


# In[99]:


onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_paid = integer_encoded_paid.reshape(len(integer_encoded_paid),1)
onehot_encoded_paid = onehot_encoder.fit_transform(integer_encoded_paid)
print(onehot_encoded_paid)
df_student['paid'] = onehot_encoded_paid


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: activities.

# In[100]:


data_activities = df_student['activities']
values_activities = array (data_activities)
print (values_activities)
onehot_encoder_activities = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_activities = label_encoder. fit_transform(values_activities)
print (integer_encoded_activities)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_activities = integer_encoded_activities.reshape(len(integer_encoded_activities),1)
onehot_encoded_activities = onehot_encoder.fit_transform(integer_encoded_activities)
print(onehot_encoded_activities)
df_student['activities'] = onehot_encoded_activities


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: nursery.

# In[101]:


data_nursery = df_student['nursery']
values_nursery = array (data_nursery)
print (values_nursery)
onehot_encoder_nursery = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_nursery = label_encoder.fit_transform(values_nursery)
print (integer_encoded_nursery)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_nursery = integer_encoded_nursery.reshape(len(integer_encoded_nursery),1)
onehot_encoded_nursery = onehot_encoder.fit_transform(integer_encoded_nursery)
print(onehot_encoded_nursery)
df_student['nursery'] = onehot_encoded_nursery


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: higher.

# In[102]:


data_higher = df_student['higher']
values_higher = array (data_higher)
print (values_higher)
onehot_encoder_higher = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_higher = label_encoder.fit_transform(values_higher)
print (integer_encoded_higher)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_higher = integer_encoded_higher.reshape(len(integer_encoded_higher),1)
onehot_encoded_higher = onehot_encoder.fit_transform(integer_encoded_higher)
print(onehot_encoded_higher)
df_student['higher'] = onehot_encoded_higher


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: internet.

# In[103]:


data_internet = df_student['internet']
values_internet = array (data_internet)
print (values_internet)
onehot_encoder_internet = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_internet = label_encoder.fit_transform(values_internet)
print (integer_encoded_internet)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_internet = integer_encoded_internet.reshape(len(integer_encoded_internet),1)
onehot_encoded_internet = onehot_encoder.fit_transform(integer_encoded_internet)
print(onehot_encoded_internet)
df_student['internet'] = onehot_encoded_internet


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: romantic.

# In[104]:


data_romantic = df_student['romantic']
values_romantic = array (data_romantic)
print (values_romantic)
onehot_encoder_romantic = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_romantic = label_encoder.fit_transform(values_romantic)
print (integer_encoded_romantic)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_romantic = integer_encoded_romantic.reshape(len(integer_encoded_romantic),1)
onehot_encoded_romantic = onehot_encoder.fit_transform(integer_encoded_romantic)
print(onehot_encoded_romantic)
df_student['romantic'] = onehot_encoded_romantic


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: school.

# In[105]:


data_school = df_student['school']
values_school = array (data_school)
print (values_school)
onehot_encoder_school = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_school = label_encoder.fit_transform(values_school)
print (integer_encoded_school)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_school = integer_encoded_school.reshape(len(integer_encoded_school),1)
onehot_encoded_school = onehot_encoder.fit_transform(integer_encoded_school)
print(onehot_encoded_school)
df_student['school'] = onehot_encoded_school


# Apply the sklearn preprocessing and one hot encoder to the binary attribute: sex.

# In[106]:


data_sex = df_student['sex']
values_sex = array (data_sex)
print (values_sex)
onehot_encoder_sex = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_sex = label_encoder.fit_transform(values_sex)
print (integer_encoded_sex)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_sex = integer_encoded_sex.reshape(len(integer_encoded_sex),1)
onehot_encoded_sex = onehot_encoder.fit_transform(integer_encoded_sex)
print(onehot_encoded_sex)
df_student['sex'] = onehot_encoded_sex


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: Medu.

# In[107]:


data_Medu = df_student['Medu']
values_Medu = array (data_Medu)
print (values_Medu)
onehot_encoder_Medu = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_Medu = label_encoder.fit_transform(values_Medu)
print (integer_encoded_Medu)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_Medu = integer_encoded_Medu.reshape(len(integer_encoded_Medu),1)
onehot_encoded_Medu = onehot_encoder.fit_transform(integer_encoded_Medu)
print(onehot_encoded_Medu)
df_student['Medu'] = onehot_encoded_Medu


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: Fedu.

# In[108]:


data_Fedu = df_student['Fedu']
values_Fedu = array (data_Fedu)
print (values_Fedu)
onehot_encoder_Fedu = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_Fedu = label_encoder.fit_transform(values_Fedu)
print (integer_encoded_Fedu)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_Fedu = integer_encoded_Fedu.reshape(len(integer_encoded_Fedu),1)
onehot_encoded_Fedu = onehot_encoder.fit_transform(integer_encoded_Fedu)
print(onehot_encoded_Fedu)
df_student['Fedu'] = onehot_encoded_Fedu


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: age.

# In[109]:


data_age = df_student['age']
values_age = array (data_age)
print (values_age)
onehot_encoder_age = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_age = label_encoder.fit_transform(values_age)
print (integer_encoded_age)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_age = integer_encoded_age.reshape(len(integer_encoded_age),1)
onehot_encoded_age = onehot_encoder.fit_transform(integer_encoded_age)
print(onehot_encoded_age)
df_student['age'] = onehot_encoded_age


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: address.

# In[110]:


data_address = df_student['address']
values_address = array (data_address)
print (values_address)
onehot_encoder_address = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_address = label_encoder.fit_transform(values_address)
print (integer_encoded_address)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_address = integer_encoded_address.reshape(len(integer_encoded_address),1)
onehot_encoded_address = onehot_encoder.fit_transform(integer_encoded_address)
print(onehot_encoded_address)
df_student['address'] = onehot_encoded_address


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: traveltime.

# In[111]:


data_traveltime = df_student['traveltime']
values_traveltime = array (data_traveltime)
print (values_traveltime)
onehot_encoder_traveltime = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_traveltime = label_encoder.fit_transform(values_traveltime)
print (integer_encoded_traveltime)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_traveltime = integer_encoded_traveltime.reshape(len(integer_encoded_traveltime),1)
onehot_encoded_traveltime = onehot_encoder.fit_transform(integer_encoded_traveltime)
print(onehot_encoded_traveltime)
df_student['traveltime'] = onehot_encoded_traveltime


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: studytime.

# In[112]:


data_studytime = df_student['studytime']
values_studytime = array (data_studytime)
print (values_studytime)
onehot_encoder_studytime = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_studytime = label_encoder.fit_transform(values_studytime)
print (integer_encoded_studytime)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_studytime = integer_encoded_studytime.reshape(len(integer_encoded_studytime),1)
onehot_encoded_studytime = onehot_encoder.fit_transform(integer_encoded_studytime)
print(onehot_encoded_studytime)
df_student['studytime'] = onehot_encoded_studytime


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: failures.

# In[113]:


data_failures = df_student['failures']
values_failures = array (data_failures)
print (values_failures)
onehot_encoder_failures = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_failures = label_encoder.fit_transform(values_failures)
print (integer_encoded_failures)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_failures = integer_encoded_failures.reshape(len(integer_encoded_failures),1)
onehot_encoded_failures = onehot_encoder.fit_transform(integer_encoded_failures)
print(onehot_encoded_failures)
df_student['failures'] = onehot_encoded_failures


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: famrel.

# In[114]:


data_famrel = df_student['famrel']
values_famrel = array (data_famrel)
print (values_famrel)
onehot_encoder_famrel = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_famrel = label_encoder.fit_transform(values_famrel)
print (integer_encoded_famrel)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_famrel = integer_encoded_famrel.reshape(len(integer_encoded_famrel),1)
onehot_encoded_famrel = onehot_encoder.fit_transform(integer_encoded_famrel)
print(onehot_encoded_famrel)
df_student['famrel'] = onehot_encoded_famrel


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: freetime.

# In[115]:


data_freetime = df_student['freetime']
values_freetime = array (data_freetime)
print (values_freetime)
onehot_encoder_freetime = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_freetime = label_encoder.fit_transform(values_freetime)
print (integer_encoded_freetime)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_freetime = integer_encoded_freetime.reshape(len(integer_encoded_freetime),1)
onehot_encoded_freetime = onehot_encoder.fit_transform(integer_encoded_freetime)
print(onehot_encoded_freetime)
df_student['freetime'] = onehot_encoded_freetime


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: goout.

# In[116]:


data_goout = df_student['goout']
values_goout = array (data_goout)
print (values_goout)
onehot_encoder_goout = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_goout = label_encoder.fit_transform(values_goout)
print (integer_encoded_goout)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_goout = integer_encoded_goout.reshape(len(integer_encoded_goout),1)
onehot_encoded_goout = onehot_encoder.fit_transform(integer_encoded_goout)
print(onehot_encoded_goout)
df_student['goout'] = onehot_encoded_goout


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: Dalc.

# In[117]:


data_Dalc = df_student['Dalc']
values_Dalc = array (data_Dalc)
print (values_Dalc)
onehot_encoder_Dalc = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_Dalc = label_encoder.fit_transform(values_Dalc)
print (integer_encoded_Dalc)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_Dalc = integer_encoded_Dalc.reshape(len(integer_encoded_Dalc),1)
onehot_encoded_Dalc = onehot_encoder.fit_transform(integer_encoded_Dalc)
print(onehot_encoded_Dalc)
df_student['Dalc'] = onehot_encoded_Dalc


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: Walc.

# In[118]:


data_Walc = df_student['Walc']
values_Walc = array (data_Walc)
print (values_Walc)
onehot_encoder_Walc = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_Walc = label_encoder.fit_transform(values_Walc)
print (integer_encoded_Walc)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_Walc = integer_encoded_Walc.reshape(len(integer_encoded_Walc),1)
onehot_encoded_Walc = onehot_encoder.fit_transform(integer_encoded_Walc)
print(onehot_encoded_Walc)
df_student['Walc'] = onehot_encoded_Walc


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: health.

# In[119]:


data_health = df_student['health']
values_health = array (data_health)
print (values_health)
onehot_encoder_health = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_health = label_encoder.fit_transform(values_health)
print (integer_encoded_health)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_health = integer_encoded_health.reshape(len(integer_encoded_health),1)
onehot_encoded_health = onehot_encoder.fit_transform(integer_encoded_health)
print(onehot_encoded_health)
df_student['health'] = onehot_encoded_health


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: absences.

# In[120]:


data_absences = df_student['absences']
values_absences = array (data_absences)
print (values_absences)
onehot_encoder_absences = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_absences = label_encoder.fit_transform(values_absences)
print (integer_encoded_absences)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_absences = integer_encoded_absences.reshape(len(integer_encoded_absences),1)
onehot_encoded_absences = onehot_encoder.fit_transform(integer_encoded_absences)
print(onehot_encoded_absences)
df_student['absences'] = onehot_encoded_absences


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: famsup.

# In[121]:


data_famsup = df_student['famsup']
values_famsup = array (data_famsup)
print (values_famsup)
onehot_encoder_famsup = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_famsup = label_encoder.fit_transform(values_famsup)
print (integer_encoded_famsup)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_famsup = integer_encoded_famsup.reshape(len(integer_encoded_famsup),1)
onehot_encoded_famsup = onehot_encoder.fit_transform(integer_encoded_famsup)
print(onehot_encoded_famsup)
df_student['famsup'] = onehot_encoded_famsup


# In[122]:


df_student['famsup']


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: G1.

# In[123]:


data_G1 = df_student['G1']
values_G1 = array (data_G1)
print (values_G1)
onehot_encoder_G1 = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_G1 = label_encoder.fit_transform(values_G1)
print (integer_encoded_G1)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_G1 = integer_encoded_G1.reshape(len(integer_encoded_G1),1)
onehot_encoded_G1 = onehot_encoder.fit_transform(integer_encoded_G1)
print(onehot_encoded_G1)
df_student['G1'] = onehot_encoded_G1


# In[124]:


df_student['G1']


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: G2.

# In[126]:


data_G2 = df_student['G2']
values_G2 = array (data_G2)
print (values_G2)
onehot_encoder_G2 = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_G2 = label_encoder.fit_transform(values_G2)
print (integer_encoded_G2)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_G2 = integer_encoded_G2.reshape(len(integer_encoded_G2),1)
onehot_encoded_G2 = onehot_encoder.fit_transform(integer_encoded_G2)
print(onehot_encoded_G2)
df_student['G2'] = onehot_encoded_G2


# Apply the sklearn preprocessing and one hot encoder to the numeric attribute: G3.

# In[127]:


data_G3 = df_student['G3']
values_G3 = array (data_G3)
print (values_G3)
onehot_encoder_G3 = OneHotEncoder (sparse = False)
label_encoder = LabelEncoder()
integer_encoded_G3 = label_encoder.fit_transform(values_G3)
print (integer_encoded_G3)
onehot_encoder = OneHotEncoder(sparse = False)
integer_encoded_G3 = integer_encoded_G3.reshape(len(integer_encoded_G3),1)
onehot_encoded_G3 = onehot_encoder.fit_transform(integer_encoded_G3)
print(onehot_encoded_G3)
df_student['G3'] = onehot_encoded_G3


# In[128]:


df_student


# In[129]:


df_student.shape


# In[130]:


df_student.corr()


# In[131]:


df_student.shape


# In[132]:


df_student


# Correlation matrix

# In[133]:


correlation = df_student.corr()


# In[142]:


correlation


# In[151]:


#def correlation_matrix(correlation):
 #   from matplotlib import pyplot as plt
  #  from matplotlib import cm as cm

   # %matplotlib inline

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #cmap = cm.get_cmap('jet', 30)
    #cax = ax1.imshow(correlation, cmap=cmap) #interpolation = 'nearest'
    #ax1.grid(True)
    #plt.title('Student Feature Correlation')
    #index=(['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'])
    #ax1.set_xticklabels(index,fontsize=10, rotation = 45)
    #ax1.set_yticklabels(index,fontsize=10, rotation = 45)
    
    #fig.set_size_inches(10,10)
    
    #plt.tight_layout()
    
    #xmin,xmax = as1.get_xlin()
    #ymin,ymax = ax1.get_ylim()
    # ax1.set_xlim(xmin, xmax)
    #ax1.set_ylim(ymin,ymax)
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    #fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,.90,1])
    #plt.show()

#correlation_matrix(correlation)


# In[152]:


def correlation_matrix(correlation):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    get_ipython().magic('matplotlib inline')

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(correlation, cmap=cmap) #interpolation = 'nearest'
    ax1.grid(True)
    plt.title('Student Feature Correlation')
    index=(['school','sex','age','address','famsize','Pstatus','Medu',             'Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime'            'failures','schoolsup','famsup','paid','activities','nursery','higher'            'internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3'])
    ax1.set_xticklabels(index,fontsize=10, rotation = 45)
    ax1.set_yticklabels(index,fontsize=10, rotation = 45)
    
    fig.set_size_inches(10,10)
    
    plt.tight_layout()
    
    #xmin,xmax = as1.get_xlin()
    #ymin,ymax = ax1.get_ylim()
    # ax1.set_xlim(xmin, xmax)
    #ax1.set_ylim(ymin,ymax)
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8,.90,1])
    plt.show()

correlation_matrix(correlation)


# In[135]:


import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[136]:


df_student.shape


# Replace yes/no data with 1/0 for the attributes: internet, romantic, and higher

# In[137]:


df_student['internet'].replace(('yes', 'no'),(1,0),inplace = True)


# In[ ]:


df_student['romantic'].replace(('yes', 'no'),(1,0),inplace = True)


# In[ ]:


df_student['higher'].replace(('yes', 'no'),(1,0),inplace = True)


# In[ ]:


df_student.shape


# In[ ]:


print(df_student)


# In[ ]:


import numpy as np
import pandas as pd
df_student_2 = df_student.T


# In[ ]:


print (df_student_2)


# In[ ]:


df_student.shape


# In[ ]:


x = df_student[['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason',           'guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities',            'nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc',            'health','absences']]

y = df_student['G3']


# In[ ]:


import pandas as pd
from sklearn import linear_model

x_train = x

y_train = y

x_test = x

y_test = y

#ols = linear_model.LinearRegression()
#model = ols.fit(x_train, y_train)

#print (model.predict(x_test)[0:5])


# In[ ]:


y.shape


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# In[ ]:


y_test.head()


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33, random_state = 42)


# In[ ]:


model.predict(np.array(x_test))


# In[ ]:


y_test


# In[ ]:


print (model.predict(x_test)[0:5])

