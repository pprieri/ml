train = []

##read data and 
df_train = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv')
df_test = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv')

for df in [df_train,df_test]:
    df['time']=pd.to_datetime(df['time'],format='%Y-%m-%d %H:%M:%S')
    # df.set_index('time',inplace=True)
    # df=df.sort_index()

##check we have the same amount of rows per date
train.groupby('date').size().sort_values(ascending=True)
test.groupby('date').size().sort_values(ascending=True)
##easier: train.date.value_counts()

##Basic Feat Eng
for df in [df_train,df_test]:
    df['hour']=df['time'].dt.time
    df['day_of_week']=df['time'].dt.dayofweek
    df['name_day_week']=df['time'].dt.day_name()
    df['day_of_month']=df['time'].dt.day
    df['relative_day']=df['time'].dt.dayofyear-90
    df['relative_month']=df['time'].dt.month-3
    df['relative_week']=df['time'].dt.isocalendar().week-13

## Define Holidays
memorial_day='1991-05-27'
independence_day='1991-07-04'
labor_day='1991-09-02'
holidays=[memorial_day,independence_day,labor_day]
df['holiday']=((df.time.dt.date.astype('str').isin(holidays))).astype('int')

##Daily mean
TARGET = ''
df_train.groupby('relative_day')[TARGET].mean().plot(figsize=(20,6))

##Weekly mean
df_train.groupby('relative_week')[TARGET].mean().plot(figsize=(20,6))

## Graph x number of days
#Set up number of days to show

nb_days=30
df_daily_mean=df_train.groupby('relative_day')[TARGET].mean().values[:nb_days]
# nb_values=1000
# df_hourly_mean = df_train.groupby('time')['congestion'].mean().values[:nb_values]

plt.figure(figsize=(20, 6))
xticks = pd.date_range(start=df_train.time.dt.date.min(), end=df_train.time.dt.date.max(),freq='D')
plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
plt.plot(xticks[:nb_days],df_daily_mean, linewidth=1)
plt.xlabel("Date")
plt.ylabel(f"{TARGET}")
# Tweak spacing to prevent clipping of tick-labels
plt.subplots_adjust(bottom=0.15)
plt.show()