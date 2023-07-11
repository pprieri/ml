#fizzbuzz 
for i in range(1,51):
    
    if i%3==0 and i%5==0:
        print("FizzBuzz")
    if i%3==0:
        print("Fizz")
    if i%5==0:
        print("Buzz")
    else:
        print(str(i))
        
#fix bug in data for a device that is halved for some reason

df.loc[df["device"]=="x"],'data'] = df.loc[df["device"]=="x"],'data']*2


#fix bug in data for 2 devices that is halved for some reason

df.loc[df["device"].isin("x","y")],'data'] = df.loc[df["device"]=="x"],'data']*2



film.sort_values(by='length',ascending=True)[['title']].head(5)

staff[staff.picture.isnull()][['first_name','last_name']].head(5)

payment['year'] = pd.to_datetime('payment_ts').dt.year
payment['month'] = pd.to_datetime('payment_ts').dt.month

payment.groupby(['year','month'])['amount'].sum().reset_index().sort_values(by=[''])



# %%

import pandas as pd

data = {
    'actor_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'first_name': ['PENELOPE', 'NICK', 'ED', 'JENNIFER', 'JOHNNY', 'BETTE', 'GRACE', 'MATTHEW', 'JOE', 'CHRISTIAN', 'ZERO', 'KARL', 'UMA', 'VIVIEN', 'CUBA', 'FRED', 'HELEN', 'DAN', 'BOB', 'LUCILLE'],
    'last_name': ['GUINESS', 'WAHLBERG', 'CHASE', 'DAVIS', 'LOLLOBRIGIDA', 'NICHOLSON', 'MOSTEL', 'JOHANSSON', 'SWANK', 'GABLE', 'CAGE', 'BERRY', 'WOOD', 'BERGEN', 'OLIVIER', 'COSTNER', 'VOIGHT', 'TORN', 'FAWCETT', 'TRACY']
}

actor = pd.DataFrame(data)
print(actor)


# %%
def myfunc():
  """
  1. Don't change this function name
  2. You must return a pandas dataframe type of object, other types won't work
  """
  def group_actors(row):
      if row['first_name'][0] == 'A':
          return 'a_actors'
      if row['first_name'][0] == 'B':
          return 'b_actors'
      if row['first_name'][0] == 'C':
          return 'c_actors'
      else:
          return 'other_actors'

  actor['group_actors'] = actor.apply(group_actors,axis=1)
  
  return actor.groupby('group_actors')['actor_id'].nunique().reset_index()

# %%
myfunc()

# %%
