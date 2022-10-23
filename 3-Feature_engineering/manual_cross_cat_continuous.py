
#Manual Cross Categorical * Continuous:
   
   oh_cat = pd.get_dummies(df[cat],prefix=continuous+'*'+cat)
   df = df.join(oh_cat.mul(df[continuous],axis=0))

#as a function
def cross_categorical(df,continuous,cat):

    df = df.copy()
    X_new = pd.get_dummies(df[cat],prefix=continuous+'*'+cat)
    crossed_columns = X_new.columns.to_list()
    df = df.join(X_new.mul(df[continuous],axis=0))
    
    return  df, crossed_columns



