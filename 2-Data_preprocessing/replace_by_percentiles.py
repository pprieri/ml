def replace_by_percentile(df,col,min_percentile=1,max_percentile=99):

    df[col]=df[col].copy()

    low_threshold=np.percentile(df[col],min_percentile)
    max_threshold=np.percentile(df[col],max_percentile)

    return np.where(df[col]<low_threshold,low_threshold,np.where(df[col]>max_threshold,max_threshold,df[col]))

#replace_by_percentile(df,col,min_percentile=1,max_percentile=99)
