import numpy as np
import pandas as pd

comps = pd.read_csv("/kaggle/input/meta-kaggle/Competitions.csv")
evaluation = ['EvaluationAlgorithmAbbreviation',
              'EvaluationAlgorithmName',
              'EvaluationAlgorithmDescription',]

compt = ['Title', 'EnabledDate', 'HostSegmentTitle']

df = comps[compt + evaluation].copy()

df['year'] = pd.to_datetime(df.EnabledDate).dt.year.values
df['comps'] = 1
time_select = df.year >= 2015
competition_type_select = df.HostSegmentTitle.isin(['Featured', 'Research'])

pd.pivot_table(df[time_select&competition_type_select],
                    values='comps',
                    index=['EvaluationAlgorithmAbbreviation'],
                    columns=['year'],
                    fill_value=0.0,
                    aggfunc=np.sum,
                    margins=True
                    ).sort_values(
                        by=('All'), ascending=False).iloc[1:,:].head(20)
metric = 'AUC'
metric_select = df['EvaluationAlgorithmAbbreviation']==metric
print(df[time_select&competition_type_select&metric_select][['Title', 'year']])

counts = (df[time_select&competition_type_select]
            .groupby('EvaluationAlgorithmAbbreviation'))
total_comps_per_year = (df[time_select&competition_type_select]
                        .groupby('year').sum())
single_metrics_per_year = (counts.sum()[counts.sum().comps==1]
                            .groupby('year').sum())
table = (total_comps_per_year.rename(columns={'comps': 'n_comps'})
            .join(single_metrics_per_year / total_comps_per_year))

print(table)

print(counts.sum()[counts.sum().comps==1].index.values)
