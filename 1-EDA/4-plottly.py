feature = ''
hue = '' 
col = '' 
row = ''
extra_feat = ''
extra_feat_2 = ''

#simple plottly
fig = px.scatter(df, y=target, x=feature, color=hue)
fig.update_traces(marker_size=10)
fig.show()

#A) adding texts, labels, color

df_mini = df.copy()

fig = px.scatter(df_mini,
                y=target,
                x='EDB chassis height front 6t rear 9t',
                color="Rated speed", #'NS 4FW', #,'Gearbox syst txt
                # facet_col='Tandem axle', #'Rated speed','Gearbox ratio', #add facets
                # facet_row='Rated speed',
                category_orders={"Rated speed":["1600.0","1700.0","1800.0","1900.0"]}, #determine orders of facets (if col is float, need to convert it to str first)
                hover_name=df_mini.index, #add ROW ID
                hover_data=['Chassi ID','Rated speed','Tandem axle','NS 4FW','Gearbox ratio','Gearbox syst txt','Tot RAT in ER','Pow_abs(T)_L (kW)','Engine in ER txt','Model type txt'], #we need to feed a list, otherwise it won't work
                labels={'EDB chassis height front 6t rear 9t':'EDB Chassis height',target:'SPL'},
                color_continuous_scale='fall' #for continuous
                # color_discrete_map=[{1:"red",0:"green"}] #for categorical

fig.update_traces(marker_size=8)

fig.update_layout(
    autosize=True,
    width=800,
    height=600)
fig.show()

#B) 2 figures with 2 datasets

import plotly.express as px
import plotly.graph_objects as go

df_mini = df.copy()
# df_mini = df_mini[df_mini['Chassi ID']=='A-802622']

fig1 = px.scatter(df_mini,
                y=target,
                x='EDB chassis height front 6t rear 9t',
                color="Rated speed", #'NS 4FW', #,'Gearbox syst txt
                # facet_col='Tandem axle', #'Rated speed','Gearbox ratio', #add facets
                # facet_row='Rated speed',
                category_orders={"Rated speed":["1600.0","1700.0","1800.0","1900.0"]}, #determine orders of facets (if col is float, need to convert it to str first)
                hover_name=df_mini.index, #add ROW ID
                hover_data=['Chassi ID','Rated speed','Tandem axle','NS 4FW','Gearbox ratio','Gearbox syst txt','Tot RAT in ER','Pow_abs(T)_L (kW)'], #we need to feed a list, otherwise it won't work
                labels={'EDB chassis height front 6t rear 9t':'EDB Chassis height',target:'SPL'},
                color_continuous_scale='fall', #for continuous
                symbol=['circle']*len(df_mini)
                # color_discrete_map=[{1:"red",0:"green"}] #for categorical
)

fig1.update_traces(marker_size=4)

fig2 = px.scatter(df_mini2,
                y=target,
                x='EDB chassis height front 6t rear 9t',
                color="Rated speed", #'NS 4FW', #,'Gearbox syst txt
                # facet_col='Tandem axle', #'Rated speed','Gearbox ratio', #add facets
                # facet_row='Rated speed',
                category_orders={"Rated speed":["1600.0","1700.0","1800.0","1900.0"]}, #determine orders of facets (if col is float, need to convert it to str first)
                hover_name=df_mini2.index, #add ROW ID
                hover_data=['Chassi ID','Rated speed','Tandem axle','NS 4FW','Gearbox ratio','Gearbox syst txt','Tot RAT in ER','Pow_abs(T)_L (kW)'], #we need to feed a list, otherwise it won't work
                labels={'EDB chassis height front 6t rear 9t':'EDB Chassis height',target:'SPL'},
                color_continuous_scale='gray', #for continuous
                symbol_sequence = ['x']*len(df_mini2)
                # color_discrete_map=[{1:"red",0:"green"}] #for categorical
)

fig2.update_traces(marker_size=8)

fig3 = go.Figure(data=fig1.data + fig2.data)

fig3.update_layout(
    autosize=True,
    width=800,
    height=600)

fig3.show()