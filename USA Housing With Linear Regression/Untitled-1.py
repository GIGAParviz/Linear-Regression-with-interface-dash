import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

o = pd.read_csv('USA_Housing.csv')
o[o.columns[:-2]]

from sklearn.model_selection import train_test_split
X = o[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = o['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression


lm = LinearRegression()
lm.fit(X_train,y_train)

# print(lm.intercept_)

# print(lm.coef_)


coef_df = pd.DataFrame(lm.coef_, X.columns , columns=['Coefficient'])
 
# print(coef_df)


d = lm.predict(X_test)
# print(d)


# plt.scatter(y_test , d)
x= [0 , 2_500_000]
y =x
# plt.plot(x , y , 'r')
# plt.show()

# a = np.array(y_test).reshape(-1 ,1)
# # print(a)

# mp = np.array(d).reshape(-1,1)
# print (mp)





from dash import Dash,dcc , Output, Input # pip install dash
import dash_bootstrap_components as dbc # pip install dash-bootstrap-components
import dash_html_components as html # pip install dash-html-components
import os
import plotly.graph_objects as go # pip install plotly
import plotly.express as px



app = Dash(__name__,external_stylesheets=[dbc.themes.CYBORG])

mydiv = html.H1("USA Housing" , style={'color' : 'red' ,'font-size' :'50pz' })

figure = go.Figure(
    # data=go.Scatter(x=list(range(0,10)), y=list(range(1,11)),mode='markers'))
    data=go.Scatter(x=d, y=y_test , mode='markers' ),

    # fig = px.scatter(o, x=d, y=y_test, opacity=0.65),
)
# fig.show()
figure.add_traces(go.Scatter(x=x, y=y, name='Regression Fit' ))

my_graph = dcc.Graph(figure=figure)

text = dbc.Row([mydiv],class_name='text-center')
app.layout = dbc.Container([text , my_graph ])


if __name__=='__main__':
    app.run_server(port='8000')

import plotly.express as px


from sklearn import metrics


# print('MAE:', metrics.mean_absolute_error(y_test, d))
# print('MSE' , metrics.mean_squared_error(y_test , d))
# print('RMSE' , np.sqrt(metrics.mean_squared_error(y_test , d)))

print(o['Price'].mean())

print(metrics.mean_absolute_error(y_test, d)/ o['Price'].mean()*100)