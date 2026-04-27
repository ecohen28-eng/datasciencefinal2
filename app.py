import numpy as np
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from dash import html
import plotly.io as pio
import yfinance as yf

# DATA
Y = pd.read_csv("LLY Data.csv")
Macro = pd.read_csv("2025-Historic_Domestic.csv")
baseline = pd.read_csv("baseline2025.csv")
bad = pd.read_csv("bad2025.csv")
external_stylesheets = [
   'https://googleapis.com'
]

def get_live_lly():
   data = yf.download("LLY", period="729d", interval="1d")

   # flatten columns
   data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

   data.reset_index(inplace=True)

   data["LLY_pct_change"] = data["Close"].pct_change()

   return data
   
Y_live = get_live_lly()

# TEMPLATE
lilly_template = dict(
   layout=dict(
       # Backgrounds
       paper_bgcolor="#FAEBE4", 
       plot_bgcolor="#FDDEDE",   

       # Font styling
       font=dict(
           family="Roboto, sans-serif",
           color="#6E0000"
       ),

       # Title styling
       title=dict(
           font=dict(
               family="EB Garamond, serif",
               size=24,
               color="#6E0000"
           ),
           x=0.5,
           xanchor="center"
       ),

       # Axes styling
       xaxis=dict(
         showgrid=True,
         gridcolor="#F2C6C2",
         linecolor="#6E0000",
         tickfont=dict(color="#6E0000"),
         title=dict(
           font=dict(color="#6E0000")
           )
       ),

       yaxis=dict(
         showgrid=True,
         gridcolor="#F2C6C2",
         linecolor="#6E0000",
         tickfont=dict(color="#6E0000"),
         title=dict(
           font=dict(color="#6E0000")
         )
       ),

       # Color palette for lines
       colorway=[
           "#D00000",  # Lilly red
           "#6E0000",  # dark red
           "#FF6B6B",  # lighter red
           "#A83232",  # muted red
           "#333333"   # neutral contrast
       ],

       # Legend styling
       legend=dict(
           bgcolor="rgba(0,0,0,0)",
           bordercolor="#6E0000"
       )
   )
)

pio.templates["lilly"] = lilly_template

# PROCESSING
Y["Date"] = pd.to_datetime(Y["Date"])
Y["year_month"] = Y["Date"].dt.to_period("M")
last_days = Y.groupby("year_month").Date.max().reset_index()
Y = Y[Y["Date"].isin(last_days["Date"])]

Y = Y[Y["Date"].dt.month.isin([3,6,9,12])]
Y["quarter"] = Y["Date"].dt.to_period("Q")
Y["quarter1"] = Y["quarter"].astype(str).str.replace('Q', ' Q')

def convert_quarter(q):
   y, qtr = q.split()
   m = {"Q1":3,"Q2":6,"Q3":9,"Q4":12}
   return datetime.datetime(int(y), m[qtr], 30)

Y["Date1"] = Y["quarter1"].apply(convert_quarter)
Macro["Date1"] = Macro["Date"].apply(convert_quarter)

Merge = pd.merge(Y, Macro, on="Date1", how="outer").sort_values("Date1")

# cutoff
cutoff = datetime.datetime(2004,12,31)
Merge = Merge[Merge["Date1"] >= cutoff]

# returns
Merge["LLY_pct_change"] = Merge["Close"].pct_change()

# DASH APP
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(style={
   'backgroundColor': "#FAEBE4",
   'color': "#6E0000",
   'padding': '5px',
   'fontFamily': "'Roboto', sans-serif"
}, children=[

# TITLE AND SUMMARY
html.H1("Eli Lilly Economic Analysis Dashboard", style={'textAlign':'center', 'fontFamily':"'EB Garamond', serif", 'fontSize': '40px'}),
html.H2("Created by Ethan Cohen for Honors Data Science and AI with Mr. Pham", style={'textAlign':'center'}),

html.Div([
   html.H3("Welcome to the Eli Lilly Economic Analysis Dashboard"),
   html.P(
       "This analysis explores the relationship between macroeconomic conditions and LLY stock performance. "
       "Across multiple views, LLY exhibits a weak relationship with economic strength, suggesting "
       "defensive characteristics typical of pharmaceutical companies."
   ),
   html.P(
       "Healthcare demand remains relatively inelastic. During economic slowdowns, demand for essential medicines "
       "persists, and in some cases increases, as health outcomes worsen or are prioritized over discretionary spending."
   )
], style={'marginBottom':'30px'}),

dcc.Tabs([

# PRICE AND ECON
dcc.Tab(label="Price vs Economy (Closing vs GDP)", children=[

   html.Br(),

   html.Div([
       html.H4("Key Observation"),
       html.P(
           "LLY stock price trends upward over time while macro variables fluctuate cyclically. "
           "This suggests limited direct dependency on macro growth."
       )
   ]),

   dcc.Graph(
       figure=px.line(
           Merge, x="Date1", y="Close",
           title="LLY Stock Price (2004–Present)"
       ).update_layout(template="lilly")
   ),

   dcc.Graph(
       figure=px.line(
           Merge, x="Date1",
           y=["Real GDP growth","Nominal GDP growth"],
           title="Macroeconomic Growth Over Time"
       ).update_layout(template="lilly")
   )
]),

# CORRELATION
dcc.Tab(label="LLY's Relationship with the Economy", children=[

   html.Br(),

   html.Div([
       html.H4("Interpretation"),
       html.P(
           "Scatter relationships show weak correlation between economic growth and LLY price. "
           "This supports the idea that LLY behaves as a defensive stock."
       ),
       html.P(
           "In weaker economic environments, healthcare spending is preserved, while other sectors decline. "
           "This relative stability can lead to outperformance."
       )
   ]),

   dcc.Dropdown(
       id="macro-x",
       options=[
           {"label":"Real GDP","value":"Real GDP growth"},
           {"label":"Disposable Income","value":"Real disposable income growth"},
       ],
       value="Real GDP growth",
       style={'color':'black'}
   ),

   dcc.Graph(id="scatter")
]),

# RETURNS
dcc.Tab(label="Returns vs GDP", children=[

   html.Br(),

   html.Div([
       html.H4("Analysis"),
       html.P(
           "Comparing percentage changes highlights how LLY responds to economic shocks. "
           "The relationship is inconsistent and weak in most places, yet slightly inverse where a relationship is present (Corr = -0.220, P=0.0497 for the 2nd half of the data under %change, 0.364 with the first half P=0.0201), reinforcing defensive behavior. Furthermore, when GDP drops, it can be noted that LLY is more likely to rise."
       )
   ]),

   dcc.Graph(
   figure=go.Figure(
       data=[
           go.Scatter(
               x=Merge["Date1"],
               y=Merge["LLY_pct_change"],
               name="LLY Returns",
               yaxis="y1"
           ),
           go.Scatter(
               x=Merge["Date1"],
               y=Merge["Real GDP growth"],
               name="Real GDP Growth",
               yaxis="y2"
           )
       ],
       layout=go.Layout(
           title="LLY Returns vs GDP Growth",
           template="lilly",
           xaxis=dict(title="Date"),


           yaxis=dict(
               title="LLY Returns",
               side="left"
           ),

           yaxis2=dict(
               title="GDP Growth",
               overlaying="y",
               side="right"
           )
       )
   )
)
]),

# FORECAST
dcc.Tab(label="Forecast", children=[

   html.Br(),

   html.Div([
       html.H4("Scenario Interpretation"),
       html.P(
           "Forecast scenarios simulate future stock paths based on economic conditions. "
           "Even under adverse economic assumptions, projected declines are limited, "
           "reflecting resilience of healthcare demand. It is noteable that the 'adverse' "
           "scenario outperforms the 'Baseline' version when GDP is used. This is because economic decline (through less economic output)"
           " typically is an indicator of worse healh, despite medicinal spending not dropping."
       )
   ]),

   dcc.Dropdown(
       id="model-var",
       options=[
           {"label":"Real GDP","value":"Real GDP growth"},
           {"label":"Nominal GDP","value":"Nominal GDP growth"},
           {"label":"Unemployment","value":"Unemployment rate"},
           {"label":"Inflation","value":"CPI inflation rate"},
       ],
       value="Real GDP growth",
       style={'color':'black'}
   ),

   dcc.Graph(id="forecast")
]),

# LIVEDATA

dcc.Tab(label="Live Market Terminal", children=[

    html.Div([
        html.H2("LLY Real-Time Market Feed",
                style={"textAlign": "center", "color": "#6E0000"}),

        html.P("Live institutional-grade price feed powered by Yahoo Finance",
               style={"textAlign": "center"}),

    ], style={"marginBottom": "20px"}),

    # KPI STRIP (LIVE PRICE + CHANGE)
    html.Div([
        html.Div([
            html.H4("Live Price"),
            html.H2(f"${float(Y_live['Close'].iloc[-1]):.2f}")
        ], style={
            "backgroundColor": "#FDDEDE",
            "padding": "15px",
            "borderRadius": "12px",
            "width": "48%",
            "textAlign": "center"
        }),

        html.Div([
            html.H4("Daily Change"),
            html.H2(f"{float(Y_live['LLY_pct_change'].iloc[-1] * 100):.2f}%")
        ], style={
            "backgroundColor": "#FDDEDE",
            "padding": "15px",
            "borderRadius": "12px",
            "width": "48%",
            "textAlign": "center"
        }),

    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "20px"}),

    # MAIN LIVE CHART
    dcc.Graph(
        figure=px.line(
            Y_live,
            x="Date",
            y="Close",
            title="LLY Live Price (5Y + Real-Time Feed)"
        ).update_layout(template="lilly")
    ),

    # VOLUME + SECOND VIEW
    dcc.Graph(
        figure=px.line(
            Y_live,
            x="Date",
            y="Volume",
            title="Trading Volume (Liquidity)"
        ).update_layout(template="lilly")
    )
])
])
])

# CALLBACKS
@app.callback(
   Output("scatter","figure"),
   Input("macro-x","value")
)
@app.callback(
   Output("scatter","figure"),
   Input("macro-x","value")
)
def update_scatter(x):
    df = Merge[[x, "Close"]].dropna()
    model = LinearRegression()
    model.fit(df[[x]], df["Close"])
    df["trend"] = model.predict(df[[x]])
    r2 = model.score(df[[x]], df["Close"])
    slope = model.coef_[0]
    intercept = model.intercept_
    corr = df[x].corr(df["Close"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df["Close"],
        mode="markers",
        name="Data"
    ))
    fig.add_trace(go.Scatter(
        x=df[x],
        y=df["trend"],
        mode="lines",
        name="Regression Line"
    ))
    fig.update_layout(
        template="lilly",
        title=f"{x} vs LLY Price"
    )
    fig.add_annotation(
        text=(
            f"R²: {r2:.3f}<br>"
            f"Slope: {slope:.3f}<br>"
            f"Intercept: {intercept:.3f}<br>"
            f"Correlation: {corr:.3f}"
        ),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="#6E0000"
    )
    return fig

@app.callback(
   Output("forecast","figure"),
   Input("model-var","value")
)
def forecast(var):

   df = Merge.dropna(subset=["LLY_pct_change", var])

   X = df[[var]]
   y = df["LLY_pct_change"]

   imp = SimpleImputer()
   X = imp.fit_transform(X)

   model = LinearRegression().fit(X,y)

   last_close = Merge["Close"].dropna().iloc[-1]
   last_date = Merge["Date1"].dropna().iloc[-1]

   def run_scenario(data):
       Xf = imp.transform(data[[var]])
       data["pct"] = model.predict(Xf)
       data["Close"] = last_close * (1 + data["pct"]).cumprod()

       data["Date1"] = pd.date_range(
           start=last_date + pd.offsets.QuarterEnd(),
           periods=len(data),
           freq="QE"
       )
       return data

   good = run_scenario(baseline.copy())
   bad_ = run_scenario(bad.copy())

   fig = go.Figure()

   fig.add_trace(go.Scatter(
       x=Merge["Date1"],
       y=Merge["Close"],
       name="Historical"
   ))

   fig.add_vline(x=last_date, line_dash="dash")

   fig.add_trace(go.Scatter(
       x=good["Date1"],
       y=good["Close"],
       name="Baseline"
   ))

   fig.add_trace(go.Scatter(
       x=bad_["Date1"],
       y=bad_["Close"],
       name="Adverse"
   ))

   fig.update_layout(
       template="lilly",
       title=f"Forecast based on {var}"
   )

   return fig
# END THE APP
if __name__ == "__main__":
   app.run(debug=False, port=8093)
