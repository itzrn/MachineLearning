import pandas as pd
df_stocks = pd.DataFrame({
    'tickers' : ['GOOGLE', 'WMT', 'NSFT'],
    'price' : [845, 65, 97],
    'pe':[30.37, 14.01, 30.97],
    'eps':[27.82, 4.61, 2.12]
})

df_weather = pd.read_excel("P03_weather_data.xlsx")

with pd.ExcelWriter("P08_stocks_weather.xlsx") as writer:
    df_stocks.to_excel(writer, sheet_name="stocks")
    df_weather.to_excel(writer, sheet_name="weather")


# there are many things still left to learn in writing and reading of csv and excel file
# which can be search on google
