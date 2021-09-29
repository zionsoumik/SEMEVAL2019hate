import pandas as pd
startdate = "10/10/2011"
print(pd.to_datetime(startdate) + pd.DateOffset(days=5))