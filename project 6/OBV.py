"""
On Balance Volume (OBV)
Params:
    data: pandas DataFrame
    trend_periods: the over which to calculate OBV
	close_col: the name of the CLOSE values column
	vol_col: the name of the VOL values column
"""

def on_balance_volume(data, trend_periods=21, close_col='<CLOSE>', vol_col='<VOL>'):
    for index, row in data.iterrows():
        if index > 0:
            last_obv = data.at[index - 1, 'obv']
            if row[close_col] > data.at[index - 1, close_col]:
                current_obv = last_obv + row[vol_col]
            elif row[close_col] < data.at[index - 1, close_col]:
                current_obv = last_obv - row[vol_col]
            else:
                current_obv = last_obv
        else:
            last_obv = 0
            current_obv = row[vol_col]

        data.at[index, 'obv'] = current_obv

    #data['obv_ema' + str(trend_periods)] = data['obv'].ewm(ignore_na=False, min_periods=0, com=trend_periods,
                                                           #adjust=True).mean()

    return data