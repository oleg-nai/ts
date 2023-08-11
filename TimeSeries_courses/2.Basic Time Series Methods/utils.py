class SimpleTSModel:
    def __init__(self, y):
        self.y = y.asfreq(y.index.inferred_freq)
        self.freq = self.y.index.freq

    def forecast(self, periods=None):
        start = pd.to_datetime(self.y.index.max())
        end = pd.to_datetime(start + periods * (self.freq or 1))
        index = pd.date_range(start, end, freq=self.freq)
        return self.predict(start, end, index, periods)
    

class TSMean(SimpleTSModel):
    def predict(self, start, end, index, periods=None):
        m = self.y.mean()
        out = pd.Series(m, index=index)
        out = out.loc[start:].copy()
        return out
    
    
class TSNaive(SimpleTSModel):
    def predict(self, start, end, index, periods=None):
        out = pd.Series(self.y.iloc[-1], index=index)
        return out

    
class TSNaiveSeasonal(SimpleTSModel):
    def __init__(self, y, lag):
        super().__init__(y)
        self.lag = lag
    
    def predict(self, start, end, index, periods=None):
        out = pd.Series(np.nan, index)
        for idx in range(0, periods + 1):
            out.iloc[idx] = self.y.loc[index[idx] - self.freq * int(self.lag * (np.floor((idx - 1)/self.lag) + 1))]
        out.iloc[0] = self.y.iloc[-1]
        return out

class TSDrift(SimpleTSModel):
    def predict(self, start, end, index, periods=None):
        out  = pd.Series(np.nan, index)
        y_last = self.y.iloc[-1]
        y_first = self.y.iloc[0]
        ts_range = len(self.y)
        out.iloc[0] = y_last
        for idx in range(1, periods + 1):
            out.iloc[idx] = y_last + idx * ((y_last - y_first) / (ts_range - 1))
        return out.copy()