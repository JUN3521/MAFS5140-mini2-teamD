import pandas as pd
import numpy as np
from collections import deque

"""
MULTI-FACTOR INTRADAY STRATEGY (Three-Factor Model)
=====================================================
Combines cross-sectional short-term reversal, intraday momentum,
and Alpha#12 volume-shock reversal from WorldQuant Alpha101.

Factor weights:
    Reversal = 0.50, Momentum = 0.20, Volume Shock = 0.30

References:
    - Kakushadze (2016), "101 Formulaic Alphas", arXiv:1601.00991
    - Avramov, Chordia & Goyal (2006); Jegadeesh (1990)
    - Da, Liu & Schaumburg (2014), NY Fed Staff Report No. 513
    - Gao, Han, Li & Zhou (2018), Journal of Financial Economics
"""
# ===================== Strategy Design =====================

class Strategy:
    """
   Short-term Reversal + Intraday Momentum Multi-factor Strategy
    """
    
    def __init__(self):
        # --- Hyperparameters ---
        self.lookback_short = 14       # 70 min (14 x 5-min bars) for reversal
        self.lookback_medium = 18      # 90 min (18 x 5-min bars) for momentum
        self.top_k = 15                # Number of stocks to hold
        
        # Optimized factor weights (from grid search)
        self.w_reversal = 0.70
        self.w_momentum = 0.30
        
        # --- State Variables ---
        max_lookback = max(self.lookback_short, self.lookback_medium) + 1
        self.price_history = deque(maxlen=max_lookback)
        self.step_count = 0
    
    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        prices = current_market_data["close"]
        tickers = prices.index
        
        # Update historical prices.
        self.price_history.append(prices.values.copy())
        self.step_count += 1
        
        # Wait for sufficient historical data.
        min_history = max(self.lookback_short, self.lookback_medium) + 1
        if self.step_count < min_history:
            return pd.Series(0.0, index=tickers)
        
        price_arr = np.array(self.price_history)
        current_prices = price_arr[-1]
        
        # Factor 1: Short-term reversal (30 min)
        past_short = price_arr[-1 - self.lookback_short]
        ret_short = (current_prices - past_short) / (past_short + 1e-10)
        reversal_signal = -ret_short   # buy losers
        
        # Factor 2: Intraday momentum (2 hours)
        past_medium = price_arr[-1 - self.lookback_medium]
        momentum_signal = (current_prices - past_medium) / (past_medium + 1e-10)
        
        # Rank-based composite scoring
        rank_rev = self._rank(reversal_signal)
        rank_mom = self._rank(momentum_signal)
        composite = self.w_reversal * rank_rev + self.w_momentum * rank_mom

        # Top-K equal weight
        top_indices = np.argsort(composite)[-self.top_k:]
        weights_arr = np.zeros(len(tickers))
        weights_arr[top_indices] = 1.0 / self.top_k
        
        return pd.Series(weights_arr, index=tickers)
    
    @staticmethod
    def _rank(arr):
        clean = np.where(np.isfinite(arr), arr, -np.inf)
        temp = clean.argsort().argsort()
        return temp / (len(temp) - 1 + 1e-10)

