"""
`python/macro/sentiment.py`

**Macro Sentiment & Economic Data Ingestion**

Ingests and normalizes macroeconomic indicators:
- Fear & Greed Index
- CPI (Consumer Price Index)
- DXY (Dollar Index)
- Interest rate decisions

Optimization Strategy:
- Lightweight REST polling with caching
- Z-score normalization for RL agent consumption
- Graceful handling of API rate limits and network timeouts
- AMD ROCm/DirectML environment checks for future GPU acceleration
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import numpy as np
import polars as pl

# Check for AMD ROCm/DirectML availability
AMD_GPU_AVAILABLE = (
    os.environ.get("ROCM_PATH") is not None or
    os.environ.get("DIRECTML_ENABLED") == "1"
)


class SentimentLevel(Enum):
    """Sentiment classification levels"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class FearGreedData:
    """Fear & Greed Index snapshot"""
    timestamp: datetime
    value: int  # 0-100
    classification: SentimentLevel
    previous_value: Optional[int] = None
    previous_close: Optional[int] = None
    one_week_avg: Optional[float] = None
    one_month_avg: Optional[float] = None


@dataclass
class EconomicIndicator:
    """Generic economic indicator"""
    name: str
    timestamp: datetime
    actual: float
    forecast: Optional[float] = None
    previous: Optional[float] = None
    impact: str = "medium"  # low, medium, high
    currency: str = "USD"


@dataclass
class MacroSnapshot:
    """Composite macroeconomic snapshot"""
    timestamp: datetime
    fear_greed: Optional[FearGreedData] = None
    dxy: Optional[float] = None
    cpi_yoy: Optional[float] = None
    interest_rate: Optional[float] = None
    vix: Optional[float] = None
    z_scores: Dict[str, float] = field(default_factory=dict)


class SentimentAnalyzer:
    """
    Fetches and analyzes market sentiment indicators.
    
    Primary source: Alternative.me Fear & Greed Index
    """
    
    API_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, cache_ttl_seconds: int = 3600):
        self.cache_ttl = cache_ttl_seconds
        self.cache: Optional[Dict] = None
        self.cache_time: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "NautilusBot/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @staticmethod
    def classify_sentiment(value: int) -> SentimentLevel:
        """Classify F&G value into sentiment level"""
        if value <= 25:
            return SentimentLevel.EXTREME_FEAR
        elif value <= 45:
            return SentimentLevel.FEAR
        elif value <= 55:
            return SentimentLevel.NEUTRAL
        elif value <= 75:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED
    
    async def fetch_fear_greed(self) -> Optional[FearGreedData]:
        """Fetch current Fear & Greed Index"""
        # Check cache first
        if self.cache and self.cache_time:
            age = datetime.utcnow() - self.cache_time
            if age.total_seconds() < self.cache_ttl:
                return self._parse_fg_data(self.cache)
        
        try:
            async with self.session.get(self.API_URL, params={"limit": "7"}) as response:
                if response.status == 200:
                    data = await response.json()
                    self.cache = data
                    self.cache_time = datetime.utcnow()
                    return self._parse_fg_data(data)
                else:
                    print(f"Fear & Greed API error: {response.status}")
        except aiohttp.ClientError as e:
            print(f"Network error fetching F&G: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        return None
    
    def _parse_fg_data(self, data: Dict) -> Optional[FearGreedData]:
        """Parse Fear & Greed API response"""
        if "data" not in data or not data["data"]:
            return None
        
        current = data["data"][0]
        value = int(current["value"])
        
        # Calculate averages from historical data
        values = [int(d["value"]) for d in data["data"][:7]]
        one_week_avg = sum(values) / len(values) if values else None
        
        one_month_avg = None  # Would need more historical data
        
        return FearGreedData(
            timestamp=datetime.fromtimestamp(int(current["timestamp"])),
            value=value,
            classification=self.classify_sentiment(value),
            previous_value=int(data["data"][1]["value"]) if len(data["data"]) > 1 else None,
            previous_close=int(data["data"][1]["value"]) if len(data["data"]) > 1 else None,
            one_week_avg=one_week_avg,
            one_month_avg=one_month_avg,
        )


class EconomicDataFetcher:
    """
    Fetches macroeconomic indicators from various sources.
    
    Sources:
    - DXY: Federal Reserve Economic Data (FRED) or similar
    - CPI: Bureau of Labor Statistics or aggregated APIs
    - Interest rates: Central bank announcements
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # Be respectful to free APIs
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_dxy(self) -> Optional[float]:
        """Fetch Dollar Index (DXY) value"""
        # Example using a financial data API
        url = "https://api.example.com/dxy"  # Replace with actual API
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("value")
                await self._handle_rate_limit(response)
        except Exception as e:
            print(f"Error fetching DXY: {e}")
        
        return None
    
    async def fetch_cpi(self, country: str = "US") -> Optional[EconomicIndicator]:
        """Fetch CPI data"""
        # Example using FRED API or similar
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": "CPIAUCSL",  # US CPI All Urban Consumers
            "api_key": self.api_keys.get("fred", ""),
            "file_type": "json",
            "limit": 1,
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    observations = data.get("observations", [])
                    if observations:
                        latest = observations[-1]
                        return EconomicIndicator(
                            name="CPI",
                            timestamp=datetime.strptime(latest["date"], "%Y-%m-%d"),
                            actual=float(latest["value"]),
                            impact="high",
                        )
                await self._handle_rate_limit(response)
        except Exception as e:
            print(f"Error fetching CPI: {e}")
        
        return None
    
    async def fetch_vix(self) -> Optional[float]:
        """Fetch VIX (Volatility Index)"""
        url = "https://api.example.com/vix"  # Replace with actual API
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("value")
        except Exception as e:
            print(f"Error fetching VIX: {e}")
        
        return None
    
    async def _handle_rate_limit(self, response):
        """Handle API rate limiting gracefully"""
        if response.status == 429:
            retry_after = int(response.headers.get("Retry-After", self.rate_limit_delay))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            await asyncio.sleep(retry_after)


class MacroSentimentEngine:
    """
    Combines multiple macro indicators into unified signals.
    
    Normalizes all inputs to Z-scores for consistent RL agent consumption.
    """
    
    def __init__(self, api_keys: Dict[str, str], lookback_days: int = 30):
        self.api_keys = api_keys
        self.lookback_days = lookback_days
        self.history: List[MacroSnapshot] = []
        self.sentiment_analyzer = SentimentAnalyzer()
        self.econ_fetcher = EconomicDataFetcher(api_keys)
        
        # Rolling statistics for Z-score calculation
        self.dxy_history: List[float] = []
        self.vix_history: List[float] = []
        self.fg_history: List[int] = []
        
    async def __aenter__(self):
        await self.sentiment_analyzer.__aenter__()
        await self.econ_fetcher.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.sentiment_analyzer.__aexit__(exc_type, exc_val, exc_tb)
        await self.econ_fetcher.__aexit__(exc_type, exc_val, exc_tb)
    
    async def fetch_macro_snapshot(self) -> MacroSnapshot:
        """Fetch all macro indicators and create snapshot"""
        snapshot = MacroSnapshot(timestamp=datetime.utcnow())
        
        # Fetch all indicators concurrently
        fg_task = self.sentiment_analyzer.fetch_fear_greed()
        dxy_task = self.econ_fetcher.fetch_dxy()
        cpi_task = self.econ_fetcher.fetch_cpi()
        vix_task = self.econ_fetcher.fetch_vix()
        
        results = await asyncio.gather(fg_task, dxy_task, cpi_task, vix_task, return_exceptions=True)
        
        snapshot.fear_greed = results[0] if isinstance(results[0], FearGreedData) else None
        snapshot.dxy = results[1] if isinstance(results[1], float) else None
        snapshot.cpi_yoy = results[2].actual if isinstance(results[2], EconomicIndicator) else None
        snapshot.vix = results[3] if isinstance(results[3], float) else None
        
        # Update history for Z-score calculations
        self._update_history(snapshot)
        
        # Calculate Z-scores
        snapshot.z_scores = self.calculate_z_scores()
        
        # Store snapshot
        self.history.append(snapshot)
        if len(self.history) > self.lookback_days:
            self.history.pop(0)
        
        return snapshot
    
    def _update_history(self, snapshot: MacroSnapshot):
        """Update rolling history for statistical calculations"""
        if snapshot.dxy is not None:
            self.dxy_history.append(snapshot.dxy)
            if len(self.dxy_history) > self.lookback_days:
                self.dxy_history.pop(0)
        
        if snapshot.vix is not None:
            self.vix_history.append(snapshot.vix)
            if len(self.vix_history) > self.lookback_days:
                self.vix_history.pop(0)
        
        if snapshot.fear_greed is not None:
            self.fg_history.append(snapshot.fear_greed.value)
            if len(self.fg_history) > self.lookback_days:
                self.fg_history.pop(0)
    
    def calculate_z_scores(self) -> Dict[str, float]:
        """
        Calculate Z-scores for each macro indicator.
        
        Z-score = (current_value - mean) / std_dev
        
        Uses Polars for efficient computation when available.
        """
        z_scores = {}
        
        # DXY Z-score
        if len(self.dxy_history) >= 5:
            mean = np.mean(self.dxy_history)
            std = np.std(self.dxy_history)
            if std > 0:
                z_scores["dxy"] = (self.dxy_history[-1] - mean) / std
            else:
                z_scores["dxy"] = 0.0
        
        # VIX Z-score
        if len(self.vix_history) >= 5:
            mean = np.mean(self.vix_history)
            std = np.std(self.vix_history)
            if std > 0:
                z_scores["vix"] = (self.vix_history[-1] - mean) / std
            else:
                z_scores["vix"] = 0.0
        
        # Fear & Greed Z-score (inverted: high F&G = potentially overbought)
        if len(self.fg_history) >= 5:
            mean = np.mean(self.fg_history)
            std = np.std(self.fg_history)
            if std > 0:
                # Invert so high F&G gives negative Z (caution signal)
                z_scores["fear_greed"] = -(self.fg_history[-1] - mean) / std
            else:
                z_scores["fear_greed"] = 0.0
        
        return z_scores
    
    def create_composite_signal(self, snapshot: MacroSnapshot) -> Dict[str, Any]:
        """
        Create a composite macro signal for the RL agent.
        
        Returns a normalized signal between -1 (bearish) and 1 (bullish).
        """
        if not snapshot.z_scores:
            return {"signal": 0.0, "confidence": 0.0, "components": {}}
        
        components = {}
        weights = {
            "dxy": 0.25,      # Strong dollar = headwind for crypto
            "vix": 0.25,      # High vol = risk off
            "fear_greed": 0.3, # Extreme greed = caution
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        # DXY: High dollar is negative for crypto
        if "dxy" in snapshot.z_scores:
            components["dxy_signal"] = -snapshot.z_scores["dxy"] * 0.5
            weighted_sum += components["dxy_signal"] * weights["dxy"]
            total_weight += weights["dxy"]
        
        # VIX: High volatility is negative
        if "vix" in snapshot.z_scores:
            components["vix_signal"] = -snapshot.z_scores["vix"] * 0.5
            weighted_sum += components["vix_signal"] * weights["vix"]
            total_weight += weights["vix"]
        
        # Fear & Greed: Extreme greed is negative (contrarian)
        if "fear_greed" in snapshot.z_scores:
            components["fg_signal"] = snapshot.z_scores["fear_greed"] * 0.5
            weighted_sum += components["fg_signal"] * weights["fear_greed"]
            total_weight += weights["fear_greed"]
        
        # Normalize
        if total_weight > 0:
            signal = weighted_sum / total_weight
        else:
            signal = 0.0
        
        # Clamp to [-1, 1]
        signal = max(-1.0, min(1.0, signal))
        
        # Confidence based on data availability
        confidence = len([k for k in snapshot.z_scores.keys()]) / 3.0
        
        return {
            "signal": signal,
            "confidence": confidence,
            "components": components,
            "raw_z_scores": snapshot.z_scores,
        }
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert history to Polars DataFrame for analysis"""
        if not self.history:
            return pl.DataFrame()
        
        data = {
            "timestamp": [s.timestamp for s in self.history],
            "dxy": [s.dxy for s in self.history],
            "vix": [s.vix for s in self.history],
            "cpi_yoy": [s.cpi_yoy for s in self.history],
            "fear_greed": [s.fear_greed.value if s.fear_greed else None for s in self.history],
        }
        
        # Add Z-scores
        for key in ["dxy", "vix", "fear_greed"]:
            data[f"z_{key}"] = [s.z_scores.get(key) for s in self.history]
        
        return pl.DataFrame(data)


async def main():
    """Example usage of macro sentiment engine"""
    api_keys = {
        "fred": os.environ.get("FRED_API_KEY", ""),
    }
    
    async with MacroSentimentEngine(api_keys) as engine:
        snapshot = await engine.fetch_macro_snapshot()
        
        print(f"Fear & Greed: {snapshot.fear_greed}")
        print(f"DXY: {snapshot.dxy}")
        print(f"VIX: {snapshot.vix}")
        print(f"Z-scores: {snapshot.z_scores}")
        
        composite = engine.create_composite_signal(snapshot)
        print(f"Composite signal: {composite['signal']:.4f} (confidence: {composite['confidence']:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
