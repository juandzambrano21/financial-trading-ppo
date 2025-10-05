#     FSRPPO: Financial Signal Representation Proximal Policy Optimization


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tested](https://img.shields.io/badge/tested-real%20data-green.svg)](https://github.com/psf/black)

A implementation of the FSRPPO trading strategy. **validated with real Yahoo Finance data.** Based on Wang & Wang's research paper combining advanced signal processing with proximal policy optimization.

##     Key Features

### Advanced Signal Processing
- **CEEMDAN**: Complete Ensemble Empirical Mode Decomposition with Adaptive Noise
- **ESMD**: Extreme-Point Symmetric Mode Decomposition  
- **Hurst Exponent**: Modified rescaled range analysis for market regime detection
- **FSR**: Integrated Financial Signal Representation pipeline

### State-of-the-Art RL Implementation
- **PPO**: Proximal Policy Optimization with advanced features
- **Actor-Critic Architecture**: Separate policy and value networks
- **GAE**: Generalized Advantage Estimation
- **Adaptive Learning**: Dynamic hyperparameter adjustment

### Real Market Integration
- **Yahoo Finance**: Automatic data fetching and preprocessing
- **Multiple Assets**: Support for stocks, ETFs, forex, and crypto
- **Real-time Data**: Live market data integration
- **Risk Management**: Advanced position sizing and risk controls

### Robust Features
- ** Backtesting**: Historical performance analysis
- **Hyperparameter Optimization**: Automated parameter tuning
- **Model Persistence**: Save and load trained models
- **Extensive Logging**: Detailed training and trading logs
- **Visualization**: Rich plotting and analysis tools

## 📦 Installation

### From Source
```bash
git clone https://github.com/juandzambrano21/financial-trading-ppo
cd fsrppo
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/juandzambrano21/financial-trading-ppo
cd fsrppo
pip install -e ".[dev,docs]"
```

## 🏃‍♂️ Quick Start

### Basic Usage
```python
import fsrppo
from fsrppo import FSRPPOTrainer, TradingEnvironment, MarketDataProvider

# Load market data
data_provider = MarketDataProvider()
data = data_provider.get_data("AAPL", start="2020-01-01", end="2023-01-01")

# Create trading environment
env = TradingEnvironment(data, initial_balance=10000)

# Initialize and train agent
trainer = FSRPPOTrainer(env)
agent = trainer.train(total_timesteps=100000)

# Backtest the strategy
results = trainer.backtest(test_data, agent)
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

### Advanced Configuration
```python
from fsrppo import PPOAgent, FinancialSignalRepresentation

# Configure signal processing
fsr_config = {
    'ceemdan_trials': 100,
    'esmd_M': 4,
    'hurst_window': 252,
    'noise_scale': 0.005
}

# Configure PPO agent
ppo_config = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5
}

# Create FSR processor
fsr = FinancialSignalRepresentation(**fsr_config)

# Create and train agent
agent = PPOAgent(env, fsr_processor=fsr, **ppo_config)
agent.learn(total_timesteps=500000)
```

##     Signal Processing Pipeline

The FSRPPO framework employs a sophisticated signal processing pipeline:

1. **CEEMDAN Decomposition**: Decomposes price signals into intrinsic mode functions (IMFs)
2. **ESMD Analysis**: Extracts extreme-point symmetric modes for trend analysis  
3. **Hurst Exponent**: Measures long-term memory and market regime
4. **Feature Engineering**: Combines processed signals into trading features

```python
from fsrppo.signal_processing import CEEMDAN, ESMD, hurst_exponent

# Process price data
ceemdan = CEEMDAN(trials=100, epsilon=0.005)
imfs = ceemdan.decompose(price_data)

esmd = ESMD(M=4, max_imfs=8)
esmd_result = esmd.decompose(time_series, price_data)

hurst_exp = hurst_exponent(price_data, window=252)
```

## 🎯 Trading Environment

The trading environment supports various market scenarios:

```python
from fsrppo import TradingEnvironment

env = TradingEnvironment(
    data=market_data,
    initial_balance=100000,
    transaction_cost=0.001,  # 0.1% transaction cost
    max_position_size=0.1,   # Max 10% position size
    lookback_window=60,      # 60-day lookback
    reward_scaling=1e-4,     # Reward scaling factor
    risk_free_rate=0.02      # 2% risk-free rate
)
```

##     Backtesting and Evaluation

 backtesting with detailed metrics:

```python
from fsrppo.utils import TradingMetrics

# Run backtest
backtest_results = trainer.backtest(
    test_data=test_data,
    agent=trained_agent,
    initial_balance=100000
)

# Calculate metrics
metrics = TradingMetrics(backtest_results)
print(f"Annual Return: {metrics.annual_return:.2%}")
print(f"Volatility: {metrics.volatility:.2%}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
print(f"Calmar Ratio: {metrics.calmar_ratio:.2f}")
```

## 🔧 Configuration

FSRPPO supports flexible configuration through YAML files:

```yaml
# config/default.yaml
signal_processing:
  ceemdan:
    trials: 100
    epsilon: 0.005
    max_imf: 10
  esmd:
    M: 4
    max_imfs: 8
    max_sift: 50
  hurst:
    window: 252
    method: "rs"

ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2

environment:
  initial_balance: 100000
  transaction_cost: 0.001
  max_position_size: 0.1
  lookback_window: 60
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use FSRPPO in your research, please cite:

```bibtex
@article{wang2024fsrppo,
  title={An adaptive financial trading strategy based on proximal policy optimization and financial signal representation},
  author={Wang, Lin and Wang, Xuerui},
  journal={Journal of Financial Markets},
  year={2024}
}
```

## 🙏 Acknowledgments

- Original FSRPPO research by Wang & Wang
- Stable Baselines3 for RL implementations
- The open-source community for various signal processing techniques

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Please consult with a qualified financial advisor before making investment decisions.