# 💰 Smart Pricing Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)](https://github.com/aakarsh-hub/smart-pricing-engine)

An AI-driven dynamic pricing simulation and optimization tool powered by machine learning. Optimize your pricing strategy using demand forecasting, competitor analysis, and revenue maximization algorithms.

## ✨ Key Features

- **Dynamic Pricing Models**: ML-based price optimization algorithms
- **Demand Forecasting**: Time series analysis and prediction
- **Competitor Analysis**: Market position and price sensitivity
- **Revenue Optimization**: Maximize revenue using elasticity models
- **A/B Testing Support**: Test pricing strategies before deployment
- **Interactive Jupyter Notebooks**: Experiment with pricing scenarios
- **REST API**: Easy integration with existing systems
- **Visualization Dashboard**: Real-time pricing insights

## 🏗️ Architecture

```
Historical Data → Feature Engineering → ML Models → Price Optimization → API/Dashboard
                                         ↓
                                  Elasticity Engine
```

## 🚦 Quick Start

### Prerequisites
- Python 3.8+
- scikit-learn, XGBoost for ML
- pandas, numpy for data processing

### Installation

```bash
git clone https://github.com/aakarsh-hub/smart-pricing-engine.git
cd smart-pricing-engine
pip install -r requirements.txt
```

### Run a Pricing Simulation

```bash
# Start the pricing API
python src/pricing_api.py

# Or run the interactive notebook
jupyter notebook notebooks/pricing_simulation.ipynb
```

## 📊 Sample Usage

### Basic Price Optimization

```python
from pricing_engine import PricingOptimizer
import pandas as pd

# Load your product data
data = pd.read_csv('data/sample_products.csv')

# Initialize optimizer
optimizer = PricingOptimizer(
    base_price=100,
    cost=60,
    elasticity=-2.5  # Price elasticity of demand
)

# Get optimal price
optimal_price = optimizer.find_optimal_price(
    competitor_prices=[95, 105, 98],
    inventory_level=500,
    demand_forecast=200
)

print(f"Optimal price: ${optimal_price:.2f}")
print(f"Expected revenue: ${optimizer.expected_revenue():.2f}")
```

### Dynamic Pricing API

```python
import requests

# Get pricing recommendation
response = requests.post('http://localhost:9000/api/price', json={
    'product_id': 'SKU_12345',
    'base_price': 100,
    'cost': 60,
    'competitor_prices': [95, 105, 98],
    'inventory': 500,
    'demand_forecast': 200,
    'time_of_day': 'peak'
})

pricing = response.json()
print(f"Recommended price: ${pricing['optimal_price']}")
print(f"Expected margin: {pricing['margin_percent']}%")
print(f"Confidence: {pricing['confidence']}")
```

### Demand Forecasting

```python
from pricing_engine import DemandForecaster

# Train demand model
forecaster = DemandForecaster()
forecaster.fit(historical_sales_data)

# Predict future demand
future_demand = forecaster.predict(
    price_point=95,
    seasonality='holiday',
    day_of_week='friday'
)

print(f"Predicted demand: {future_demand} units")
```

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src
```

Test coverage: **91%**

## 📁 Project Structure

```
smart-pricing-engine/
├── src/
│   ├── pricing_api.py        # REST API server
│   ├── pricing_engine.py     # Core optimization logic
│   ├── demand_forecaster.py  # Demand prediction models
│   ├── elasticity_model.py   # Price elasticity calculations
│   └── revenue_optimizer.py  # Revenue maximization
├── notebooks/
│   ├── pricing_simulation.ipynb
│   ├── demand_analysis.ipynb
│   └── competitor_insights.ipynb
├── data/
│   ├── sample_products.csv
│   ├── historical_sales.csv
│   └── competitor_prices.csv
├── models/
│   └── trained_models/       # Pre-trained ML models
├── tests/
│   ├── test_pricing.py
│   ├── test_forecasting.py
│   └── test_optimization.py
├── requirements.txt
└── README.md
```

## 🔧 Key Technologies

- **ML/AI**: scikit-learn, XGBoost, Prophet
- **Data**: pandas, numpy
- **API**: Flask, FastAPI
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Optimization**: SciPy
- **Testing**: Pytest

## 📊 Pricing Models

### 1. Cost-Plus Pricing
Simple markup over cost with competitive adjustments

### 2. Value-Based Pricing
Pricing based on perceived customer value

### 3. Dynamic Pricing
Real-time adjustments based on:
- Demand levels
- Competitor prices
- Inventory levels
- Time of day/season
- Customer segments

### 4. Price Elasticity Model
Optimize using demand elasticity curves

### 5. Revenue Maximization
Find price point that maximizes total revenue

## 🎯 Use Cases

- **E-commerce**: Dynamic pricing for online stores
- **SaaS**: Subscription pricing optimization
- **Retail**: Seasonal pricing strategies
- **Hotels/Travel**: Demand-based pricing
- **Marketplaces**: Seller price recommendations

## 📊 Sample Datasets Included

- E-commerce product sales (10,000+ records)
- Competitor pricing data
- Seasonal demand patterns
- Customer segmentation data

## 🚀 Interactive Notebooks

Explore pricing strategies with Jupyter notebooks:

1. **pricing_simulation.ipynb**: Interactive price optimization
2. **demand_analysis.ipynb**: Demand forecasting experiments
3. **competitor_insights.ipynb**: Market analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

**Aakarsh**
- GitHub: [@aakarsh-hub](https://github.com/aakarsh-hub)

---

⭐ Star this repository if you're interested in AI-powered pricing strategies!
