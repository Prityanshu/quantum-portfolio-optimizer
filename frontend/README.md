# Quantum Portfolio Optimizer Frontend

A modern, responsive web interface for the Quantum Portfolio Optimization API.

## Features

- 🎨 **Modern UI/UX**: Beautiful gradient design with smooth animations
- 📊 **Interactive Visualizations**: Real-time charts using Plotly.js
- ⚡ **Real-time Updates**: Live server status monitoring
- 🔄 **Multiple Optimization Methods**: QAOA, Quantum-Inspired, Classical MPT
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- 🚀 **Fast Performance**: Optimized for quick portfolio optimization

## Quick Start

### Prerequisites

- Python 3.7+ (for serving the frontend)
- Quantum Portfolio API running on `http://localhost:8000`

### Installation

1. **Start the Backend API** (in the main project directory):
   ```bash
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Start the Frontend** (in the frontend directory):
   ```bash
   cd frontend
   python -m http.server 3000
   ```

3. **Open in Browser**:
   ```
   http://localhost:3000
   ```

## Usage

### 1. Configure Portfolio
- Enter your investment amount
- Add stock symbols (comma-separated)
- Select optimization method:
  - **Advanced QAOA**: Most sophisticated quantum algorithm
  - **QAOA**: Quantum Approximate Optimization Algorithm
  - **Quantum-Inspired**: Classical algorithm with quantum principles
  - **Classical MPT**: Traditional Modern Portfolio Theory

### 2. Set Parameters
- **Risk Tolerance**: Conservative, Moderate, or Aggressive
- **Quantum Backend**: AER Simulator, Statevector, or MPS Simulator
- **QAOA Layers**: Number of quantum circuit layers (1-10)
- **Max Iterations**: Optimization iterations (10-500)

### 3. Optimize Portfolio
- Click "Validate Stocks" to check symbol validity
- Click "Optimize Portfolio" to run the optimization
- View results in the interactive dashboard

## Features Explained

### Real-time Metrics
- **Sharpe Ratio**: Risk-adjusted return measure
- **Expected Return**: Predicted portfolio return
- **Risk (Std Dev)**: Portfolio volatility
- **Quantum Metrics**: Backend info, execution time, QAOA layers

### Interactive Visualizations
- **Portfolio Allocation**: Pie chart showing asset distribution
- **Performance Comparison**: Bar chart comparing optimization methods
- **Real-time Updates**: Charts update automatically with new data

### Server Monitoring
- **Status Indicator**: Shows API server status
- **Auto-refresh**: Checks server health every 30 seconds
- **Error Handling**: Graceful error messages and fallbacks

## API Integration

The frontend communicates with the following API endpoints:

- `GET /health` - Server status check
- `POST /validate-stocks` - Stock symbol validation
- `POST /optimize-portfolio` - Portfolio optimization
- `POST /test-advanced-quantum` - Advanced quantum testing

## Customization

### Styling
The frontend uses CSS custom properties for easy theming:

```css
:root {
  --primary-color: #667eea;
  --secondary-color: #764ba2;
  --success-color: #43e97b;
  --error-color: #f5576c;
}
```

### Adding New Features
1. Add new form fields in `index.html`
2. Update the JavaScript in `app.js`
3. Extend the API calls as needed

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure the backend has CORS enabled
2. **API Connection**: Check that the backend is running on port 8000
3. **Stock Validation**: Verify stock symbols are valid
4. **Quantum Backend**: Some backends may require additional setup

### Debug Mode
Open browser developer tools to see detailed API requests and responses.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details. 