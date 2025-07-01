// Quantum Portfolio Optimizer Frontend
class QuantumPortfolioApp {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.currentResults = null;
        this.init();
    }

    init() {
        this.checkServerStatus();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Auto-validate stocks when symbols change
        document.getElementById('stock-symbols').addEventListener('input', this.debounce(() => {
            this.validateStocks();
        }, 1000));

        // Update backend options based on method
        document.getElementById('optimization-method').addEventListener('change', (e) => {
            this.updateBackendOptions(e.target.value);
        });
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    async checkServerStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const status = await response.json();
            
            const indicator = document.getElementById('status-indicator');
            if (response.ok) {
                indicator.className = 'status-indicator status-online';
                indicator.innerHTML = '<i class="fas fa-circle"></i> Server Online';
            } else {
                indicator.className = 'status-indicator status-offline';
                indicator.innerHTML = '<i class="fas fa-circle"></i> Server Offline';
            }
        } catch (error) {
            const indicator = document.getElementById('status-indicator');
            indicator.className = 'status-indicator status-offline';
            indicator.innerHTML = '<i class="fas fa-circle"></i> Server Offline';
        }
    }

    updateBackendOptions(method) {
        const backendSelect = document.getElementById('quantum-backend');
        const qaoaLayersInput = document.getElementById('qaoa-layers');
        
        if (method.includes('qaoa')) {
            backendSelect.innerHTML = `
                <option value="aer_simulator">AER Simulator</option>
                <option value="statevector_simulator">Statevector Simulator</option>
                <option value="mps_simulator">MPS Simulator</option>
            `;
            qaoaLayersInput.parentElement.style.display = 'block';
        } else {
            backendSelect.innerHTML = `
                <option value="local">Local Optimization</option>
            `;
            qaoaLayersInput.parentElement.style.display = 'none';
        }
    }

    showLoading(message = 'Processing...') {
        document.getElementById('loading').style.display = 'flex';
        document.getElementById('loading').innerHTML = `
            <div class="spinner"></div>
            <span>${message}</span>
        `;
        document.getElementById('results-content').style.display = 'none';
        document.getElementById('error-message').style.display = 'none';
        document.getElementById('success-message').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loading').style.display = 'none';
    }

    showError(message) {
        document.getElementById('error-message').style.display = 'block';
        document.getElementById('error-message').textContent = message;
        this.hideLoading();
    }

    showSuccess(message) {
        document.getElementById('success-message').style.display = 'block';
        document.getElementById('success-message').textContent = message;
        this.hideLoading();
    }

    async validateStocks() {
        const symbolsText = document.getElementById('stock-symbols').value.trim();
        if (!symbolsText) return;

        const symbols = symbolsText.split(',').map(s => s.trim()).filter(s => s);
        if (symbols.length === 0) return;

        this.showLoading('Validating stocks...');

        try {
            const response = await fetch(`${this.apiBase}/validate-stocks`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symbols })
            });

            const data = await response.json();

            if (response.ok) {
                const validStocks = data.filter(stock => stock.valid);
                const invalidStocks = data.filter(stock => !stock.valid);
                
                if (invalidStocks.length > 0) {
                    this.showError(`Invalid stocks: ${invalidStocks.map(s => s.symbol).join(', ')}`);
                } else {
                    this.showSuccess(`All ${validStocks.length} stocks are valid!`);
                }
            } else {
                this.showError(data.detail || 'Failed to validate stocks');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }

    async optimizePortfolio() {
        const investmentAmount = parseFloat(document.getElementById('investment-amount').value);
        const symbolsText = document.getElementById('stock-symbols').value.trim();
        const optimizationMethod = document.getElementById('optimization-method').value;
        const riskTolerance = document.getElementById('risk-tolerance').value;
        const quantumBackend = document.getElementById('quantum-backend').value;
        const qaoaLayers = parseInt(document.getElementById('qaoa-layers').value);
        const maxIterations = parseInt(document.getElementById('max-iterations').value);

        if (!symbolsText) {
            this.showError('Please enter stock symbols');
            return;
        }

        const symbols = symbolsText.split(',').map(s => s.trim()).filter(s => s);
        if (symbols.length < 2) {
            this.showError('Please enter at least 2 stock symbols');
            return;
        }

        this.showLoading('Running quantum portfolio optimization...');

        try {
            const response = await fetch(`${this.apiBase}/optimize-portfolio`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    investment_amount: investmentAmount,
                    stock_symbols: symbols,
                    optimization_method: optimizationMethod,
                    risk_tolerance: riskTolerance,
                    quantum_backend: quantumBackend,
                    qaoa_layers: qaoaLayers,
                    max_iterations: maxIterations,
                    include_visualization: true
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.currentResults = data;
                this.displayResults(data);
                this.showSuccess('Portfolio optimization completed successfully!');
            } else {
                this.showError(data.detail || 'Failed to optimize portfolio');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }

    async testAdvancedQuantum() {
        this.showLoading('Testing advanced quantum optimizer...');

        try {
            const response = await fetch(`${this.apiBase}/test-advanced-quantum`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (response.ok) {
                this.currentResults = data;
                this.displayResults(data);
                this.showSuccess('Advanced quantum test completed successfully!');
            } else {
                this.showError(data.detail || 'Failed to test advanced quantum optimizer');
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }

    displayResults(data) {
        document.getElementById('results-content').style.display = 'block';
        
        // Display metrics
        this.displayMetrics(data);
        
        // Display allocations
        this.displayAllocations(data);
        
        // Display visualizations
        this.displayVisualizations(data);
    }

    displayMetrics(data) {
        const metricsGrid = document.getElementById('metrics-grid');
        metricsGrid.innerHTML = `
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <div class="value">${data.sharpe_ratio.toFixed(3)}</div>
            </div>
            <div class="metric-card">
                <h3>Expected Return</h3>
                <div class="value">${(data.expected_return * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <h3>Risk (Std Dev)</h3>
                <div class="value">${(data.risk_std_dev * 100).toFixed(2)}%</div>
            </div>
            <div class="metric-card">
                <h3>Method</h3>
                <div class="value">${data.optimization_method.replace('_', ' ').toUpperCase()}</div>
            </div>
        `;

        if (data.quantum_metrics) {
            metricsGrid.innerHTML += `
                <div class="metric-card">
                    <h3>Quantum Backend</h3>
                    <div class="value">${data.quantum_metrics.backend_used}</div>
                </div>
                <div class="metric-card">
                    <h3>QAOA Layers</h3>
                    <div class="value">${data.quantum_metrics.qaoa_layers}</div>
                </div>
                <div class="metric-card">
                    <h3>Execution Time</h3>
                    <div class="value">${data.quantum_metrics.execution_time.toFixed(2)}s</div>
                </div>
            `;
        }
    }

    displayAllocations(data) {
        const tbody = document.getElementById('allocations-body');
        tbody.innerHTML = '';

        Object.entries(data.allocations).forEach(([symbol, allocation]) => {
            const investment = data.investment_allocations[symbol];
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${symbol}</strong></td>
                <td>${(allocation * 100).toFixed(2)}%</td>
                <td>$${investment.toLocaleString()}</td>
                <td>$${Math.random() * 200 + 50}</td>
            `;
            tbody.appendChild(row);
        });
    }

    displayVisualizations(data) {
        // Portfolio allocation pie chart
        if (data.allocations) {
            const pieData = [{
                values: Object.values(data.allocations).map(v => v * 100),
                labels: Object.keys(data.allocations),
                type: 'pie',
                hole: 0.4,
                marker: {
                    colors: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
                }
            }];

            const pieLayout = {
                title: 'Portfolio Allocation',
                height: 400,
                showlegend: true,
                legend: {
                    orientation: 'h',
                    y: -0.1
                }
            };

            Plotly.newPlot('portfolio-chart', pieData, pieLayout);
        }

        // Performance comparison chart
        if (data.performance_comparison) {
            const comparisonData = [
                {
                    x: Object.keys(data.performance_comparison),
                    y: Object.values(data.performance_comparison).map(p => p.sharpe_ratio),
                    type: 'bar',
                    name: 'Sharpe Ratio',
                    marker: { color: '#667eea' }
                }
            ];

            const comparisonLayout = {
                title: 'Method Comparison',
                height: 400,
                xaxis: { title: 'Method' },
                yaxis: { title: 'Sharpe Ratio' }
            };

            Plotly.newPlot('comparison-chart', comparisonData, comparisonLayout);
        }
    }
}

// Tab switching functionality
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(`${tabName}-tab`).classList.add('active');
    
    // Add active class to selected tab
    event.target.classList.add('active');
}

// Global functions for button clicks
function validateStocks() {
    app.validateStocks();
}

function optimizePortfolio() {
    app.optimizePortfolio();
}

function testAdvancedQuantum() {
    app.testAdvancedQuantum();
}

// Initialize the app
const app = new QuantumPortfolioApp();

// Auto-check server status every 30 seconds
setInterval(() => {
    app.checkServerStatus();
}, 30000); 