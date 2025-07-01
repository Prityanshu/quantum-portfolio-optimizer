"""
Advanced Quantum Portfolio Optimizer
===================================
Integrates Qiskit-based QAOA optimization with multi-server architecture
and real-time stock data from multiple sources.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import os
import time
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Quantum Computing Imports
try:
    from qiskit import transpile
    from qiskit_aer import Aer
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit import QuantumCircuit, Parameter
    from qiskit_algorithms.optimizers import SLSQP, COBYLA
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
    from scipy.optimize import minimize
    QUANTUM_AVAILABLE = True
except ImportError as e:
    QUANTUM_AVAILABLE = False
    logging.warning(f"Quantum libraries not available: {e}")

logger = logging.getLogger(__name__)

# Set Random Seed for Reproducibility
np.random.seed(42)

class QuantumPortfolioOptimizer:
    """Advanced quantum portfolio optimizer with multi-server support"""
    
    def __init__(self, use_ibm_quantum: bool = True, max_backend_time: int = 180):
        """
        Initialize the quantum portfolio optimizer
        
        Args:
            use_ibm_quantum: Whether to use IBM Quantum backends
            max_backend_time: Maximum backend time in seconds (default 3 minutes)
        """
        # Ensure IBM Quantum API key is set
        ibm_api_key = os.getenv('IBM_QUANTUM_API_KEY')
        if ibm_api_key and not os.getenv('QISKIT_IBMQ_API_TOKEN'):
            os.environ['QISKIT_IBMQ_API_TOKEN'] = ibm_api_key
        # QiskitRuntimeService will pick up from env automatically
        self.use_ibm_quantum = use_ibm_quantum and QUANTUM_AVAILABLE
        self.max_backend_time = max_backend_time
        self.service = None
        self.backend = None
        self.cache_dir = "cache"
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize quantum backend if available
        if self.use_ibm_quantum:
            self._setup_ibm_quantum()
    
    def _setup_ibm_quantum(self):
        """Setup IBM Quantum backend"""
        try:
            ibm_api_key = os.getenv('IBM_QUANTUM_API_KEY')
            if ibm_api_key:
                self.service = QiskitRuntimeService(channel="ibm_quantum", token=ibm_api_key)
            else:
                self.service = QiskitRuntimeService(channel="ibm_quantum")
            backends = self.service.backends()
            suitable_backends = [b for b in backends if b.status().operational]
            suitable_backends.sort(key=lambda b: -b.configuration().n_qubits)
            
            if suitable_backends:
                self.backend = self.service.backend(suitable_backends[0].name)
                logger.info(f"✓ Connected to IBM Quantum backend: {self.backend.name}")
            else:
                logger.warning("⚠️ No suitable quantum backends available")
                self.use_ibm_quantum = False
        except Exception as e:
            logger.warning(f"⚠️ Error connecting to IBM Quantum: {e}")
            self.use_ibm_quantum = False
    
    def get_stock_data(self, tickers: List[str], years: int = 5) -> pd.DataFrame:
        """Fetch stock data with caching and parallel processing. Fallback to sample data if all fail."""
        cache_file = os.path.join(self.cache_dir, f"stock_data_{'_'.join(tickers[:5])}_etc.pkl")
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    stock_prices = pickle.load(f)
                    logger.info(f"✓ Loaded cached stock data from {cache_file}")
                    if not stock_prices.empty:
                        return stock_prices
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}")
        # Fetch new data
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.DateOffset(years=years)
        stock_data = {}
        def download_stock(ticker):
            try:
                # Try Alpha Vantage first
                alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
                if alpha_vantage_key:
                    try:
                        url = "https://www.alphavantage.co/query"
                        params = {
                            'function': 'TIME_SERIES_DAILY',
                            'symbol': ticker,
                            'apikey': alpha_vantage_key,
                            'outputsize': 'full'
                        }
                        response = requests.get(url, params=params, timeout=10)
                        if response.status_code == 200:
                            data = response.json()
                            if 'Time Series (Daily)' in data:
                                time_series = data['Time Series (Daily)']
                                dates = list(time_series.keys())
                                dates.sort(reverse=True)
                                dates = dates[:years * 252]
                                data_list = []
                                for date in dates:
                                    daily_data = time_series[date]
                                    data_list.append({'Date': date, 'Close': float(daily_data['4. close'])})
                                df = pd.DataFrame(data_list)
                                df['Date'] = pd.to_datetime(df['Date'])
                                df.set_index('Date', inplace=True)
                                df = df.sort_index()
                                logger.info(f"✓ Downloaded {ticker} from Alpha Vantage")
                                return ticker, df['Close']
                    except Exception as e:
                        logger.warning(f"Alpha Vantage failed for {ticker}: {e}")
                # Fallback to Yahoo Finance
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date, interval="1mo")["Close"]
                logger.info(f"✓ Downloaded {ticker} from Yahoo Finance")
                return ticker, data
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                return ticker, None
        # Parallel download
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(download_stock, tickers))
        for ticker, data in results:
            if data is not None and len(data) > 0:
                stock_data[ticker] = data
        stock_prices = pd.DataFrame(stock_data)
        # Drop stocks with too many missing values
        threshold = len(stock_prices) * 0.2
        stocks_to_drop = []
        for column in stock_prices.columns:
            if stock_prices[column].isna().sum() > threshold:
                stocks_to_drop.append(column)
                logger.warning(f"Dropping {column} due to excessive missing data")
        stock_prices = stock_prices.drop(columns=stocks_to_drop)
        stock_prices = stock_prices.fillna(method='ffill').dropna()
        # Fallback: If no valid data, generate sample data
        if stock_prices.empty or len(stock_prices.columns) < 2:
            logger.warning("Insufficient real data, using sample data for demonstration.")
            stock_prices = self.generate_sample_data(tickers)
        # Cache the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(stock_prices, f)
                logger.info(f"✓ Cached stock data to {cache_file}")
        except Exception as e:
            logger.warning(f"Error caching data: {e}")
        return stock_prices
    
    def generate_sample_data(self, tickers: List[str]) -> pd.DataFrame:
        """Generate realistic sample data for demonstration purposes."""
        logger.info("Generating sample data for demonstration")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')[-252:]
        base_prices = {t: 100 + 10 * i for i, t in enumerate(tickers)}
        prices_data = {}
        for symbol in tickers:
            np.random.seed(hash(symbol) % 1000)
            start_price = base_prices.get(symbol, 100)
            volatility = 0.03
            daily_return = 0.0008
            returns = np.random.normal(daily_return, volatility, len(dates))
            prices = [start_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1))
            price_series = pd.Series(prices, index=dates[:len(prices)])
            prices_data[symbol] = price_series
            logger.info(f"✓ {symbol}: Generated {len(price_series)} sample data points")
        return pd.DataFrame(prices_data)
    
    def preprocess_data(self, stock_prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess stock data for quantum algorithm"""
        # Compute Log Returns
        log_returns = np.log(stock_prices / stock_prices.shift(1)).dropna()
        expected_returns = log_returns.mean().values
        cov_matrix = log_returns.cov().values
        
        # Normalize Returns and Covariance Matrix
        expected_returns = (expected_returns - np.min(expected_returns)) / (np.max(expected_returns) - np.min(expected_returns))
        cov_matrix /= np.max(np.abs(cov_matrix))
        
        return expected_returns, cov_matrix
    
    def portfolio_hamiltonian(self, returns: np.ndarray, cov_matrix: np.ndarray, 
                            gamma: float = 1.5, alpha: float = 3.0, max_assets: Optional[int] = None) -> SparsePauliOp:
        """Define Portfolio Hamiltonian for quantum optimization"""
        num_assets = len(returns)
        
        # If max_assets specified, use only top performing assets
        if max_assets and max_assets < num_assets:
            volatility = np.sqrt(np.diag(cov_matrix))
            sharpe = returns / (volatility + 1e-10)
            top_indices = np.argsort(sharpe)[-max_assets:]
            returns = returns[top_indices]
            cov_matrix = cov_matrix[np.ix_(top_indices, top_indices)]
            num_assets = len(returns)
            logger.info(f"Using {num_assets} assets based on Sharpe ratio")
        
        terms, coeffs = [], []
        
        # Build return terms (single qubits)
        for i in range(num_assets):
            term = ["I"] * num_assets
            term[i] = "Z"
            terms.append("".join(term))
            coeffs.append(-alpha * returns[i])
        
        # Build covariance terms (qubit pairs)
        for i in range(num_assets):
            for j in range(i + 1, num_assets):
                term = ["I"] * num_assets
                term[i] = "Z"
                term[j] = "Z"
                terms.append("".join(term))
                coeffs.append(gamma * cov_matrix[i, j])
        
        return SparsePauliOp(terms, coeffs)
    
    def create_qaoa_ansatz(self, hamiltonian: SparsePauliOp, p: int = 2, measure: bool = True) -> QuantumCircuit:
        """Create QAOA ansatz circuit"""
        ansatz = QAOAAnsatz(hamiltonian, reps=p)
        if measure:
            ansatz.measure_all()
        return ansatz
    
    def check_remaining_backend_time(self) -> Optional[float]:
        """Check remaining backend time"""
        try:
            if self.service is None:
                return None
            
            account_info = self.service.account()
            
            if hasattr(account_info, "remaining_runtime_seconds"):
                remaining_seconds = account_info.remaining_runtime_seconds
            elif hasattr(account_info, "runtime") and hasattr(account_info.runtime, "remaining_seconds"):
                remaining_seconds = account_info.runtime.remaining_seconds
            elif isinstance(account_info, dict) and "runtime" in account_info:
                if "remaining_seconds" in account_info["runtime"]:
                    remaining_seconds = account_info["runtime"]["remaining_seconds"]
                else:
                    remaining_seconds = 600  # Default to 10 minutes
            else:
                remaining_seconds = 600  # Default to 10 minutes
            
            logger.info(f"Remaining backend time: {remaining_seconds:.1f} seconds")
            return remaining_seconds
            
        except Exception as e:
            logger.warning(f"Error checking remaining backend time: {e}")
            return 600  # Default to 10 minutes
    
    def local_qaoa_simulation(self, params: np.ndarray, ansatz: QuantumCircuit, 
                            expected_returns: np.ndarray, cov_matrix: np.ndarray, 
                            alpha: float, gamma: float) -> float:
        """Perform local QAOA simulation"""
        num_assets = len(expected_returns)
        circuit = ansatz.assign_parameters(params)
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(circuit, simulator)
        
        # Run the simulation
        job = simulator.run(transpiled_circuit, shots=4096)
        result = job.result()
        counts = result.get_counts()
        
        # Compute the expectation value
        energy = 0
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            bitstring_padded = bitstring.zfill(num_assets)[-num_assets:]
            z_vals = np.array([1 if bit == '0' else -1 for bit in bitstring_padded])
            
            # Compute energy contribution
            for i in range(num_assets):
                energy -= alpha * expected_returns[i] * z_vals[i] * count / total_shots
            
            for i in range(num_assets):
                for j in range(i + 1, num_assets):
                    energy += gamma * cov_matrix[i, j] * z_vals[i] * z_vals[j] * count / total_shots
        
        return energy
    
    def optimize_parameters_locally(self, ansatz: QuantumCircuit, expected_returns: np.ndarray, 
                                  cov_matrix: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
        """Optimize parameters locally"""
        logger.info(f"Number of parameters in QAOA ansatz: {ansatz.num_parameters}")
        
        num_trials = 3
        best_params = None
        best_energy = float('inf')
        
        for trial in range(num_trials):
            initial_params = np.random.rand(ansatz.num_parameters) * np.pi * 2
            
            logger.info(f"Starting local optimization trial {trial+1}/{num_trials}...")
            
            objective_func = lambda params: self.local_qaoa_simulation(
                params, ansatz, expected_returns, cov_matrix, alpha, gamma
            )
            
            result = minimize(
                objective_func, 
                initial_params,
                method='COBYLA',
                options={'maxiter': 150}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
            
            logger.info(f"Trial {trial+1} completed with cost: {result.fun}")
        
        logger.info(f"Local optimization completed with best cost: {best_energy}")
        return best_params
    
    def generate_weights_from_local_simulation(self, params: np.ndarray, ansatz: QuantumCircuit, 
                                             num_assets: int) -> np.ndarray:
        """Generate portfolio weights from local simulation"""
        circuit = ansatz.assign_parameters(params)
        simulator = Aer.get_backend('qasm_simulator')
        transpiled_circuit = transpile(circuit, simulator)
        
        job = simulator.run(transpiled_circuit, shots=16384)
        result = job.result()
        counts = result.get_counts()
        
        # Find the most frequently occurring bitstring
        max_count = 0
        max_bitstring = None
        for bitstring, count in counts.items():
            if count > max_count:
                max_count = count
                max_bitstring = bitstring
        
        # Convert to portfolio weights
        weights = np.ones(num_assets) / num_assets
        
        if max_bitstring:
            bitstring_padded = max_bitstring.zfill(num_assets)[-num_assets:]
            weights = np.array([0.8 if bit == '0' else 0.2 for bit in bitstring_padded])
        
        # Ensure normalization
        weights = weights / np.sum(weights)
        return weights
    
    def hybrid_quantum_optimization(self, ansatz: QuantumCircuit, expected_returns: np.ndarray, 
                                  cov_matrix: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
        """Main hybrid quantum-classical optimization function"""
        num_assets = len(expected_returns)
        
        if not self.use_ibm_quantum or self.backend is None:
            logger.warning("⚠️ Cannot connect to quantum backend. Using local optimization only.")
            optimized_params = self.optimize_parameters_locally(ansatz, expected_returns, cov_matrix, alpha, gamma)
            return self.generate_weights_from_local_simulation(optimized_params, ansatz, num_assets)
        
        # Check remaining backend time
        initial_backend_seconds = self.check_remaining_backend_time()
        if initial_backend_seconds is None:
            logger.warning("⚠️ Cannot determine remaining backend time. Using conservative estimate.")
            initial_backend_seconds = 600
        
        target_seconds = min(self.max_backend_time, initial_backend_seconds - 30)
        logger.info(f"Planning to use {target_seconds} seconds of backend time")
        
        overall_start_time = time.time()
        
        # Get good starting parameters from local simulation
        logger.info("Starting local pre-optimization to get good initial parameters...")
        initial_params = self.optimize_parameters_locally(ansatz, expected_returns, cov_matrix, alpha, gamma)
        logger.info("Local pre-optimization completed!")
        
        best_params = initial_params.copy()
        best_energy = float('inf')
        iteration = 1
        session = None
        job_ids = []
        
        try:
            logger.info("Starting hybrid quantum-classical optimization...")
            with Session(backend=self.backend) as session:
                sampler = Sampler()
                
                while True:
                    remaining_backend_seconds = self.check_remaining_backend_time()
                    if remaining_backend_seconds is None:
                        elapsed_seconds = time.time() - overall_start_time
                        remaining_backend_seconds = max(0, target_seconds - elapsed_seconds)
                    
                    logger.info(f"Iteration {iteration}: {remaining_backend_seconds:.1f} seconds remaining")
                    
                    if remaining_backend_seconds <= 30:
                        logger.warning("⚠️ Less than 30 seconds remaining. Stopping optimization.")
                        break
                    
                    # Perturb parameters every few iterations
                    if iteration > 1 and iteration % 3 == 0:
                        logger.info("Adding parameter perturbation to avoid local minima")
                        current_params = best_params + np.random.normal(0, 0.1, size=len(best_params))
                    else:
                        current_params = best_params
                    
                    # Create circuit with current parameters
                    circuit = ansatz.assign_parameters(current_params)
                    transpiled_circuit = transpile(circuit, backend=self.backend, optimization_level=3)
                    
                    max_job_seconds = min(remaining_backend_seconds - 25, 180)
                    if max_job_seconds < 30:
                        logger.warning("Insufficient time for another quantum job. Finalizing optimization.")
                        break
                    
                    logger.info(f"Running quantum job {iteration} with max duration {max_job_seconds:.1f} seconds...")
                    iter_start = time.time()
                    
                    # Submit the job to the quantum computer
                    job = sampler.run([transpiled_circuit], shots=4096)
                    job_ids.append(job.job_id())
                    
                    # Wait for job completion with timeout
                    job_completed = False
                    job_wait_start = time.time()
                    
                    while time.time() - job_wait_start < max_job_seconds:
                        if (time.time() - job_wait_start) % 60 < 2:
                            backend_check = self.check_remaining_backend_time()
                            if backend_check is not None and backend_check <= 30:
                                logger.warning("⚠️ Less than 30 seconds remaining during job. Cancelling job.")
                                try:
                                    job.cancel()
                                except:
                                    pass
                                job_completed = False
                                break
                        
                        status = job.status()
                        status_name = status.name if hasattr(status, "name") else str(status)
                        logger.info(f"Job status: {status_name}, waited {(time.time() - job_wait_start):.1f} seconds")
                        
                        if hasattr(status, "name") and status.name == 'DONE':
                            job_completed = True
                            break
                        
                        time.sleep(15)
                    
                    if not job_completed:
                        logger.warning("Job timeout exceeded or cancelled.")
                        try:
                            job.cancel()
                        except:
                            pass
                        
                        backend_check = self.check_remaining_backend_time()
                        if backend_check is not None and backend_check <= 30:
                            logger.warning("⚠️ Less than 30 seconds remaining. Stopping optimization.")
                            break
                        
                        iteration += 1
                        continue
                    
                    # Job completed successfully, analyze results
                    job_time = time.time() - iter_start
                    logger.info(f"Job completed in {job_time:.2f} seconds")
                    
                    try:
                        result = job.result()
                        
                        # Extract measurement results and compute energy
                        energy = 0
                        if hasattr(result, "quasi_dists"):
                            quasi_dist = result.quasi_dists[0]
                            for bitstring_int, prob in quasi_dist.items():
                                bitstring = bin(bitstring_int)[2:].zfill(num_assets)
                                bitstring_padded = bitstring[-num_assets:]
                                z_vals = np.array([1 if bit == '0' else -1 for bit in bitstring_padded])
                                
                                for i in range(num_assets):
                                    energy -= alpha * expected_returns[i] * z_vals[i] * prob
                                
                                for i in range(num_assets):
                                    for j in range(i + 1, num_assets):
                                        energy += gamma * cov_matrix[i, j] * z_vals[i] * z_vals[j] * prob
                        
                        elif hasattr(result, "samples"):
                            samples = result.samples
                            total_samples = len(samples)
                            
                            for sample in samples:
                                z_vals = np.array([1 if bit == 0 else -1 for bit in sample[-num_assets:]])
                                
                                sample_energy = 0
                                for i in range(num_assets):
                                    sample_energy -= alpha * expected_returns[i] * z_vals[i]
                                
                                for i in range(num_assets):
                                    for j in range(i + 1, num_assets):
                                        sample_energy += gamma * cov_matrix[i, j] * z_vals[i] * z_vals[j]
                                
                                energy += sample_energy / total_samples
                        
                        else:
                            energy = best_energy if best_energy != float('inf') else 0
                        
                        logger.info(f"Computed energy: {energy}")
                        
                        # Update best parameters if current energy is better
                        if energy < best_energy:
                            best_energy = energy
                            best_params = current_params.copy()
                            logger.info(f"Found better parameters with energy: {best_energy}")
                    
                    except Exception as e:
                        logger.error(f"Error processing job result: {e}")
                    
                    iteration += 1
                    
                    # Check remaining backend time
                    backend_check = self.check_remaining_backend_time()
                    if backend_check is not None and backend_check <= 120:
                        if backend_check > 45:
                            logger.info(f"Only {backend_check:.1f} seconds remaining - running final measurement...")
                            break
                        else:
                            logger.info(f"Only {backend_check:.1f} seconds remaining - stopping optimization.")
                            break
                
                logger.info("Optimization iterations completed!")
                
                # Check if we have time for a final job with best parameters
                final_backend_seconds = self.check_remaining_backend_time()
                
                if final_backend_seconds is not None and final_backend_seconds > 40:
                    logger.info(f"Running final measurement with best parameters ({final_backend_seconds:.1f} seconds remaining)...")
                    final_circuit = ansatz.assign_parameters(best_params)
                    transpiled_final = transpile(final_circuit, backend=self.backend, optimization_level=3)
                    
                    final_timeout = max(30, final_backend_seconds - 20)
                    final_job = sampler.run([transpiled_final], shots=8192)
                    job_ids.append(final_job.job_id())
                    
                    start_wait = time.time()
                    final_completed = False
                    
                    while time.time() - start_wait < final_timeout:
                        if (time.time() - start_wait) % 30 < 2:
                            backend_check = self.check_remaining_backend_time()
                            if backend_check is not None and backend_check <= 10:
                                logger.warning("⚠️ Critical low backend time remaining. Cancelling final job.")
                                try:
                                    final_job.cancel()
                                except:
                                    pass
                                break
                        
                        status = final_job.status()
                        if hasattr(status, "name") and status.name == 'DONE':
                            final_completed = True
                            break
                        
                        logger.info(f"Final job status: {status.name if hasattr(status, 'name') else status}, waiting...")
                        time.sleep(10)
                    
                    if final_completed:
                        try:
                            final_result = final_job.result()
                            return self._generate_weights_from_quantum_result(final_result, num_assets)
                        except Exception as e:
                            logger.error(f"Error processing final result: {e}")
                            return self.generate_weights_from_local_simulation(best_params, ansatz, num_assets)
                    else:
                        try:
                            final_job.cancel()
                        except:
                            pass
                else:
                    logger.info(f"Insufficient remaining time for final job ({final_backend_seconds} seconds).")
                
                return self.generate_weights_from_local_simulation(best_params, ansatz, num_assets)
        
        except Exception as e:
            logger.error(f"⚠️ Error during hybrid optimization: {e}")
            if session:
                try:
                    session.close()
                except:
                    pass
            return self.generate_weights_from_local_simulation(best_params if 'best_params' in locals() else initial_params, ansatz, num_assets)
        
        finally:
            # Log total quantum resource usage
            total_elapsed_seconds = time.time() - overall_start_time
            final_backend_time = self.check_remaining_backend_time()
            
            if initial_backend_seconds is not None and final_backend_time is not None:
                actual_backend_usage = initial_backend_seconds - final_backend_time
                logger.info(f"Estimated quantum backend usage: {actual_backend_usage:.2f} seconds")
            else:
                logger.info(f"Total elapsed time: {total_elapsed_seconds:.2f} seconds")
            
            logger.info(f"Quantum jobs submitted: {len(job_ids)}")
            if job_ids:
                logger.info(f"Job IDs: {', '.join(job_ids)}")
    
    def _generate_weights_from_quantum_result(self, result, num_assets: int) -> np.ndarray:
        """Generate weights from quantum computer result"""
        weights = np.ones(num_assets) / num_assets
        
        try:
            if hasattr(result, "quasi_dists"):
                quasi_dist = result.quasi_dists[0]
                most_probable = max(quasi_dist.items(), key=lambda x: x[1])
                bitstring = bin(most_probable[0])[2:].zfill(num_assets)
                bitstring_padded = bitstring[-num_assets:]
                weights = np.array([0.8 if bit == '0' else 0.2 for bit in bitstring_padded])
            
            elif hasattr(result, "samples"):
                samples = result.samples
                portfolio_counts = {}
                
                for sample in samples:
                    bitstring = ''.join([str(bit) for bit in sample[-num_assets:]])
                    if bitstring in portfolio_counts:
                        portfolio_counts[bitstring] += 1
                    else:
                        portfolio_counts[bitstring] = 1
                
                most_frequent = max(portfolio_counts.items(), key=lambda x: x[1])
                bitstring_padded = most_frequent[0]
                weights = np.array([0.8 if bit == '0' else 0.2 for bit in bitstring_padded])
            
            weights = weights / np.sum(weights)
        
        except Exception as e:
            logger.error(f"Error extracting weights from quantum result: {e}")
        
        return weights
    
    def classical_markowitz_portfolio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray, 
                                    risk_free_rate: float = 0.02) -> Tuple[np.ndarray, float, float]:
        """Classical Markowitz portfolio optimization for comparison"""
        def portfolio_stats(weights):
            weights = np.array(weights)
            returns = np.dot(weights, expected_returns)
            risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return returns, risk
        
        def neg_sharpe_ratio(weights):
            r, risk = portfolio_stats(weights)
            return -(r - risk_free_rate) / risk
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(expected_returns)))
        initial_guess = np.array([1/len(expected_returns)] * len(expected_returns))
        
        result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        classical_weights = result['x']
        classical_return, classical_risk = portfolio_stats(classical_weights)
        
        return classical_weights, classical_return, classical_risk
    
    def optimize_portfolio(self, tickers: List[str], years: int = 5, risk_aversion: float = 1.5, 
                          max_assets: Optional[int] = None) -> Dict[str, Any]:
        """
        Main portfolio optimization function
        
        Args:
            tickers: List of stock tickers
            years: Number of years of historical data
            risk_aversion: Risk aversion parameter
            max_assets: Maximum number of assets to use
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        logger.info(f"🚀 Starting quantum portfolio optimization with {len(tickers)} assets...")
        
        # Get stock data
        logger.info("📊 Fetching historical stock data...")
        stock_prices = self.get_stock_data(tickers, years=years)
        actual_tickers = list(stock_prices.columns)
        logger.info(f"Using {len(actual_tickers)} assets after data cleaning: {', '.join(actual_tickers)}")
        
        # Preprocess the data
        logger.info("🔄 Processing stock data...")
        expected_returns, cov_matrix = self.preprocess_data(stock_prices)
        
        # Determine circuit size based on backend or local constraints
        if self.use_ibm_quantum and self.backend:
            max_circuit_assets = min(len(actual_tickers), self.backend.configuration().n_qubits - 5)
        else:
            max_circuit_assets = min(len(actual_tickers), 20)
        
        if max_circuit_assets < len(actual_tickers):
            logger.info(f"⚠️ Reducing asset count from {len(actual_tickers)} to {max_circuit_assets} due to backend constraints")
            
            # Select top assets by Sharpe ratio
            volatility = np.sqrt(np.diag(cov_matrix))
            sharpe = expected_returns / (volatility + 1e-10)
            top_indices = np.argsort(sharpe)[-max_circuit_assets:]
            
            final_tickers = [actual_tickers[i] for i in top_indices]
            expected_returns = expected_returns[top_indices]
            cov_matrix = cov_matrix[np.ix_(top_indices, top_indices)]
        else:
            final_tickers = actual_tickers
        
        logger.info(f"🔢 Final optimization will use {len(final_tickers)} assets: {', '.join(final_tickers)}")
        
        # Build the Hamiltonian
        alpha = 3.0  # Return importance factor
        gamma = risk_aversion  # Risk aversion factor
        logger.info(f"⚙️ Building quantum Hamiltonian with α={alpha} (return) and γ={gamma} (risk)...")
        hamiltonian = self.portfolio_hamiltonian(expected_returns, cov_matrix, gamma=gamma, alpha=alpha)
        
        # Create QAOA ansatz
        p_layers = min(2, max(1, len(final_tickers) // 10))
        logger.info(f"🔮 Creating QAOA ansatz with p={p_layers} layers...")
        ansatz = self.create_qaoa_ansatz(hamiltonian, p=p_layers)
        
        # Run hybrid quantum-classical optimization
        logger.info("⚛️ Starting quantum portfolio optimization...")
        quantum_weights = self.hybrid_quantum_optimization(
            ansatz, expected_returns, cov_matrix, alpha, gamma
        )
        
        # Map weights to tickers
        quantum_portfolio = dict(zip(final_tickers, quantum_weights))
        
        # Calculate expected return and risk for quantum portfolio
        quantum_return = np.dot(quantum_weights, expected_returns)
        quantum_risk = np.sqrt(np.dot(quantum_weights.T, np.dot(cov_matrix, quantum_weights)))
        quantum_sharpe = (quantum_return - 0.02) / quantum_risk
        
        # Run classical optimization for comparison
        logger.info("🧮 Running classical portfolio optimization for comparison...")
        classical_weights, classical_return, classical_risk = self.classical_markowitz_portfolio(expected_returns, cov_matrix)
        classical_portfolio = dict(zip(final_tickers, classical_weights))
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"✅ Total execution time: {total_time:.2f} seconds")
        
        return {
            'quantum_portfolio': quantum_portfolio,
            'quantum_return': quantum_return,
            'quantum_risk': quantum_risk,
            'quantum_sharpe': quantum_sharpe,
            'classical_portfolio': classical_portfolio,
            'classical_return': classical_return,
            'classical_risk': classical_risk,
            'tickers': final_tickers,
            'execution_time': total_time,
            'quantum_available': self.use_ibm_quantum,
            'backend_used': self.backend.name if self.backend else 'local_simulator'
        }