#!/usr/bin/env python3
"""
Sales Uplift Forecasting - Comprehensive Inference & Demo Script
===============================================================

PRISM Worklet 8 - Samsung Project
Advanced AI-powered sales performance prediction and uplift analysis

This script provides comprehensive demonstrations of the Sales Uplift Forecasting system
including health checks, individual predictions, batch processing, and API testing.

For Samsung Mentors / SRI-B Team:
- Run with minimal modifications required
- Comprehensive logging and error handling
- Sample data generation included
- API endpoint documentation and testing

Author: PRISM Team
Version: 2.0
Date: December 2024
"""

import sys
import os
import time
import json
import requests
import pandas as pd
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sales_uplift_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SalesUpliftDemoSystem:
    """
    Comprehensive demo system for Sales Uplift Forecasting application
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:7003"):
        """
        Initialize the demo system
        
        Args:
            base_url: Base URL for the Sales Uplift Forecasting application
        """
        self.base_url = base_url
        self.app_name = "Sales Uplift Forecasting"
        self.version = "2.0"
        
        # Expected model files in pipeline directory
        self.model_files = [
            'xgb_model.pkl',
            'encoder.pkl',
            'scaler.pkl'
        ]
        
        # API endpoints
        self.endpoints = {
            'health': f"{base_url}/health",
            'predict': f"{base_url}/predict",
            'batch_predict': f"{base_url}/batch_predict",
            'upload': f"{base_url}/upload"
        }
        
        logger.info(f"Initialized {self.app_name} Demo System v{self.version}")
        logger.info(f"Base URL: {self.base_url}")

    def print_banner(self):
        """Print welcome banner"""
        banner = f"""
{'='*80}
🏆 PRISM Worklet 8 - {self.app_name}
📊 Advanced AI-powered Sales Performance Prediction & Uplift Analysis
{'='*80}

🎯 Samsung Project - Preparing and Inspiring Student Minds
📈 Unified AI Platform for Business Intelligence

Application: {self.app_name}
Version: {self.version}
Port: 7003
Base URL: {self.base_url}

{'='*80}
"""
        print(banner)
        logger.info("Sales Uplift Forecasting Demo System Started")

    def check_system_health(self) -> bool:
        """
        Comprehensive system health check
        
        Returns:
            bool: True if system is healthy, False otherwise
        """
        logger.info("🔍 Starting comprehensive system health check...")
        
        health_status = {
            'server_running': False,
            'models_loaded': False,
            'dependencies_ok': False,
            'api_responsive': False,
            'pipeline_structure': False
        }
        
        try:
            # Check if server is running
            logger.info("📡 Checking server connectivity...")
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                health_status['server_running'] = True
                logger.info("✅ Server is running and accessible")
            else:
                logger.error(f"❌ Server returned status code: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to server. Please ensure the application is running:")
            logger.error(f"   cd 'Sell-out performance forecasting (sales uplift)/pipeline'")
            logger.error(f"   python app.py")
            return False
        except Exception as e:
            logger.error(f"❌ Server check failed: {e}")
            return False
        
        # Check pipeline directory structure
        logger.info("📁 Checking pipeline directory structure...")
        pipeline_path = "pipeline"
        if os.path.exists(pipeline_path):
            health_status['pipeline_structure'] = True
            logger.info("✅ Pipeline directory structure found")
            
            # Check model files in pipeline directory
            logger.info("🤖 Checking model files...")
            missing_models = []
            for model_file in self.model_files:
                model_path = os.path.join(pipeline_path, model_file)
                if not os.path.exists(model_path):
                    missing_models.append(model_file)
            
            if missing_models:
                logger.warning(f"⚠️ Missing model files: {missing_models}")
                logger.warning("   Models may be loaded dynamically by the application")
            else:
                logger.info("✅ All model files found in pipeline directory")
                health_status['models_loaded'] = True
        else:
            logger.warning("⚠️ Pipeline directory not found in current location")
            logger.warning("   This script should be run from the application root directory")
        
        # Check Python dependencies
        logger.info("📦 Checking Python dependencies...")
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'xgboost', 'flask', 'pickle']
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'pickle':
                    import pickle
                else:
                    __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {missing_packages}")
            logger.error("   Please install: pip install " + " ".join(missing_packages))
        else:
            logger.info("✅ All required packages installed")
        
        health_status['dependencies_ok'] = len(missing_packages) == 0
        
        # Test API responsiveness
        logger.info("🔗 Testing API endpoints...")
        try:
            test_response = requests.get(f"{self.base_url}/", timeout=5)
            if test_response.status_code == 200:
                health_status['api_responsive'] = True
                logger.info("✅ API endpoints are responsive")
            else:
                logger.warning(f"⚠️ API returned status: {test_response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ API test failed: {e}")
        
        # Summary
        logger.info("\n📋 Health Check Summary:")
        for check, status in health_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {check.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}")
        
        overall_health = all(health_status.values())
        if overall_health:
            logger.info("🎉 System is healthy and ready for demonstrations!")
        else:
            logger.warning("⚠️ Some health checks failed. Proceeding with available functionality.")
        
        return overall_health

    def generate_sample_sales_data(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Generate realistic sample sales forecasting data
        
        Args:
            num_samples: Number of sample records to generate
            
        Returns:
            List of sample sales data dictionaries
        """
        logger.info(f"🎲 Generating {num_samples} sample sales forecasting records...")
        
        # Sample product categories and channels
        product_categories = ["Smartphones", "Tablets", "Wearables", "Audio", "Home_Appliances"]
        sales_channels = ["Online", "Retail", "Partner", "Enterprise"]
        regions = ["Seoul", "Busan", "Daegu", "Incheon", "Gwangju", "Daejeon"]
        
        samples = []
        np.random.seed(42)  # For reproducible results
        
        base_date = datetime.now() - timedelta(days=365)  # Start from a year ago
        
        for i in range(num_samples):
            # Generate realistic time series data
            days_offset = i * 30  # Monthly intervals
            current_date = base_date + timedelta(days=days_offset)
            
            # Seasonal and trend factors
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * (current_date.month - 1) / 12)
            trend_factor = 1 + 0.02 * i  # Slight growth trend
            
            # Base performance with realistic variation
            base_performance = np.random.uniform(0.7, 1.2)
            
            sample = {
                'date': current_date.strftime('%Y-%m-%d'),
                'product_category': np.random.choice(product_categories),
                'sales_channel': np.random.choice(sales_channels),
                'region': np.random.choice(regions),
                'historical_sales': int(np.random.uniform(10000, 100000) * base_performance * seasonal_factor * trend_factor),
                'units_sold': int(np.random.uniform(50, 500) * base_performance * seasonal_factor),
                'avg_selling_price': round(np.random.uniform(200, 2000), 2),
                'marketing_spend': round(np.random.uniform(2000, 20000) * seasonal_factor, 2),
                'promotion_intensity': round(np.random.uniform(0.0, 1.0), 2),
                'competitor_price_ratio': round(np.random.uniform(0.8, 1.3), 2),
                'inventory_level': int(np.random.uniform(100, 1000)),
                'customer_satisfaction': round(np.random.uniform(3.5, 5.0), 1),
                'market_share': round(np.random.uniform(0.1, 0.4), 3),
                'economic_index': round(np.random.uniform(95, 105), 1),
                'seasonality_factor': round(seasonal_factor, 2),
                'weather_impact': round(np.random.uniform(0.8, 1.2), 2),
                'launch_flag': np.random.choice([0, 1], p=[0.8, 0.2])  # 20% chance of product launch
            }
            samples.append(sample)
        
        logger.info("✅ Sample sales data generated successfully")
        return samples

    def demonstrate_individual_prediction(self, sample_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Demonstrate individual sales uplift prediction
        
        Args:
            sample_data: Sample sales data for prediction
            
        Returns:
            Prediction results or None if failed
        """
        logger.info("🔮 Demonstrating Individual Sales Uplift Prediction...")
        logger.info(f"📊 Product: {sample_data.get('product_category', 'Unknown')} in {sample_data.get('region', 'Unknown')}")
        
        try:
            # Prepare data for API (remove non-feature fields if needed)
            feature_data = sample_data.copy()
            
            logger.info(f"🎯 Input Features: {len(feature_data)} parameters")
            logger.info(f"📅 Date: {sample_data.get('date', 'N/A')}")
            logger.info(f"🛒 Sales Channel: {sample_data.get('sales_channel', 'N/A')}")
            
            # Make prediction request
            response = requests.post(
                self.endpoints['predict'],
                json=feature_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Sales prediction successful!")
                
                # Extract and display results
                if 'prediction' in result:
                    prediction = result['prediction']
                    confidence = result.get('confidence', 'N/A')
                    
                    logger.info(f"📈 Predicted Sales: {prediction}")
                    logger.info(f"🎯 Confidence: {confidence}")
                    
                    # Calculate uplift if baseline is provided
                    baseline = sample_data.get('historical_sales', 0)
                    if baseline > 0 and isinstance(prediction, (int, float)):
                        uplift_pct = ((prediction - baseline) / baseline) * 100
                        logger.info(f"📊 Sales Uplift: {uplift_pct:.2f}%")
                    
                    # Additional metrics if available
                    if 'uplift_category' in result:
                        logger.info(f"🏆 Uplift Category: {result['uplift_category']}")
                    
                    if 'recommendations' in result:
                        logger.info("💡 Recommendations:")
                        for rec in result['recommendations']:
                            logger.info(f"   • {rec}")
                    
                    return result
                else:
                    logger.error("❌ Unexpected response format")
                    logger.error(f"Response: {result}")
                    
            else:
                logger.error(f"❌ Prediction failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error during prediction: {e}")
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
        
        return None

    def demonstrate_batch_prediction(self, sample_data_list: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """
        Demonstrate batch sales uplift prediction
        
        Args:
            sample_data_list: List of sample sales data for predictions
            
        Returns:
            List of prediction results or None if failed
        """
        logger.info(f"📊 Demonstrating Batch Sales Uplift Prediction for {len(sample_data_list)} records...")
        
        try:
            # Create DataFrame for CSV upload
            df = pd.DataFrame(sample_data_list)
            csv_file = 'temp_batch_sales.csv'
            df.to_csv(csv_file, index=False)
            
            logger.info(f"📄 Created temporary CSV file: {csv_file}")
            logger.info(f"📋 Batch size: {len(sample_data_list)} records")
            logger.info(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
            
            # Upload file for batch prediction
            with open(csv_file, 'rb') as f:
                files = {'file': (csv_file, f, 'text/csv')}
                response = requests.post(
                    self.endpoints['upload'],
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                logger.info("✅ Batch prediction successful!")
                
                # Parse results
                if response.headers.get('content-type', '').startswith('text/html'):
                    logger.info("📊 Received HTML results page")
                    # In a real scenario, you might parse HTML or get JSON
                    return [{"status": "success", "message": "Batch processed via web interface"}]
                else:
                    result = response.json()
                    predictions = result.get('predictions', [])
                    logger.info(f"📈 Processed {len(predictions)} predictions")
                    
                    # Summary statistics
                    if predictions and isinstance(predictions[0], dict) and 'prediction' in predictions[0]:
                        pred_values = [p['prediction'] for p in predictions if 'prediction' in p]
                        if pred_values:
                            logger.info(f"📊 Prediction Range: ${min(pred_values):,.0f} - ${max(pred_values):,.0f}")
                            logger.info(f"📈 Average Prediction: ${np.mean(pred_values):,.0f}")
                    
                    return predictions
                    
            else:
                logger.error(f"❌ Batch prediction failed with status {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # Cleanup
            if os.path.exists(csv_file):
                os.remove(csv_file)
                
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            # Cleanup on error
            if 'csv_file' in locals() and os.path.exists(csv_file):
                os.remove(csv_file)
        
        return None

    def demonstrate_api_endpoints(self):
        """Demonstrate and document all available API endpoints"""
        logger.info("🔗 Demonstrating API Endpoints...")
        
        api_docs = {
            "Sales Uplift Forecasting API": {
                "base_url": self.base_url,
                "endpoints": [
                    {
                        "path": "/",
                        "method": "GET",
                        "description": "Main application interface",
                        "example": f"curl -X GET {self.base_url}/"
                    },
                    {
                        "path": "/predict",
                        "method": "POST",
                        "description": "Individual sales uplift prediction",
                        "content_type": "application/json",
                        "example": f"""curl -X POST {self.base_url}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"historical_sales": 50000, "marketing_spend": 10000, "promotion_intensity": 0.5, "product_category": "Smartphones"}}'"""
                    },
                    {
                        "path": "/upload",
                        "method": "POST",
                        "description": "Batch prediction via CSV upload",
                        "content_type": "multipart/form-data",
                        "example": f"curl -X POST {self.base_url}/upload -F 'file=@sales_data.csv'"
                    }
                ]
            }
        }
        
        logger.info("📚 API Documentation:")
        for api_name, api_info in api_docs.items():
            logger.info(f"\n🔷 {api_name}")
            logger.info(f"   Base URL: {api_info['base_url']}")
            logger.info(f"   Endpoints:")
            
            for endpoint in api_info['endpoints']:
                logger.info(f"\n   📍 {endpoint['method']} {endpoint['path']}")
                logger.info(f"      Description: {endpoint['description']}")
                if 'content_type' in endpoint:
                    logger.info(f"      Content-Type: {endpoint['content_type']}")
                logger.info(f"      Example:")
                logger.info(f"      {endpoint['example']}")

    def run_comprehensive_demo(self):
        """Run the complete demonstration suite"""
        start_time = time.time()
        
        try:
            # Print banner
            self.print_banner()
            
            # Health check
            logger.info("🏥 Step 1: System Health Check")
            health_ok = self.check_system_health()
            
            if not health_ok:
                logger.warning("⚠️ Some health checks failed, but continuing with demo...")
            
            # Generate sample data
            logger.info("\n🎲 Step 2: Sample Sales Data Generation")
            sample_sales = self.generate_sample_sales_data(5)
            
            # Display sample data overview
            if sample_sales:
                logger.info("📊 Sample Data Overview:")
                for i, sample in enumerate(sample_sales[:3]):  # Show first 3
                    logger.info(f"   Record {i+1}: {sample['product_category']} - ${sample['historical_sales']:,}")
            
            # Individual prediction demo
            logger.info("\n🔮 Step 3: Individual Sales Uplift Prediction Demo")
            if sample_sales:
                individual_result = self.demonstrate_individual_prediction(sample_sales[0])
                time.sleep(2)  # Brief pause for readability
            
            # Batch prediction demo
            logger.info("\n📊 Step 4: Batch Sales Uplift Prediction Demo")
            batch_results = self.demonstrate_batch_prediction(sample_sales)
            time.sleep(2)
            
            # API documentation
            logger.info("\n🔗 Step 5: API Endpoints Documentation")
            self.demonstrate_api_endpoints()
            
            # Summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"\n{'='*80}")
            logger.info("🎉 Sales Uplift Forecasting Demo Complete!")
            logger.info(f"⏱️ Total Duration: {duration:.2f} seconds")
            logger.info(f"📊 Application: {self.app_name}")
            logger.info(f"🌐 Web Interface: {self.base_url}")
            logger.info(f"📋 Log File: sales_uplift_demo.log")
            logger.info("="*80")
            
            # Integration instructions
            logger.info("\n🔧 Integration Instructions for Samsung Mentors:")
            logger.info("1. Ensure application is running from pipeline directory:")
            logger.info("   cd 'Sell-out performance forecasting (sales uplift)/pipeline'")
            logger.info("   python app.py")
            logger.info("2. Access web interface at: http://127.0.0.1:7003")
            logger.info("3. Use API endpoints for programmatic access")
            logger.info("4. Upload CSV files for batch processing")
            logger.info("5. Review logs for detailed performance metrics")
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ Demo interrupted by user")
        except Exception as e:
            logger.error(f"\n❌ Demo failed with error: {e}")
            raise
        
        return True

def main():
    """Main execution function"""
    print("🚀 Starting Sales Uplift Forecasting Demo System...")
    
    # Initialize demo system
    demo = SalesUpliftDemoSystem()
    
    # Run comprehensive demonstration
    try:
        demo.run_comprehensive_demo()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
