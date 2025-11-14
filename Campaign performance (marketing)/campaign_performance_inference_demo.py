#!/usr/bin/env python3
"""
Campaign Performance Marketing - Comprehensive Inference & Demo Script
=====================================================================

PRISM Worklet 8 - Samsung Project
Advanced AI-powered store performance analysis and marketing campaign optimization

This script provides comprehensive demonstrations of the Campaign Performance system
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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('campaign_performance_demo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CampaignPerformanceDemoSystem:
    """
    Comprehensive demo system for Campaign Performance Marketing application
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:7002"):
        """
        Initialize the demo system
        
        Args:
            base_url: Base URL for the Campaign Performance application
        """
        self.base_url = base_url
        self.app_name = "Campaign Performance Marketing"
        self.version = "2.0"
        
        # Expected model files
        self.model_files = [
            'catboost_model.pkl',
            'lgbm_model.pkl', 
            'ridge_model.pkl',
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
📈 Advanced AI-powered Store Performance Analysis & Marketing Optimization
{'='*80}

🎯 Samsung Project - Preparing and Inspiring Student Minds
📊 Unified AI Platform for Business Intelligence

Application: {self.app_name}
Version: {self.version}
Port: 7002
Base URL: {self.base_url}

{'='*80}
"""
        print(banner)
        logger.info("Campaign Performance Demo System Started")

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
            'api_responsive': False
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
            logger.error(f"   cd 'Campaign performance (marketing)'")
            logger.error(f"   python app.py")
            return False
        except Exception as e:
            logger.error(f"❌ Server check failed: {e}")
            return False
        
        # Check model files
        logger.info("🤖 Checking model files...")
        missing_models = []
        for model_file in self.model_files:
            if not os.path.exists(model_file):
                missing_models.append(model_file)
        
        if missing_models:
            logger.warning(f"⚠️ Missing model files: {missing_models}")
            logger.warning("   Models may be loaded dynamically by the application")
        else:
            logger.info("✅ All model files found")
        
        health_status['models_loaded'] = len(missing_models) == 0
        
        # Check Python dependencies
        logger.info("📦 Checking Python dependencies...")
        required_packages = ['pandas', 'numpy', 'scikit-learn', 'catboost', 'lightgbm', 'flask']
        missing_packages = []
        
        for package in required_packages:
            try:
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
            # Test health endpoint if available
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

    def generate_sample_store_data(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Generate realistic sample store performance data
        
        Args:
            num_samples: Number of sample records to generate
            
        Returns:
            List of sample store performance dictionaries
        """
        logger.info(f"🎲 Generating {num_samples} sample store performance records...")
        
        # Sample store names and locations
        store_names = [
            "Samsung Experience Store Seoul", "Galaxy Hub Busan", "Innovation Center Daegu",
            "Smart Store Incheon", "Tech Plaza Gwangju", "Digital Store Daejeon",
            "Mobile Center Ulsan", "Electronics Hub Suwon", "Future Store Goyang",
            "Premium Outlet Seongnam"
        ]
        
        # Sample product categories
        categories = ["Smartphones", "Tablets", "Wearables", "Audio", "Home Appliances"]
        
        samples = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(num_samples):
            # Generate realistic store performance metrics
            base_performance = np.random.uniform(0.6, 0.95)  # Base performance factor
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * (i % 12) / 12)  # Seasonal variation
            
            sample = {
                'store_id': f"ST{1000 + i:04d}",
                'store_name': np.random.choice(store_names),
                'category': np.random.choice(categories),
                'monthly_sales': int(np.random.uniform(50000, 500000) * base_performance * seasonal_factor),
                'customer_visits': int(np.random.uniform(1000, 10000) * base_performance),
                'conversion_rate': round(np.random.uniform(0.1, 0.4) * base_performance, 3),
                'avg_transaction_value': round(np.random.uniform(200, 1500) * base_performance, 2),
                'marketing_spend': round(np.random.uniform(5000, 50000), 2),
                'promotions_count': int(np.random.uniform(2, 15)),
                'staff_count': int(np.random.uniform(3, 20)),
                'floor_space_sqm': int(np.random.uniform(100, 1000)),
                'location_score': round(np.random.uniform(0.3, 1.0), 2),
                'competition_density': round(np.random.uniform(0.1, 0.8), 2),
                'customer_satisfaction': round(np.random.uniform(3.5, 5.0), 1),
                'return_rate': round(np.random.uniform(0.02, 0.15), 3)
            }
            samples.append(sample)
        
        logger.info("✅ Sample data generated successfully")
        return samples

    def demonstrate_individual_prediction(self, sample_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Demonstrate individual store performance prediction
        
        Args:
            sample_data: Sample store data for prediction
            
        Returns:
            Prediction results or None if failed
        """
        logger.info("🔮 Demonstrating Individual Store Performance Prediction...")
        logger.info(f"📊 Store: {sample_data.get('store_name', 'Unknown')} (ID: {sample_data.get('store_id', 'N/A')})")
        
        try:
            # Prepare data for API (remove non-feature fields)
            feature_data = sample_data.copy()
            exclude_fields = ['store_id', 'store_name']
            for field in exclude_fields:
                feature_data.pop(field, None)
            
            logger.info(f"🎯 Input Features: {len(feature_data)} parameters")
            
            # Make prediction request
            response = requests.post(
                self.endpoints['predict'],
                json=feature_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Prediction successful!")
                
                # Extract and display results
                if 'prediction' in result:
                    prediction = result['prediction']
                    confidence = result.get('confidence', 'N/A')
                    
                    logger.info(f"📈 Performance Score: {prediction}")
                    logger.info(f"🎯 Confidence: {confidence}")
                    
                    # Additional metrics if available
                    if 'performance_category' in result:
                        logger.info(f"🏆 Performance Category: {result['performance_category']}")
                    
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
        Demonstrate batch store performance prediction
        
        Args:
            sample_data_list: List of sample store data for predictions
            
        Returns:
            List of prediction results or None if failed
        """
        logger.info(f"📊 Demonstrating Batch Store Performance Prediction for {len(sample_data_list)} stores...")
        
        try:
            # Prepare batch data
            batch_data = []
            for sample in sample_data_list:
                feature_data = sample.copy()
                exclude_fields = ['store_id', 'store_name']
                for field in exclude_fields:
                    feature_data.pop(field, None)
                batch_data.append(feature_data)
            
            # Create DataFrame for CSV upload simulation
            df = pd.DataFrame(batch_data)
            csv_file = 'temp_batch_stores.csv'
            df.to_csv(csv_file, index=False)
            
            logger.info(f"📄 Created temporary CSV file: {csv_file}")
            logger.info(f"📋 Batch size: {len(batch_data)} stores")
            
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
                    logger.info(f"📈 Processed {len(result.get('predictions', []))} predictions")
                    return result.get('predictions', [])
                    
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
            "Campaign Performance Marketing API": {
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
                        "description": "Individual store performance prediction",
                        "content_type": "application/json",
                        "example": f"""curl -X POST {self.base_url}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"monthly_sales": 150000, "customer_visits": 3000, "conversion_rate": 0.25, "marketing_spend": 15000}}'"""
                    },
                    {
                        "path": "/upload",
                        "method": "POST", 
                        "description": "Batch prediction via CSV upload",
                        "content_type": "multipart/form-data",
                        "example": f"curl -X POST {self.base_url}/upload -F 'file=@stores_data.csv'"
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
            logger.info("\n🎲 Step 2: Sample Data Generation")
            sample_stores = self.generate_sample_store_data(5)
            
            # Individual prediction demo
            logger.info("\n🔮 Step 3: Individual Store Performance Prediction Demo")
            if sample_stores:
                individual_result = self.demonstrate_individual_prediction(sample_stores[0])
                time.sleep(2)  # Brief pause for readability
            
            # Batch prediction demo
            logger.info("\n📊 Step 4: Batch Store Performance Prediction Demo")
            batch_results = self.demonstrate_batch_prediction(sample_stores)
            time.sleep(2)
            
            # API documentation
            logger.info("\n🔗 Step 5: API Endpoints Documentation")
            self.demonstrate_api_endpoints()
            
            # Summary
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"\n{'='*80}")
            logger.info("🎉 Campaign Performance Marketing Demo Complete!")
            logger.info(f"⏱️ Total Duration: {duration:.2f} seconds")
            logger.info(f"📊 Application: {self.app_name}")
            logger.info(f"🌐 Web Interface: {self.base_url}")
            logger.info(f"📋 Log File: campaign_performance_demo.log")
            logger.info("='*80}")
            
            # Integration instructions
            logger.info("\n🔧 Integration Instructions for Samsung Mentors:")
            logger.info("1. Ensure application is running: python app.py")
            logger.info("2. Access web interface at: http://127.0.0.1:7002")
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
    print("🚀 Starting Campaign Performance Marketing Demo System...")
    
    # Initialize demo system
    demo = CampaignPerformanceDemoSystem()
    
    # Run comprehensive demonstration
    try:
        demo.run_comprehensive_demo()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
