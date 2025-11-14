#!/usr/bin/env python3
"""
Browser Form Test Script
Tests the browser-based form to ensure JavaScript form submission works correctly
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import sys

def test_browser_form():
    """Test the form in a real browser environment"""
    
    print("🌐 BROWSER FORM TESTING")
    print("=" * 50)
    
    try:
        # Try Chrome first, then Safari/Firefox
        try:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')  # Run in background
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            print("✅ Chrome WebDriver initialized")
            
        except Exception as e:
            print(f"⚠️  Chrome not available: {e}")
            # Fallback to Safari (macOS default)
            driver = webdriver.Safari()
            print("✅ Safari WebDriver initialized")
    
    except Exception as e:
        print(f"❌ Browser setup failed: {e}")
        print("💡 Install ChromeDriver or use Safari for testing")
        return False
    
    try:
        # Navigate to the form
        url = "http://127.0.0.1:5001"
        print(f"🔗 Loading: {url}")
        driver.get(url)
        
        # Wait for form to load
        wait = WebDriverWait(driver, 10)
        form = wait.until(EC.presence_of_element_located((By.TAG_NAME, "form")))
        print("✅ Form loaded successfully")
        
        # Fill out the form with test data
        test_data = {
            'interest_rate': '3.75',
            'unpaid_principal_bal': '200000',
            'Loan_term': '18',
            'loan_to_value': '80',
            'source': 'Y',
            'loan_purpose': 'C86',
            'number_of_borrowers': '2',
            'insurance_percent': '10',
            'debt_to_income_ratio': '29',
            'borrower_credit_score': '809',
            'co-borrower_credit_score': '812',
            'Annual_Income': '66207',  # Note: underscore version
            'Age': '51',
            'NumberOfDependents': '2',
            'EducationLevel': "Bachelor's",
            'MaritalStatus': 'Married',
            'Gender': 'Female',
            'EmploymentStatus': 'Employed',
            'total_on_time_payments': '7',
            'total_late_payments': '2',
            'avg_payment_delay': '18.5',
            'current_dpd': '0'
        }
        
        print(f"📝 Filling {len(test_data)} form fields...")
        
        # Fill text and number inputs
        for field_name, value in test_data.items():
            try:
                if field_name in ['source', 'loan_purpose', 'EducationLevel', 'MaritalStatus', 'Gender', 'EmploymentStatus']:
                    # Handle select dropdowns
                    select_element = driver.find_element(By.NAME, field_name)
                    select = Select(select_element)
                    select.select_by_value(value)
                    print(f"  ✅ {field_name}: {value} (dropdown)")
                else:
                    # Handle regular inputs
                    input_element = driver.find_element(By.NAME, field_name)
                    input_element.clear()
                    input_element.send_keys(value)
                    print(f"  ✅ {field_name}: {value}")
                    
            except Exception as e:
                print(f"  ❌ {field_name}: Failed to fill ({e})")
        
        # Submit the form
        print("\n🚀 Submitting form...")
        submit_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        submit_button.click()
        
        # Wait for results
        try:
            results_section = wait.until(
                EC.visibility_of_element_located((By.ID, "results"))
            )
            print("✅ Results section appeared!")
            
            # Check for specific result elements
            risk_level = driver.find_element(By.ID, "riskLevel").text
            prob_result = driver.find_element(By.ID, "probabilityResult").text
            
            print(f"📊 Risk Level: {risk_level}")
            print(f"📊 Probability: {prob_result}")
            
            return True
            
        except TimeoutException:
            print("❌ Results did not appear within 10 seconds")
            
            # Check for errors
            try:
                error_section = driver.find_element(By.ID, "errorSection")
                if error_section.is_displayed():
                    error_text = error_section.text
                    print(f"❌ Error displayed: {error_text}")
                else:
                    print("⚠️  No error section visible")
            except:
                print("⚠️  No error section found")
            
            # Check browser console for JavaScript errors
            logs = driver.get_log('browser')
            if logs:
                print("🔍 Browser Console Logs:")
                for log in logs:
                    print(f"  {log['level']}: {log['message']}")
            
            return False
    
    except Exception as e:
        print(f"❌ Browser test failed: {e}")
        return False
    
    finally:
        driver.quit()
        print("🔚 Browser closed")

def main():
    """Run the browser test"""
    success = test_browser_form()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ BROWSER FORM TEST: PASSED")
        print("   Form submission and results display working correctly")
    else:
        print("❌ BROWSER FORM TEST: FAILED") 
        print("   Check browser console and server logs for details")
    
    print("\n💡 MANUAL TEST STEPS:")
    print("   1. Open http://127.0.0.1:5001 in your browser")
    print("   2. Fill out the loan application form")
    print("   3. Click 'Get Risk Assessment'")
    print("   4. Verify results appear below the form")
    print("=" * 50)

if __name__ == "__main__":
    main()
