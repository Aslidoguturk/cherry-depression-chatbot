import subprocess
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


def test_ui_loads():
    process = subprocess.Popen(["streamlit", "run", "chtr3.py"], stdout=subprocess.PIPE)
    time.sleep(90)  # Wait for Streamlit to start

    driver = None
    try:
        driver = webdriver.Chrome()
        driver.get("http://localhost:8501")

        WebDriverWait(driver, 10).until(
            ec.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Meet Cherry')]"))
        )

        assert "Meet Cherry" in driver.page_source

    finally:
        if driver:
            driver.quit()
        process.terminate()
