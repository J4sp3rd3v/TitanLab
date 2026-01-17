import os
import time
import re
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==============================================================================
# TITAN AUTO-LAUNCHER (EXPERIMENTAL)
# Uses Chrome to drive Google Colab automatically
# ==============================================================================

print("👾 TITAN AUTO-LAUNCHER INITIATED")
print("   -> Launching Chrome...")

# Setup Chrome Options
options = webdriver.ChromeOptions()
# We need to use a user data dir to save login session if possible, 
# but usually Google blocks automation logins. 
# We will ask user to login if not detected.
user_data_dir = os.path.join(os.getcwd(), "chrome_profile")
options.add_argument(f"user-data-dir={user_data_dir}")
options.add_argument("--start-maximized")
# options.add_argument("--headless") # CANNOT BE HEADLESS for Colab

try:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
except Exception as e:
    print(f"<!> CHROME DRIVER ERROR: {e}")
    print("   -> Please install Google Chrome.")
    input("Press Enter to exit...")
    exit()

COLAB_URL = "https://colab.research.google.com/github/J4sp3rd3v/TitanLab/blob/main/TITAN_VIRAL_LAB.ipynb"

print(f"🚀 OPENING COLAB: {COLAB_URL}")
driver.get(COLAB_URL)

print("⏳ Waiting for page load...")
time.sleep(5)

# Check if logged in
if "Sign in" in driver.title or "Google Accounts" in driver.title:
    print("\n⚠️  GOOGLE LOGIN REQUIRED")
    print("   -> Please log in to your Google Account in the opened Chrome window.")
    print("   -> Once logged in and the Notebook is visible, press ENTER here.")
    input("   [PRESS ENTER AFTER LOGIN] >> ")

print("⚡ STARTING RUNTIME...")

# Try to find "Runtime" menu or just "Run all" button (often hidden in menu)
# Shortcut for Run All is usually Ctrl+F9, but let's try to click the specific button.
# The "Play" button on the first cell is safer.

try:
    # Wait for the notebook to fully load
    WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CLASS_NAME, "cell")))
    
    print("   -> Finding 'Run All'...")
    # Send Ctrl+F9 to body
    body = driver.find_element(By.TAG_NAME, "body")
    body.send_keys(Keys.CONTROL, Keys.F9)
    print("   -> Sent 'Run All' command (Ctrl+F9).")
    
    # Handle "Warning: This notebook was not authored by Google"
    try:
        time.sleep(2)
        run_anyway = driver.find_element(By.XPATH, "//mwc-button[contains(., 'Run anyway')]")
        if run_anyway.is_displayed():
            run_anyway.click()
            print("   -> Clicked 'Run anyway'.")
    except:
        pass
        
except Exception as e:
    print(f"<!> AUTOMATION ERROR: {e}")
    print("   -> Please manually click 'Runtime' -> 'Run all' in the browser.")

print("\n📡 MONITORING OUTPUT FOR GRADIO LINK...")
print("   (This may take 2-3 minutes - Do not close Chrome)")

gradio_url = None
start_time = time.time()

while True:
    try:
        # Get page source (or just the output area)
        # We look for "Running on public URL: https://..."
        page_source = driver.page_source
        
        match = re.search(r"Running on public URL: (https://[a-z0-9]+\.gradio\.live)", page_source)
        if match:
            gradio_url = match.group(1)
            print(f"\n✅ FOUND URL: {gradio_url}")
            break
            
        if time.time() - start_time > 600: # 10 mins timeout
            print("<!> TIMEOUT: Gradio link not found after 10 minutes.")
            break
            
        time.sleep(2)
        print(".", end="", flush=True)
        
    except Exception as e:
        pass

if gradio_url:
    print("\n🚀 LAUNCHING LOCAL UI...")
    # Launch titan_remote.py and pass the URL if possible?
    # titan_remote.py expects user input. 
    # We can modify titan_remote.py to accept args, or just print it.
    
    # Let's write the URL to a file that titan_remote checks?
    # Or better, just launch it and let user paste it?
    # The user wanted "zero config".
    
    # Let's modify titan_remote.py to look for a 'last_session.url' file.
    with open("titan_connect.url", "w") as f:
        f.write(gradio_url)
        
    print("   -> Saved URL to titan_connect.url")
    print("   -> Starting Titan Remote...")
    
    os.system("python titan_remote.py")
    
else:
    print("\n<!> FAILED TO CAPTURE URL.")
    print("   -> Please copy it manually from Chrome and run 'python titan_remote.py'")

input("\n[PRESS ENTER TO CLOSE AUTOMATION]")
