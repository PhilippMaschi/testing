# download tif files from a website

import numpy as np
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
from pathlib import Path
import requests


fist_numbers = np.arange(519, 936)
second_numbers = np.arange(1, 10)
last_numbers = np.arange(1, 10)

folder = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/building-stock-analysis/T3.1-dynamic-analysis/Case-study-II-III-PV-analysis/solar-panel-classifier/new_data/input_tifs/")
for first_nr in fist_numbers:
    for second_nr in second_numbers:
        for last_nr in last_numbers:

            url_text = f"https://descargas.icv.gva.es/dcd/02_imagen/02_ortofoto/2023_CVAL_0025/01_Imagen/01_25830_01_8bits_02_RGBI_02_TIFJPG/020201_2023CVAL0025_25830_8bits_RGBI_{first_nr}_{second_nr}-{last_nr}.tif"
            response = requests.get(url_text, stream=True)

            if response.status_code == 200:
                # Define the file name based on the URL or your desired name
                file_name = url_text.split("/")[-1]  # Extracts the last part of the URL as file name

                # Open a file in binary-write mode and save the content in chunks
                with open(file_name, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # Filter out keep-alive new chunks
                            file.write(chunk)
                    print(f"Downloaded file saved as: {file_name}")



def selenium_try():  # doesnt work for the website
    # Specify your download directory
    download_dir = r"X:\projects4\workspace_philippm\building-stock-analysis\T3.1-dynamic-analysis\Case-study-II-III-PV-analysis\solar-panel-classifier\new_data\input_tifs"



    geckodriver_path = r"C:\DRIVERS\WebDriver\geckodriver.exe"
    service = Service(executable_path=geckodriver_path)


    # Set the download directory in browser preferences
    firefox_options = Options()  
    firefox_options.headless = False 

    firefox_options.set_preference("browser.download.folderList", 2)  # Use custom download path
    firefox_options.set_preference("browser.download.dir", download_dir)
    firefox_options.set_preference("browser.helperApps.neverAsk.saveToDisk", "image/tiff")  # MIME type for TIF files
    firefox_options.set_preference("pdfjs.disabled", True)  # Disable PDF viewer if it's relevant


    driver = webdriver.Firefox(service=service, options=firefox_options)


    # Open the webpage

    url = "https://descargas.icv.gva.es/00/geoprocesos/descarga/descargaHoja.php"
    initial_page_url = "https://geocataleg.gva.es/#/"  # Replace with the actual URL
    driver.get(initial_page_url)
    action = ActionChains(driver)

    # click on ortofotos:
    element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 
                                            ".pb-3 > div:nth-child(2)"))
        )
    action.click(element)
    action.perform()

    previous_height = driver.execute_script("return document.body.scrollHeight")
    # Loop to scroll down until the element is found and interactable
    while True:
        try:
            # Try to locate the button after scrolling
            ortofoto_button = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 
                                                "div.list--group__container:nth-child(69) > div:nth-child(1) > li:nth-child(1) > a:nth-child(1) > div:nth-child(2) > div:nth-child(1)"))
            )
            
            # Scroll the element into view and click it
            driver.execute_script("arguments[0].scrollIntoView(true);", ortofoto_button)
            action.move_to_element(ortofoto_button).click().perform()
            break  # Exit the loop after clicking the element

        except Exception as e:
            # Scroll down if the element isn't interactable yet
            driver.execute_script("window.scrollBy(0, window.innerHeight);")
            
            # Wait for the page to load additional content
            time.sleep(1)

            # Check if we've reached the bottom of the page without finding the element
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == previous_height:
                print("Reached the bottom of the page without finding the element.")
                break  # Exit loop if no more content loads
            previous_height = new_height


    download_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR,
                                            "div.ol-layer:nth-child(2) > canvas:nth-child(1)"))
        )
    action.click(element)
    action.perform()


    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".pb-3 > div:nth-child(2)"))
    )

    driver.execute_script("arguments[0].scrollIntoView(true);", tile)
    # now the page with the orthophotos is loaded:
    tiles = driver.find_elements(By.CSS_SELECTOR, "div.ol-layer:nth-child(2) > canvas:nth-child(1)")

    tiles = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.ol-layer:nth-child(2) > canvas:nth-child(1)"))
    )
    canvases = driver.find_elements(By.CSS_SELECTOR, "div.ol-layer:nth-child(2) > canvas:nth-child(1)")
    wait = WebDriverWait(driver, 10)

    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, ".ol-overlaycontainer .ol-layer img"))
    )
    tiles = driver.find_elements(By.CSS_SELECTOR, ".ol-overlaycontainer .ol-layer")

    tile=tiles[0]
    action.click(tile)
    action.perform()

    driver.execute_script("arguments[0].scrollIntoView(true);", element)
    action.move_to_element(tile).click(tile).perform()

    iframe = driver.find_element(By.TAG_NAME, "iframe") 
    print(driver.page_source)

    tiles = driver.find_elements(By.CSS_SELECTOR, "a.list__tile")

    tiles = driver.find_elements(By.XPATH, "//div[@id='map']//canvas")

    canvas_element = wait.until(EC.presence_of_element_located((By.XPATH, "//div[@id='map']//canvas")))
    driver.execute_script("arguments[0].scrollIntoView();", canvas_element)  # Scroll into view
    driver.execute_script("arguments[0].click();", canvas_element)  # Simulate a click with JavaScript

    iframes = driver.find_elements(By.TAG_NAME, "iframe")

    for tile in tiles:
        try:
            tile.click()  # Click on the tile
            # Wait for the download button to become active
            download_button = wait.until(EC.element_to_be_clickable((By.ID, 'botonEnviar')))
            download_button.click()  # Click the download button
            time.sleep(5)  # Add a delay to allow the download to process before moving to the next tile
        except Exception as e:
            print(f"Error clicking tile or downloading: {e}")

    # After processing all the tiles
    driver.quit()

    print("All downloads completed.")


