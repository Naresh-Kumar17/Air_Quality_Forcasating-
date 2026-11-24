# PHASE 1 - Step 2: collect NEW data every hour
# Keep this running (laptop on) until the end of the project.
# Every hour it saves a reading to live_data.csv and Open-Meteo's own
# 24-hour forecast to forecast_log.csv (used for Check B in the notebook).
# To stop it: press Ctrl + C
#
# No laptop? The GitHub Action in .github/workflows/collect.yml does the same job every hour.

import time
from collect_once import collect_one_reading

while True:
    try:
        collect_one_reading()
    except Exception as error:
        # if the internet or API fails, show the problem and try again next time
        print("Problem this time:", error)

    # wait one hour (3600 seconds)
    time.sleep(3600)
