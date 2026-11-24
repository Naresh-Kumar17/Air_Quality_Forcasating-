# PHASE 1 - Step 1: download old data for Karachi
# Run this ONE time. It makes the file karachi_aq_history.csv
#
# history_days = how many days to download (more days = more seasons = better results)
# The air quality service has data from 2022, so up to about 1000 days is possible.
# 730 days (2 years) gives two examples of every season. It takes a few minutes to download.
# If the download stops early, the script says so and keeps what it got.

from data_download import get_history

history_days = 730

table = get_history(history_days)

if table is not None:
    table.to_csv("karachi_aq_history.csv", index=False)
    print("Rows and columns:", table.shape)
    print("First hour:", table["time"].min())
    print("Last hour:", table["time"].max())
    print(table.head())
    print(table.tail())
