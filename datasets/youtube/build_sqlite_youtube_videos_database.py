import sqlite3

database_filepath = '/home/anton/work/fitMate/datasets/rep_fit_yt_links.db'

# conn = sqlite3.connect(database_filepath)
conn = sqlite3.connect(':memory:')

c = conn.cursor()
c.execute("""""")