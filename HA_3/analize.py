import json
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange, AutoDateLocator, MinuteLocator
import numpy as np


with open('data-2017-05.json') as json_file:
    data = json.load(json_file)

activity = []
ac_time = []

for entry in data:
    try:
        tmp_t = datetime.strptime(entry['Creation Date'], '%m/%d/%y %I:%M:%S %p PDT')
        tmp_t = tmp_t + timedelta(hours=10)
        activity.append(tmp_t)
        ac_time.append(tmp_t.time())
    except:
        None

y = np.ones_like(ac_time)
f1, ax = plt.subplots()
ax.hist(ac_time, 47)
ax.xaxis.set_major_locator(AutoDateLocator())
ax.xaxis.set_minor_locator(AutoDateLocator())
ax.fmt_xdate = DateFormatter('%H')
f1.autofmt_xdate()
plt.show()
