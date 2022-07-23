import time
import datetime

import requests
import os

import pandas as pd
import numpy as np

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "notebook_connected"

import matplotlib.pyplot as plt

from prophet import Prophet

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

os.listdir()
