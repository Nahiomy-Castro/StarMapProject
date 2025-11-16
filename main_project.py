from tkinter import *
from tkinter import messagebox
import astropy
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import ascii
import numpy as np
import matplotlib as plt

import db_connection as db

#db.create_MariaDB()

db.upload_MariaDB()

