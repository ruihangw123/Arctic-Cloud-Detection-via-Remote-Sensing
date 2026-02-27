#############
## Purpose ##
#############
#
# This is to run the main ipynb file and save it to a 
# notebook.
#
##########
## Ruff ##
##########
#
# Zach will be looking at code style as well using ruff
# You will need to run:
# ruff check code/clean.py code/models.py --output-format json
# 
# If I get errors as well, I should run
# ruff check code/clean.py code/models.py --fix
# 
# and it will automatically fix the style errors 
#
#############
## Content ##
#############

#!/bin/bash
conda activate stat214
# run clean.py
# command to run models.py
jupyter nbconvert --to notebook --execute --inplace lab1.ipynb