# foodwasteAI
AI Project to Reduce Food Waste in Woodstock School, Mussoorie.
Code written in MATLAB/Octave.

Started in September 21, 2019.

## **Status** | Version: 0.4
Current program: `predict2.m`\
Legacy version: `predict1_regularized.m`

## Objective
Create a supply and demand optimization algorithm to reduce food waste in the school.

## Data Collection
Data collection started September 2nd, 2019 and will be updated weekly.

There is existing but meager data from April 2017 to July 2017.

## File Navigation

### Data
Folders breakfast, lunch, and dinner include txt and csv files of data. The students txt files are self-explanatory.

The txt files that end with [meal].txt in /dinner are comma separated, similar to csv files. The four columns in order from left to right are
rice supply (in kg), dal supply (in kg), students (estimated by number of trays used), and total wastage.

Example:
> **Rice (X1)**, **Dal (X2)**, **Students (X3)**, **Total Wastage (y)**
>
> 20,10,100,15
>
> 14,6,160,3
>
> 20,10,260,11
>
> 3.5,4.8,36,2.3

### Main Programs
Starts with `predict`, all legacy version present.

*Legacy versions may be archived in another folder in the future.*

Version order: from 0, incremented by 1 unless special additions – e.g., `predict1_regularized.m` from `predict1.m`.

## Current Tasks
+ Make simple linear regression models with a few factors
+ **Simulate fake data**

## Future Tasks
+ Recommender system
