The entire code of our project is split into two files:
1. team_6_acs_code.py
2. team_6_model_code.py

First run the file 'team_6_acs_code.py'. This code uses the following files gathered from different sources:
i. Allegheny_County_Zip_Code_Boundaries.csv (has zip codes for Allegheny county)
ii. Zipcode-ZCTA-Population-Density-And-Area-Unsorted.csv (has area of each ZIP of the US in sq. miles)

Upon running these codes, you would generate a new csv called 'zip_level_data.csv'

Then run the file 'team_6_model_code.py'. This code uses the following files:
i. zip_level_data.csv (has household and pop_density for each zip)
ii. Distance Matrix.csv (has distance of each zip from every candidate POD)
iii. POD_capacity.csv (has capacity of each POD)

The code results in 3 following curves:
i. Model_1_pareto.png (shows total distance travelled by households for each value of alpha)
ii. Model_2_pareto.png (shows maximum distance travelled by any households for each value of alpha)
iii. global_vs_local.png (shows mean distance between zips and their assigned POD that would get infected in a local vs global scenario for each value of alpha)