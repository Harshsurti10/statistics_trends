#importing necessary libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def readFile(a):
    '''
        This function is used to read a csv file in its original form and in a transposed version. 

        energy_use : variable for storing csv file
        t_energy_use : variable for storing transposed csv file
    '''
    
    energy_use = pd.read_csv("Energy_use.csv");
    energy_use = energy_use.drop(['Country Code', 'Indicator Name', \
                                    'Indicator Code', '1960', '1961', '1962', \
                                      '1963', '1964', '1965', '1966', '1967', \
                                      '1968', '1969', '1970', '1971', '1972', \
                                      '1973', '1974', '1975', '1976', '1977', \
                                      '1978', '1979', '1980', '1981', '1982', \
                                      '1983', '1984', '1985', '1986', '1987', \
                                      '1988', '1989', '2021'], axis=1)
    t_energy_use = pd.read_csv(a).T
    return energy_use, t_energy_use

# calling function to show dataframe
energy_use, t_energy_use = readFile("Energy_use.csv")
print("\nEnergy usage \n", energy_use)
print("\nTransposed Energy usage: ", t_energy_use)


# select certain rows using slicing method and iloc used for selecting fixed rows and all columns
new_energy_use = energy_use.iloc[[35, 81, 109, 115, 111, 119, 170]]
print("\nExtracted Energy usage: \n", new_energy_use)


# removing NaN values
drop_null = new_energy_use.dropna()
print("\nExtracted Energy usage after dropping NaN: \n", drop_null)


# removing first two lines
drop_null = drop_null.iloc[2:]
print("\nRemoving first two lines from Energy usage: \n", drop_null)


# indexing the "Country Name" column
index_energy = drop_null.set_index('Country Name')
print("\nEnergy usage Country Name Index: \n", index_energy)


# selecting "Country Name" column 
energy_sel = (new_energy_use["Country Name"])


# arranging length of column
x = np.arange(len(energy_sel))


# extracting particular years and storing it in a particular variable
energy_one = (new_energy_use["1990"])
energy_two = (new_energy_use["1995"])
energy_three = (new_energy_use["2000"])
energy_four = (new_energy_use["2005"])
energy_five = (new_energy_use["2010"])
energy_six = (new_energy_use["2015"])


# plotting figure and adjusting figure size in plot
plt.figure(figsize=(10,8))


# bar graph plot
plt.bar(x-0.3,energy_one, width=0.1, label="1990", edgecolor="black", color="orange")
plt.bar(x-0.2,energy_two, width=0.1, label="1995", edgecolor="black", color="darkgrey")
plt.bar(x-0.1,energy_three, width=0.1, label="2000", edgecolor="black", color="seagreen")
plt.bar(x+0.1,energy_four, width=0.1, label="2005", edgecolor="black", color="cyan")
plt.bar(x+0.2,energy_five, width=0.1, label="2010", edgecolor="black", color="yellow")
plt.bar(x+0.3,energy_six, width=0.1, label="2015", edgecolor="black", color="violet")

# manipulating ticks on x & y axis
plt.xticks(x, energy_sel, rotation = 45, fontsize=15) 
plt.yticks(fontsize=15)

plt.title("Energy usage", fontsize=15)
plt.xlabel("Countries", fontsize=15)
plt.ylabel("Energy usage", fontsize=15)
plt.legend(fontsize=15)
plt.savefig("Energy_usage.png")
plt.show()

# In[]:
    
def readFile(b):
    '''
        This function is used to read a csv file in its original form and in a transposed version.
        
        gdp : variable for storing csv file
        gdp_t : variable for storing transposed csv file
   '''
    gdp = pd.read_csv("GDP.csv");
    gdp = gdp.drop(['Country Code', 'Indicator Name', \
                                'Indicator Code', '1960'], axis=1)
    gdp_t = pd.DataFrame.transpose(gdp)
    return gdp, gdp_t

# calling function above to show dataframe
gdp, gdp_t = readFile("GDP.csv")
print("\n GDP: \n", gdp)
print("\nTransposed GDP: \n", gdp_t)

# selecting certain rows using slicing method
new_gdp = gdp.iloc[[35, 81, 109, 115, 111, 119, 170]]
print("\nExtracted GDP: \n", new_gdp)


# removing NaN values
drop_null_gdp = new_gdp.dropna()
print("\nExtracted GDP after removing NaN: \n", drop_null_gdp)


# calculating Normal Distribution of certain year through "scipy" module
print("\nNormal Distribution: \n", stats.skew(new_gdp["1965"]))


# calculating average of total GDP through "numpy" module
print("\nAverage GDP: \n", gdp.mean())


# removing first two lines
drop_null_gdp = drop_null_gdp.iloc[2:]
print("\nRemoving first two lines from GDP: \n", drop_null_gdp)


# indexing the "Country Name" column
gdp_index = drop_null_gdp.set_index('Country Name')
print("\n GDP Country Name Index: \n", gdp_index)


# select "Country Name" column for further use
gdp_sel = (new_gdp["Country Name"])


# arranging length of column
x = np.arange(len(gdp_sel))


# extracting certain years and storing it in a variable
gdp_one = (new_gdp["1990"])
gdp_two = (new_gdp["1995"])
gdp_three = (new_gdp["2000"])
gdp_four = (new_gdp["2005"])
gdp_five = (new_gdp["2010"])
gdp_six = (new_gdp["2015"])


# plotting figure and adjusting figure size
plt.figure(figsize=(10,8))


# bar graph plot
plt.bar(x-0.3,gdp_one, width=0.1, label="1990", edgecolor="black", color="pink")
plt.bar(x-0.2,gdp_two, width=0.1, label="1995", edgecolor="black", color="brown")
plt.bar(x-0.1,gdp_three, width=0.1, label="2000", edgecolor="black", color="blue")
plt.bar(x+0.1,gdp_four, width=0.1, label="2005", edgecolor="black", color="gray")
plt.bar(x+0.2,gdp_five, width=0.1, label="2010", edgecolor="black", color="green")
plt.bar(x+0.3,gdp_six, width=0.1, label="2015", edgecolor="black", color="orange")


# manipulating ticks on x & y axis
plt.xticks(x, gdp_sel, rotation = 45, fontsize=13) 
plt.yticks(fontsize=13)

plt.title("GDP Area", fontsize=15)
plt.xlabel("Countries", fontsize=15)
plt.ylabel("GDP (%)", fontsize=15)
plt.legend()
plt.savefig("GDP.png")
plt.show()

# In[]:

def readFile(d):
    '''
         elec : variable for storing csv file
        elec_t : variable for storing transposed csv file

    '''
    elec = pd.read_csv("Ele_Acc.csv");
    elec = pd.read_csv(d)
    elec = elec.drop(['Country Code', 'Indicator Name', 'Indicator Code', \
                      '1960', '1961', '1962', '1963', '1964', '1965', '1966', \
                      '1967', '1968', '1969', '1970', '1971', '1972', '1973', \
                      '1974', '1975', '1976', '1978', '1979', '1980', '1981', \
                      '1982', '1983', '1984', '1985', '1986', '1987', '1988', \
                      '1989', '2021'], axis=1)
    elec_t = pd.DataFrame.transpose(elec)
    return elec, elec_t

# calling function to display dataframe
elec, elec_t = readFile("Ele_Acc.csv")
print("\nAccess Electricity: \n", elec)
print("\nTransposed Access Electricity: \n", elec_t)


# populating header with header information
header4 = elec_t.iloc[0].values.tolist()
elec_t.columns = header4
print("\nAccess Electricity Header: \n", elec_t)


# removing first two lines
elec_t = elec_t.iloc[2:]
print("\nRemoving first two lines from Access Electricity: \n", elec_t)


# arranging length of column
print(len(elec_t))


# extracting particular countries and storing it in a variable

elec_t = elec_t[elec_t["Canada"].notna()]
elec_t = elec_t[elec_t["United Kingdom"].notna()]
elec_t = elec_t[elec_t["India"].notna()]
elec_t = elec_t[elec_t["Israel"].notna()]
elec_t = elec_t[elec_t["Ireland"].notna()]
elec_t = elec_t[elec_t["Japan"].notna()]
elec_t = elec_t[elec_t["North America"].notna()]


# indexing change as integer type
elec_t.index = elec_t.index.astype(int)


# plotting figure and adjusting figure size in plot
plt.figure(figsize=(10,8))


# line graph plot

plt.plot(elec_t.index, elec_t["Canada"], label="Canada", linestyle='dashed')
plt.plot(elec_t.index, elec_t["United Kingdom"], label="United Kingdom", linestyle='dashed')
plt.plot(elec_t.index, elec_t["India"], label="India", linestyle='dashed')
plt.plot(elec_t.index, elec_t["Israel"], label="Israel", linestyle='dashed')
plt.plot(elec_t.index, elec_t["Ireland"], label="Ireland", linestyle='dashed')
plt.plot(elec_t.index, elec_t["Japan"], label="Japan", linestyle='dashed')
plt.plot(elec_t.index, elec_t["North America"], label="North America", linestyle='dashed')


# manipulating ticks on x & y axis
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


plt.title("Electricity Access", fontsize=15)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Electricity Access (%)", fontsize=15)
plt.legend(bbox_to_anchor=(1.0,0.5), loc="center left", fontsize=15)
plt.savefig("elec.png")
plt.show()

# In[]:

def readFile(c):
    '''       
        pop : variable for storing csv file
        pop_t : variable for storing transposed csv file

    '''
    pop = pd.read_csv("Pop_grow.csv");
    pop = pop.drop(['Country Code', 'Indicator Name', 'Indicator Code'], \
                   axis=1)
    pop_t = pd.DataFrame.transpose(pop)
    return pop, pop_t

# calling function to show dataframe
pop, pop_t = readFile("Pop_grow.csv")
print("\nPopulation: \n", pop)
print("\nTranposed Population: \n", pop_t)


# populating header with header information
header3 = pop_t.iloc[0].values.tolist()
pop_t.columns = header3
print("\nPopulation Header: \n", pop_t)


# removing first two lines
pop_t = pop_t.iloc[2:]
print("\nRemoving first two lines from population: \n", pop_t)


# arranging length of column
print(len(pop_t))


# extracting particular countries and storing it in a variable
pop_t = pop_t[pop_t["Canada"].notna()]
pop_t = pop_t[pop_t["United Kingdom"].notna()]
pop_t = pop_t[pop_t["India"].notna()]
pop_t = pop_t[pop_t["Israel"].notna()]
pop_t = pop_t[pop_t["Ireland"].notna()]
pop_t = pop_t[pop_t["Japan"].notna()]
pop_t = pop_t[pop_t["North America"].notna()]


# indexing change as integer type
pop_t.index = pop_t.index.astype(int)


# plotting figure and adjusting figure size in plot
plt.figure(figsize=(10,8))


# line graph plot
plt.plot(pop_t.index, pop_t["Canada"], label="Canada", linestyle='dashed')
plt.plot(pop_t.index, pop_t["United Kingdom"], label="United Kingdom", linestyle='dashed')
plt.plot(pop_t.index, pop_t["India"], label="India", linestyle='dashed')
plt.plot(pop_t.index, pop_t["Israel"], label="Israel", linestyle='dashed')
plt.plot(pop_t.index, pop_t["Ireland"], label="Ireland", linestyle='dashed')
plt.plot(pop_t.index, pop_t["Japan"], label="Japan", linestyle='dashed')
plt.plot(pop_t.index, pop_t["North America"], label="North America", linestyle='dashed')


# manipulating ticks on x & y axis
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


plt.title("Population Growth", fontsize=15)
plt.xlabel("Year", fontsize=15)
plt.ylabel("Population (%)", fontsize=15)
plt.legend(bbox_to_anchor=(1,0.5), loc="center left", fontsize=15)
plt.savefig("pop.png")
plt.show()

# In[]

# read csv file adn storing it in a variable
table = pd.read_csv("GDP.csv")
table_df = pd.DataFrame(table)
print("\n GDP Dataframe - \n", table_df)

# accessing particular rows
new_table = table_df.iloc[[35, 81, 109, 115, 111, 119, 170]]
print("\nExtracted GDP Dataframe -  \n", new_table)