# 2345042
import pandas as pd
import numpy as np

# --- Phase 1: Data Loading and Initial Inspection ---

# Load dataset
df = pd.read_csv("C:\\Users\\Raksha Shetty\\OneDrive\\Documents\\online retail csv.csv")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Print data types
print("\nData types of each column:")
print(df.dtypes)

# Number of rows and columns
print("\nDataset shape:", df.shape)

# Initial Data Exploration
print("\nMissing values per column:")
print(df.isnull().sum())

# Descriptive statistics for numerical columns
print("\nDescriptive statistics:")
print(df.describe())

# Identify potential outliers in Quantity and UnitPrice
for col in ['Quantity', 'UnitPrice']:
    mean = df[col].mean()
    std = df[col].std()
    outliers = df[(df[col] > mean + 3 * std) | (df[col] < mean - 3 * std)][col]
    print(f"\nPotential outliers in {col}:", outliers.count())
    
# Observations
print("\nInitial Observations: Check for negative Quantity, missing CustomerID, and inconsistent Description.")

# --- Phase 2: Data Cleaning ---

# 1. Handling Missing Values
# Drop rows with missing UnitPrice (critical for analysis)
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')  # Ensure numeric
df.dropna(subset=['UnitPrice'], inplace=True)

# For CustomerID, assign a placeholder (-1) for missing values
df['CustomerID'] = df['CustomerID'].fillna(-1)  # Placeholder for unidentified customers
print("\nMissing values after handling:")
print(df.isnull().sum())

# Justification: UnitPrice is essential; missing CustomerID can be tagged for analysis.

# 2. Data Type Conversion
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')  # Convert to datetime
df['CustomerID'] = df['CustomerID'].astype(int)  # Convert to integer
df['Country'] = df['Country'].astype('category')  # Convert to categorical
print("\nUpdated data types:")
print(df.dtypes)

# 3. Handling Duplicates
# Remove duplicate rows based on InvoiceNo and CustomerID
df = df.drop_duplicates(subset=['InvoiceNo', 'CustomerID'], keep='first')
print("\nShape after removing duplicates:", df.shape)

# 4. Addressing Inconsistencies
# Standardize Country names (title case)
df['Country'] = df['Country'].str.title()

# Clean Description (remove extra spaces, uppercase)
df['Description'] = df['Description'].str.strip().str.upper()
print("\nUnique countries after standardization:", df['Country'].unique())

# 5. Outlier Handling
# Cap Quantity and UnitPrice at 99th percentile
for col in ['Quantity', 'UnitPrice']:
    cap_value = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=cap_value)
print("\nMax Quantity after capping:", df['Quantity'].max())
print("Max UnitPrice after capping:", df['UnitPrice'].max())

# Justification: Capping preserves data while mitigating extreme outliers.

# --- Phase 3: Data Transformation ---

# 1. Feature Engineering
# Calculate total purchase value (Quantity * UnitPrice)
df['TotalPurchase'] = df['Quantity'] * df['UnitPrice']

# Extract month from InvoiceDate
df['PurchaseMonth'] = df['InvoiceDate'].dt.month
print("\nSample of new features:")
print(df[['TotalPurchase', 'PurchaseMonth']].head())

# Rationale: TotalPurchase reflects customer spending; PurchaseMonth aids temporal analysis.

# 2. Data Aggregation and Summarization
# Average TotalPurchase per Country
country_summary = df.groupby('Country')['TotalPurchase'].agg(['mean', 'count'])
print("\nCountry summary:")
print(country_summary)

# Pivot table: TotalPurchase by Country and PurchaseMonth
pivot = df.pivot_table(values='TotalPurchase', index='Country', columns='PurchaseMonth', aggfunc='mean')
print("\nPivot table (TotalPurchase by Country and Month):")
print(pivot)

# 3. Data Standardization/Normalization
# Normalize TotalPurchase (min-max scaling)
df['TotalPurchase_normalized'] = (df['TotalPurchase'] - df['TotalPurchase'].min()) / \
                                 (df['TotalPurchase'].max() - df['TotalPurchase'].min())
print("\nNormalized TotalPurchase sample:")
print(df['TotalPurchase_normalized'].head())

# When needed: For models requiring comparable scales.

# 4. Data Binning
# Bin Quantity into categories
quantity_bins = [0, 10, 50, 100, float('inf')]
quantity_labels = ['Low', 'Medium', 'High', 'Very High']
df['QuantityCategory'] = pd.cut(df['Quantity'], bins=quantity_bins, labels=quantity_labels, right=False)
print("\nQuantity category distribution:")
print(df['QuantityCategory'].value_counts())

# --- Phase 4: Reporting and Documentation ---

# 1. Save Cleaned Dataset
df.to_csv(r"C:\Users\Shagun Jain\OneDrive\Documents\cleaned_online_retail.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_online_retail.csv'")

# 2. Data Wrangling Report (Printed Summary)
print("\n--- Data Wrangling Report ---")
print("Initial Findings: Missing CustomerID (25% rows), negative Quantity values, inconsistent Description.")
print("Cleaning Decisions: Dropped missing UnitPrice rows; tagged missing CustomerID as -1; capped outliers.")
print("Transformations: Added TotalPurchase and PurchaseMonth; binned Quantity for segmentation.")
print("Assumptions: Negative Quantity treated as returns, capped at 0 if needed; missing CustomerID valid for analysis.")
print("Summary: Final dataset has", df.shape[0], "rows, no missing UnitPrice, and enhanced features.")
First 5 rows of the dataset:
  InvoiceNo StockCode                          Description  Quantity  \
0    536365    85123A   WHITE HANGING HEART T-LIGHT HOLDER         6   
1    536365     71053                  WHITE METAL LANTERN         6   
2    536365    84406B       CREAM CUPID HEARTS COAT HANGER         8   
3    536365    84029G  KNITTED UNION FLAG HOT WATER BOTTLE         6   
4    536365    84029E       RED WOOLLY HOTTIE WHITE HEART.         6   

        InvoiceDate  UnitPrice  CustomerID         Country  
0  01-12-2010 08:26       2.55     17850.0  United Kingdom  
1  01-12-2010 08:26       3.39     17850.0  United Kingdom  
2  01-12-2010 08:26       2.75     17850.0  United Kingdom  
3  01-12-2010 08:26       3.39     17850.0  United Kingdom  
4  01-12-2010 08:26       3.39     17850.0  United Kingdom  

Data types of each column:
InvoiceNo       object
StockCode       object
Description     object
Quantity         int64
InvoiceDate     object
UnitPrice      float64
CustomerID     float64
Country         object
dtype: object

Dataset shape: (541909, 8)

Missing values per column:
InvoiceNo           0
StockCode           0
Description      1454
Quantity            0
InvoiceDate         0
UnitPrice           0
CustomerID     135080
Country             0
dtype: int64

Descriptive statistics:
            Quantity      UnitPrice     CustomerID
count  541909.000000  541909.000000  406829.000000
mean        9.552250       4.611114   15287.690570
std       218.081158      96.759853    1713.600303
min    -80995.000000  -11062.060000   12346.000000
25%         1.000000       1.250000   13953.000000
50%         3.000000       2.080000   15152.000000
75%        10.000000       4.130000   16791.000000
max     80995.000000   38970.000000   18287.000000

Potential outliers in Quantity: 346

Potential outliers in UnitPrice: 374

Initial Observations: Check for negative Quantity, missing CustomerID, and inconsistent Description.

Missing values after handling:
InvoiceNo         0
StockCode         0
Description    1454
Quantity          0
InvoiceDate       0
UnitPrice         0
CustomerID        0
Country           0
dtype: int64

Updated data types:
InvoiceNo              object
StockCode              object
Description            object
Quantity                int64
InvoiceDate    datetime64[ns]
UnitPrice             float64
CustomerID              int32
Country              category
dtype: object

Shape after removing duplicates: (25900, 8)

Unique countries after standardization: ['United Kingdom' 'France' 'Australia' 'Netherlands' 'Germany' 'Norway'
 'Eire' 'Switzerland' 'Spain' 'Poland' 'Portugal' 'Italy' 'Belgium'
 'Lithuania' 'Japan' 'Iceland' 'Channel Islands' 'Denmark' 'Cyprus'
 'Sweden' 'Austria' 'Israel' 'Finland' 'Bahrain' 'Greece' 'Hong Kong'
 'Singapore' 'Lebanon' 'United Arab Emirates' 'Saudi Arabia'
 'Czech Republic' 'Canada' 'Unspecified' 'Brazil' 'Usa'
 'European Community' 'Malta' 'Rsa']

Max Quantity after capping: 336
Max UnitPrice after capping: 125.0

Sample of new features:
    TotalPurchase  PurchaseMonth
0           15.30            1.0
7           11.10            1.0
9           54.08            1.0
21          25.50            1.0
25          17.85            1.0

Country summary:
                            mean  count
Country                                
Australia             105.695797     69
Austria                43.512632     19
Bahrain                12.475000      4
Belgium                18.702185    119
Brazil                175.200000      1
Canada                 92.693333      6
Channel Islands        38.137879     33
Cyprus                 27.055000     20
Czech Republic         -1.106000      5
Denmark                56.600476     21
Eire                   35.266722    360
European Community     16.350000      5
Finland                61.629583     48
France                 25.863905    461
Germany                23.213980    603
Greece                 31.016667      6
Hong Kong              30.898000     15
Iceland                23.308571      7
Israel                 10.523333      9
Italy                  16.388727     55
Japan                  72.427857     28
Lebanon                17.400000      1
Lithuania              42.000000      4
Malta                  37.100000     10
Netherlands           126.959109    101
Norway                 22.494500     40
Poland                 20.595417     24
Portugal               15.627183     71
Rsa                     6.800000      1
Saudi Arabia           -4.855000      2
Singapore              28.446000     10
Spain                  32.291619    105
Sweden                 49.591522     46
Switzerland            24.467027     74
United Arab Emirates   29.000000      3
United Kingdom         17.801835  23494
Unspecified            18.495385     13
Usa                    12.647143      7

Pivot table (TotalPurchase by Country and Month):
PurchaseMonth               1.0         2.0        3.0         4.0   \
Country                                                               
Australia             299.600000  148.380000  70.486667  154.413333   
Austria                19.500000    7.560000  39.600000         NaN   
Bahrain                      NaN         NaN        NaN         NaN   
Belgium                 8.130000    7.230000  33.333333   17.225000   
Canada                       NaN         NaN        NaN  356.160000   
Channel Islands        35.400000   40.800000        NaN         NaN   
Cyprus                       NaN  -33.900000  20.800000    0.000000   
Czech Republic        -21.750000         NaN        NaN         NaN   
Denmark                      NaN         NaN        NaN         NaN   
Eire                   47.870000   26.200000  18.857143    9.754286   
European Community           NaN         NaN        NaN         NaN   
Finland               137.800000   46.640000        NaN  121.346000   
France                 45.262667   19.929167  34.541176   24.732667   
Germany                 6.030556    6.623333  17.410714   13.753333   
Greece                       NaN         NaN        NaN    9.900000   
Hong Kong                    NaN         NaN        NaN  -10.950000   
Iceland                      NaN   13.200000        NaN         NaN   
Israel                 17.000000         NaN        NaN    6.950000   
Italy                 120.000000   85.920000  15.000000    3.937500   
Japan                        NaN   -7.650000        NaN -528.000000   
Lithuania                    NaN         NaN        NaN         NaN   
Malta                 -19.900000         NaN        NaN         NaN   
Netherlands           120.550000  211.076667  -2.400000         NaN   
Norway                 -8.566667   25.200000  19.800000         NaN   
Poland                -39.800000   16.500000  24.150000   56.100000   
Portugal                     NaN   18.600000   6.100000         NaN   
Saudi Arabia                 NaN         NaN -14.750000         NaN   
Singapore                    NaN         NaN        NaN    0.000000   
Spain                  10.500000         NaN  35.210000   23.587500   
Sweden                114.450000   25.500000  -8.500000   13.555000   
Switzerland            19.800000   -1.250000  14.250000   17.000000   
United Arab Emirates   19.800000         NaN        NaN         NaN   
United Kingdom         31.379287   35.441179  38.529759   31.438760   
Unspecified                  NaN    2.950000        NaN         NaN   
Usa                          NaN    5.040000        NaN         NaN   

PurchaseMonth               5.0         6.0         7.0        8.0   \
Country                                                               
Australia             338.780000   92.550000   17.025000  22.500000   
Austria                      NaN   39.600000         NaN  74.720000   
Bahrain              -205.740000         NaN         NaN        NaN   
Belgium                12.487500   17.750000    5.091429  14.387500   
Canada                       NaN         NaN         NaN        NaN   
Channel Islands        35.575000   25.350000         NaN  -4.250000   
Cyprus                       NaN         NaN   14.190000        NaN   
Czech Republic               NaN         NaN   10.080000        NaN   
Denmark                10.200000         NaN  132.800000        NaN   
Eire                   22.326364   38.161111   40.346000  14.217692   
European Community     10.200000   45.000000         NaN        NaN   
Finland                67.000000         NaN  -21.040000        NaN   
France                 28.179583    9.955455   18.738571  11.945000   
Germany                27.074348    9.695185   30.698929   6.620435   
Greece                       NaN  135.000000         NaN  24.960000   
Hong Kong                    NaN         NaN         NaN        NaN   
Iceland                      NaN         NaN   19.866667        NaN   
Israel                       NaN         NaN         NaN        NaN   
Italy                   3.675000   -3.066667   23.850000        NaN   
Japan                 160.050000   -2.925000  -18.980000        NaN   
Lithuania              35.000000         NaN         NaN  63.000000   
Malta                        NaN         NaN         NaN  14.850000   
Netherlands           149.916667   32.127500   62.400000  71.824000   
Norway                 29.700000   15.000000         NaN  10.080000   
Poland                 17.282500         NaN         NaN  15.000000   
Portugal                9.530000   15.187500    6.730000  26.865000   
Saudi Arabia                 NaN         NaN         NaN        NaN   
Singapore             202.500000         NaN         NaN        NaN   
Spain                 175.200000    8.750000   28.012000  18.750000   
Sweden                  0.000000   97.080000         NaN  32.216667   
Switzerland            14.750000  109.900000         NaN  67.960000   
United Arab Emirates         NaN         NaN         NaN        NaN   
United Kingdom         36.639474   26.044808   38.910975  23.942295   
Unspecified                  NaN         NaN         NaN   9.360000   
Usa                    27.040000         NaN         NaN -20.800000   

PurchaseMonth               9.0         10.0        11.0        12.0  
Country                                                               
Australia              17.700000   81.600000   13.500000   17.550000  
Austria                      NaN         NaN         NaN   15.000000  
Bahrain                30.000000         NaN         NaN         NaN  
Belgium                36.337500    8.190000   16.893333   13.812000  
Canada                       NaN         NaN   23.400000         NaN  
Channel Islands        18.150000   50.850000         NaN         NaN  
Cyprus                       NaN   60.480000         NaN   10.000000  
Czech Republic               NaN         NaN         NaN         NaN  
Denmark                46.683333   34.800000   33.300000         NaN  
Eire                    6.930000  104.288125   32.877857   38.444444  
European Community           NaN   -8.500000         NaN         NaN  
Finland                      NaN         NaN   29.500000   65.350000  
France                 14.961667   35.313125   42.540000   33.652941  
Germany                19.468421   34.405789   14.007826   35.508889  
Greece                       NaN         NaN  -22.480000         NaN  
Hong Kong                    NaN         NaN         NaN   16.500000  
Iceland                24.960000         NaN         NaN         NaN  
Israel                 14.850000   15.000000         NaN         NaN  
Italy                  42.960000   33.000000   -4.125000         NaN  
Japan                 458.100000         NaN         NaN   87.000000  
Lithuania                    NaN         NaN         NaN         NaN  
Malta                        NaN         NaN         NaN         NaN  
Netherlands            47.383333  150.586667  162.533333  128.820000  
Norway                 29.960000   51.000000    6.000000   16.453333  
Poland                       NaN         NaN         NaN         NaN  
Portugal               22.600000   16.500000   34.600000    1.983333  
Saudi Arabia                 NaN         NaN         NaN         NaN  
Singapore                    NaN         NaN         NaN         NaN  
Spain                  18.316667   20.175000  137.690000         NaN  
Sweden                 42.960000  -28.800000   11.675000  132.000000  
Switzerland            55.350000   43.700000   -3.390000   10.475000  
United Arab Emirates         NaN         NaN         NaN         NaN  
United Kingdom       -169.435631   26.657030   29.090242  -11.158619  
Unspecified                  NaN   30.000000   15.900000         NaN  
Usa                          NaN   19.800000         NaN  -10.000000  

Normalized TotalPurchase sample:
0     0.957466
7     0.957442
9     0.957686
21    0.957524
25    0.957480
Name: TotalPurchase_normalized, dtype: float64

Quantity category distribution:
QuantityCategory
Low          10679
Medium        8114
Very High     1121
High           814
Name: count, dtype: int64

Cleaned dataset saved as 'cleaned_online_retail.csv'

--- Data Wrangling Report ---
Initial Findings: Missing CustomerID (25% rows), negative Quantity values, inconsistent Description.
Cleaning Decisions: Dropped missing UnitPrice rows; tagged missing CustomerID as -1; capped outliers.
Transformations: Added TotalPurchase and PurchaseMonth; binned Quantity for segmentation.
Assumptions: Negative Quantity treated as returns, capped at 0 if needed; missing CustomerID valid for analysis.
Summary: Final dataset has 25900 rows, no missing UnitPrice, and enhanced features.
