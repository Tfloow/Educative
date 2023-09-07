## Data Analysis with pandas
For this we are gonna use some data that were found [here](http://seanlahman.com/download-baseball-database/)

### Series
Series is the object in pandas to deal with 1D data. we create this `pandas.Series` thanks to its constructor. It takes as *first* arguments some data and can also use the `dtype` argument. (this time we use `"float64"` instead of `np.float` but we can still use NumPy data type and format) We can multiply a serie with a simple list array.

```python
series = pd.Series([1, 2, 3])
print(f"{series}\n")
# 0    1
# 1    2
# 2    3
# dtype: int64
```
We see that the index of our series are display on the left hand side. Usually it goes from $0$ up to $n-1$. But we can specify our index thanks to the `index` parameter when creating a new Serie.
```python
series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(f"{series}\n")
# a    1
# b    2
# c    3
# dtype: int64
```
Pandas also support series loaded from a *dictionary*. It will make the keys of the dictionary the index and the value of the dictionary the value of the serie.

### Dataframe
Pandas use Dataframe to handle 2D array. It's the most important object in pandas because it will help us handle tables, spreadsheets, ... The object name is `pandas.DataFrame`. It takes the same arguments as a Serie but with another parameter called `columns`.

In pandas when we have multiple format type of data we do what we call an **upcasting**. We change the type of the data to the highest one in the column.
```python
upcast = pd.DataFrame([[5, 6], [1.2, 3]])
print('{}\n'.format(upcast))
#      0  1
# 0  5.0  6
# 1  1.2  3
print(upcast.dtypes)
# 0    float64
# 1      int64
# dtype: object
```
We can append a row of data to an existing dataframe. **Watch out** : it doesn't change the old DF but return a new one. To have an unified index we can use `ignore_index=True`.

To drop lines in pandas, we use the `df.drop()`. We don't need an arguments but we can use some parameters:
- `label`: to define which row to delete.
- `axis`: to choose if we drop rows (0) or columns (1)
- `index`: to drow a row.
- `columns`: so no need to use label and axis, we immediatly specify the column to drop.

We can use `index` and `columns` together and even list.

### Combining
Here, we are gonna see how we can concatenate multiple DataFrame. We use `pd.concat()`. We pass an **array** of DF. We can choose the ``axis``. If we set to 1 then we had to the right. **Watch out**: if we had some data to the right and the two DF don't share the same index, then it will form this:
```python
df1 = pd.DataFrame({'c1':[1,2], 'c2':[3,4]},
                   index=['r1','r2'])
df3 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]})

concat = pd.concat([df1, df3], axis=1)
print('{}\n'.format(concat))
#      c1   c2   c1   c2
# r1  1.0  3.0  NaN  NaN
# r2  2.0  4.0  NaN  NaN
# 0   NaN  NaN  5.0  7.0
# 1   NaN  NaN  6.0  8.0
```
We can also *merge* DF together ! Thank to `pd.merge(df1, df2)`. It merges two DataFrames using their **common column labels**. If 2 data differs, pandas will use the data from `df2` and add a column describing how alike the two data are thaks to `rbi` column.

### Indexing
Now we will see how we retrieve data:
```python
df = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4],
                   'c3': [5, 6]}, index=['r1', 'r2'])
col1 = df['c1']
# Newline for separating print statements
print('{}\n'.format(col1))

col1_df = df[['c1']]
print('{}\n'.format(col1_df))

col23 = df[['c2', 'c3']]
print('{}\n'.format(col23))
```
<details>
<summary>Output</summary>
<br>

```
r1    1
r2    2
Name: c1, dtype: int64

    c1
r1   1
r2   2

    c2  c3
r1   3   5
r2   4   6
```
</details>


We can see that if we use a **single** name of a colum, we get a Series. If we use **double** bracket like `[['c1']]` we get a DF as an output.

To get rows, we can simply use number as we would do  with list in Python.

There is other way to index. DF also contains `loc` and `iloc`. If we use `iloc`, we can get rows based on int. We can give integers or list of integers:

```python
df = pd.DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6],
                   'c3': [7, 8, 9]}, index=['r1', 'r2', 'r3'])
                   
print('{}\n'.format(df))

print('{}\n'.format(df.iloc[1]))

print('{}\n'.format(df.iloc[[0, 2]]))

bool_list = [False, True, True]
print('{}\n'.format(df.iloc[bool_list]))
```
<details>
<summary>Output</summary>
<br>

```
    c1  c2  c3
r1   1   4   7
r2   2   5   8
r3   3   6   9

c1    2
c2    5
c3    8
Name: r2, dtype: int64
```
</details>

The `loc` works like iloc but we need to use **row label**. We can perform column indexing with row indexing, and set new values in a DF for specific rows and columns:
```python
df = pd.DataFrame({'c1': [1, 2, 3], 'c2': [4, 5, 6],
                   'c3': [7, 8, 9]}, index=['r1', 'r2', 'r3'])
                   
print('{}\n'.format(df))

print('{}\n'.format(df.loc['r2']))

bool_list = [False, True, True]
print('{}\n'.format(df.loc[bool_list]))

single_val = df.loc['r1', 'c2']
print('Single val: {}\n'.format(single_val))

print('{}\n'.format(df.loc[['r1', 'r3'], 'c2']))

df.loc[['r1', 'r3'], 'c2'] = 0
print('{}\n'.format(df))
```
<details>
<summary>Output</summary>
<br>

```
    c1  c2  c3
r1   1   4   7
r2   2   5   8
r3   3   6   9

c1    2
c2    5
c3    8
Name: r2, dtype: int64

    c1  c2  c3
r2   2   5   8
r3   3   6   9

Single val: 4

r1    4
r3    6
Name: c2, dtype: int64

    c1  c2  c3
r1   1   0   7
r2   2   5   8
r3   3   0   9
```
</details>


### File I/O
Now we will learn how to read dat from various type of format.

#### Read CSV
We use `read_csv()`. We sepcify the filepath. By default, the index is simply numbered from 0 to n-1. If we want to specify which columns should be used for index, we use `index_col`.

#### Read XLSX (Excel)
We use `read_excel()`. It's a bit different than CSV because a XLSX can have other spreadsheet. We need to precise which sheet we want to use thanks to `sheet_name`. We can pass a `None` or a list of number and we will have a dictionary of DF.

#### Read JSON
We use `read_json()`. We can change the orientation as the JSON is read by passing the `orient` parameter. We can use `"index"` to transpose it.

#### Write CSV
We write with `to_csv()`. It can save without any specified name. We can say we don't need the index by passing `index=False`.

#### Write XLSX
We write with `to_excel()`. To open an excel notebook we need more work like this:
```python
with pd.ExcelWriter('data.xlsx') as writer:
  mlb_df1.to_excel(writer, index=False, sheet_name='NYY')
  mlb_df2.to_excel(writer, index=False, sheet_name='BOS')
```

#### Write JSON
We write with `to_json()`. It takes the same parameter as in `read_json()`.

### Grouping
When we have a large set of data it's usually ideal to group them by common categories. We can do this thanks to `groupby()`. We can pass a specific column.

```python
# Predefined df of MLB stats
print('{}\n'.format(df))

groups = df.groupby('yearID')
for name, group in groups:
  print('Year: {}'.format(name))
  print('{}\n'.format(group))
  
print('{}\n'.format(groups.get_group(2016)))
print('{}\n'.format(groups.sum()))
print('{}\n'.format(groups.mean()))
```
<details>
<summary>Output</summary>
<br>

```
   yearID teamID     H    R
0    2017    CLE  1449  818
1    2015    CLE  1395  669
2    2016    BOS  1598  878
3    2015    DET  1515  689
4    2016    DET  1476  750
5    2016    CLE  1435  777
6    2015    BOS  1495  748
7    2017    BOS  1461  785
8    2017    DET  1435  735

Year: 2015
   yearID teamID     H    R
1    2015    CLE  1395  669
3    2015    DET  1515  689
6    2015    BOS  1495  748

Year: 2016
   yearID teamID     H    R
2    2016    BOS  1598  878
4    2016    DET  1476  750
5    2016    CLE  1435  777

Year: 2017
   yearID teamID     H    R
0    2017    CLE  1449  818
7    2017    BOS  1461  785
8    2017    DET  1435  735

   yearID teamID     H    R
2    2016    BOS  1598  878
4    2016    DET  1476  750
5    2016    CLE  1435  777

           H     R
yearID            
2015    4405  2106
2016    4509  2405
2017    4345  2338

                  H           R
yearID                         
2015    1468.333333  702.000000
2016    1503.000000  801.666667
2017    1448.333333  779.333333
```
</details>


Here, the grouping produced 3 DF for each years. We can get them thanks to the `groups` variable. We then use `sum` and `mean` function to perform analytics. We can also *filter* thanks to the `filter` function.
```python
no2015 = groups.filter(lambda x: x.name > 2015)
print(no2015)
```
<details>
<summary>Output</summary>
<br>

```
   yearID teamID     H    R
0    2017    CLE  1449  818
2    2016    BOS  1598  878
4    2016    DET  1476  750
5    2016    CLE  1435  777
7    2017    BOS  1461  785
8    2017    DET  1435  735
```y(['cruzne02', 'pedrodu01', 'troutmi01'], dtype=object)
```
</details>


We can also group by **multiple** columns if we pass in `groupby()` a list of columns.


### Features
We often refer to the columns of a DF as the features. Those features can be **quantitative** or **categorical**. *Quantitative* is something that can be measured. Categorical is something that is useful paired with a `groupby()`.

we can sum a DF, show the mean and in those two case we can choose the axis by passing `axis` as a parameter.

We can also apply some weights to some data. We use the `multiply` function. `multiply` takes a list of weight or a constant. 

### Filtering
We can also do the same as in NumPy for DF by doing comparison. For example, we can put a condition on a specific column. Some useful function:
- `str.startwith()`: we want the word to start with a specific string
- `str.endwith()`: we want the word to end with a specific string
- `str.contains()`: we want the word to contain a specific string.

We can also use `.isin([a, b, ...])` which does multiple comparison.

To find if a data is missing we can use the `isna()` or not with `notna()`.

It is also pretty easy to filter DF's row with some conditions. we do this:
```python
hr40_df = df[df['HR'] > 40]
print('{}\n'.format(hr40_df))
```

<details>
<summary>Output</summary>
<br>

```
   playerID  yearID teamID  HR
2  cruzne02    2016    SEA  43
```
</details>

### Sorting
To sort in pandas, we use `sort_values()`. It takes as first argument the column we want to sort and we can set it into ascending or descending order thanks to `ascending`. We can also sort multiple columns when usinig a list of columns name.


### Metrics
In pandas, rather than just computing everything by hand, we can have a description with useful data about the DF with `describe`. It returns an usefule DF with this init:
| Metric |                                 Description                                 |
| :----: | :-------------------------------------------------------------------------: |
| count  |                     The number of rows in the DataFrame                     |
|  mean  |                        The mean value for a feature                         |
|  std   |                    The standard deviation for a feature                     |
|  min   |                       The minimum value in a feature                        |
|  25%   |                      The 25th percentile of a feature                       |
|  50%   | The 50th percentile of a feature. Note that this is identical to the median |
|  75%   |                      The 75th percentile of a feature                       |
|  max   |                       The maximum value in a feature                        |

We can also set the percentiles ourself by passing a list like `percentiles=[.2, .8]`.

#### Categorical Features
We cannot really compute or sum *categorical* features so we need to use `value_counts()` that simply count the recurrence. We also can sort by ascending or descending thanks to `ascending` or even *normalize* the data thanks to `normalize`.

Something also useful is to get each *unique* data  that apperars in a column. We do this thanks to `unique()`.
```python
unique_players = df['playerID'].unique()
print('{}\n'.format(repr(unique_players)))
```

<details>
<summary>Output</summary>
<br>

```
array(['cruzne02', 'pedrodu01', 'troutmi01'], dtype=object)
```
</details>

### Plotting
To plot a DF we need to combine **matplotlib** and DF:
```python
print('{}\n'.format(df))

df.plot(kind='line',x='yearID',y='HR')
plt.show()
```
Output:
![](plotExample.png)

### To Numpy
DF is useful for storing data but not that fast and it's not so convenient with ML framework so we need to make our life easier and use NumPy.

When we deal with categorical data, we need to convert them to quantitative data to be useful with NumPy:
```python
# predefined non-indicator DataFrame
print('{}\n'.format(df))

# predefined indicator Dataframe
print('{}\n'.format(indicator_df))
```

<details>
<summary>Output</summary>
<br>

```
    color
r1    red
r2   blue
r3  green
r4    red
r5    red
r6   blue

    blue  red  green
r1     0    1      0
r2     1    0      0
r3     0    0      1
r4     0    1      0
r5     0    1      0
r6     1    0      0
```
</details>

To have this result we must transform our initial dataframe:
1. Transform it with `get_dummies(df)`
2. Slice the converted data. `get_dummies` will give us $\sum_i col_i \times possibilities_i$. So we need to split it back into $col_i$ DF.
```python
print('{}\n'.format(df))

converted = pd.get_dummies(df)
print('{}\n'.format(converted.columns))

print('{}\n'.format(converted[['teamID_BOS',
                               'teamID_PIT']]))
print('{}\n'.format(converted[['lgID_AL',
                               'lgID_NL']]))
```
Then to convert to NumPy, we can simply (*after doing the 2 first steps*) do `df.values`.
