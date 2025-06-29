import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Data from the summary.txt file
summary_data = """
   k_value n_params  train_time baseline_accuracy noise_0.1 noise_0.2 noise_0.3 bootstrap_mean bootstrap_std
1        1       16    0.002495          0.493333  0.485333     0.488  0.485333       0.490333      0.032824
2        2      121    0.336728              0.78     0.772  0.744667  0.734667         0.7845      0.025125
3        3      576    1.674391          0.756667  0.764667     0.752  0.743333       0.754083      0.025826
4        4     1941    5.234012          0.766667  0.758667  0.751333     0.744         0.7655      0.023304
5        5     4944   13.765295          0.746667     0.734  0.747333  0.735333          0.749      0.028401
6        6     9949   28.647451          0.726667     0.742  0.727333     0.732         0.7255      0.026986
7        7    16384   51.255158          0.733333  0.743333      0.73  0.709333       0.737667      0.026783
8        8    22819   77.453654          0.733333     0.734  0.718667  0.708667       0.739583      0.029771
9        9    27824  103.430181          0.733333  0.725333  0.724667     0.708        0.73875      0.028127
10      10    30827  124.885461          0.733333  0.732667  0.716667     0.708       0.738167      0.023144
11      11    32192  142.252029              0.73  0.731333      0.72     0.708       0.730333      0.024495
12      12    32647  154.900397          0.733333  0.715333  0.708667  0.700667       0.732583      0.025816
13      13    32752  160.280006              0.73  0.731333  0.705333  0.696667       0.731833      0.028428
14      14    32767  161.839792              0.73  0.736667  0.718667  0.700667       0.723917      0.024264
15      15    32768  161.090248              0.73     0.722  0.720667  0.708667        0.72225      0.030614
"""

# Read the data into a pandas DataFrame
data = pd.read_csv(io.StringIO(summary_data), delim_whitespace=True)

# Extract the required columns
accuracy = data['baseline_accuracy']
n_params = data['n_params']
k_values = data['k_value']

# Calculate the complexity metric
complexity_metric = accuracy / np.log2(n_params)

# Calculate the adjusted accuracy and complexity metrics
adjusted_accuracy = accuracy - 0.5
log_n_params = np.log2(n_params)
metric1 = adjusted_accuracy / log_n_params

# Create the plot for Metric 1
plt.figure(figsize=(10, 6))
plt.plot(k_values, metric1, marker='o')
plt.xlabel('k_value')
plt.ylabel('(Accuracy - 0.5) / log2(n_params)')
plt.title('Adjusted Accuracy-Complexity Tradeoff')
plt.grid(True)

plt.savefig('c:/Users/Tomas/OneDrive - Universidade de Lisboa/3ºano_LEFT/PIC-I/research/paper improvement/complexity_accuracy_metric1.png')
plt.show()


# Data from the COVID summary.txt file
summary_data_covid = """
  k_value n_params  train_time baseline_accuracy noise_0.1 noise_0.2 noise_0.3 bootstrap_mean bootstrap_std
1       1       10    0.099087           0.68659  0.686651  0.685492   0.68383       0.686841       0.00291
2       2       46    0.373591          0.689525  0.690509   0.68909  0.686415       0.688818      0.002989
3       3      130    1.350533          0.693871  0.693634  0.691126   0.68723       0.694231      0.003298
4       4      256   71.147153          0.694557  0.692277  0.687772  0.682344       0.694032      0.002507
5       5      382  106.845533          0.695052  0.692994  0.687962  0.680529       0.695261      0.003604
6       6      466  129.158613          0.693947  0.693017  0.688107  0.678669       0.694042      0.003237
7       7      502  142.641851          0.694099   0.69336  0.687566  0.676481       0.694671      0.002665
8       8      511  146.330889          0.694099  0.693078  0.687268  0.679027       0.694339      0.003101
9       9      512  149.094442          0.694252  0.693375  0.689334  0.681307        0.69381       0.00312
"""

# Read the data into a pandas DataFrame
data_covid = pd.read_csv(io.StringIO(summary_data_covid), delim_whitespace=True)

# Extract the required columns
accuracy_covid = data_covid['baseline_accuracy']
n_params_covid = data_covid['n_params']
k_values_covid = data_covid['k_value']

# Calculate the adjusted accuracy and complexity metrics
adjusted_accuracy_covid = accuracy_covid - 0.5
log_n_params_covid = np.log2(n_params_covid)
metric1_covid = adjusted_accuracy_covid / log_n_params_covid

# Create the plot for Metric 1 for COVID data
plt.figure(figsize=(10, 6))
plt.plot(k_values_covid, metric1_covid, marker='o', color='r')
plt.xlabel('k_value')
plt.ylabel('(Accuracy - 0.5) / log2(n_params)')
plt.title('Adjusted Accuracy-Complexity Tradeoff (COVID Dataset)')
plt.grid(True)

plt.savefig('c:/Users/Tomas/OneDrive - Universidade de Lisboa/3ºano_LEFT/PIC-I/research/paper improvement/complexity_accuracy_metric1_covid.png')
plt.show()
