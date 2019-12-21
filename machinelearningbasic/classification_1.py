#plt = matplotlib pyplot
#sns = seaborn

# Machine Learning first tutorial from DataCamp

import matplotlib.pyplot as plt
import seaborn as sns

# building a new figure
plt.figure()
sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['N', 'Y'])
plt.show()
# I think, plt memory should be destroyed before using another figure, IIRC?

plt.figure()
sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['N', 'Y'])
plt.show()