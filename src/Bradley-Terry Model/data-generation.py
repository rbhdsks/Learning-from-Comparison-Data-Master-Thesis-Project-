# for the generation of the fake data to implement the bradley terry model
import numpy as np
n_items =5
# here the n items means 5 different product with the help of which the model will be implemented 
true_w= np.array([2.0,1.0,0.5,-0.5,-1.5])
# the above one is the true strength weights 
n_comparisons=2000
rng=np.random.default_rng(seed=42)
data=[]
for _ in range (n_comparisons):
    i,j=rng.choice(n_items, size=2,replace= False)
    # now here the formula for bradley terry model
    p_i_beats_j= np.exp(true_w[i]) /np.exp(true_w[j]) + np.exp(true_w[i])
    if rng.random()< p_i_beats_j:
        winner,loser=i,j
    else :
        winner, loser = j,i
    data.append((winner,loser))

data = np.array(data)

# now let's print first 10 comparison
print(data[:10])

np.save("bt_comparisons.npy", data)
np.save("bt_true_w.npy", true_w)


# to convert to pandas to see the data 
import pandas as pd
 
# Comparisons -> CSV with column names
df_comparisons = pd.DataFrame(data, columns=["winner", "loser"])
df_comparisons.to_csv("bt_comparisons.csv", index=False)
 
# True weights -> CSV with item index and weight
df_true_w = pd.DataFrame({
    "item": np.arange(n_items),
    "true_w": true_w
})
df_true_w.to_csv("bt_true_w.csv", index=False)
 
print("Saved comparisons to 'bt_comparisons.csv'")
print("Saved true weights to 'bt_true_w.csv'")
 




