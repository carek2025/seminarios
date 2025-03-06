import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lista=np.linspace(0,360,num=360)
df=pd.DataFrame({"Angulo Normal":lista})

df["Radianes"]=np.radians(df["Angulo Normal"])
df["Seno"]=np.sin(df["Radianes"])
df["Coseno"]=np.cos(df["Radianes"])

plt.figure(figsize=(10,6))

plt.plot(df["Angulo Normal"],df["Seno"],label="Seno",color="yellow")
plt.plot(df["Angulo Normal"],df["Coseno"],label="Coseno",color="blue",linestyle="-")
plt.plot(df["Angulo Normal"],df["Coseno"],label="Coseno (discontinua)",color="red",linestyle="--")
plt.grid(True)
plt.show()