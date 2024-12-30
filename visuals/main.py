import matplotlib.pyplot as plt

def show_graph(df_col, title, swarming_indexes, queencell_indexes, feeding_indexes, honey_indexes, treatment_indexes, died_indexes):
    df_col.plot(figsize=(20,4))
    
    for i in swarming_indexes:
        plt.scatter(i, df_col.iloc[i], color='red', marker='o')
    for i in queencell_indexes:
        plt.scatter(i, df_col.iloc[i], color='green', marker='o')
    for i in feeding_indexes:
        plt.scatter(i, df_col.iloc[i], color='blue', marker='o')
    for i in honey_indexes:
        plt.scatter(i, df_col.iloc[i], color='yellow', marker='o')
    for i in treatment_indexes:
        plt.scatter(i, df_col.iloc[i], color='purple', marker='o')
    for i in died_indexes:
        plt.scatter(i, df_col.iloc[i], color='black', marker='o')

    if title is not None:
        plt.title(title)
    plt.show()