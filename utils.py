import matplotlib.pyplot as plt

def plot_dict(dictionary):
    # Extract keys and values
    x = list(dictionary.keys())
    y = list(dictionary.values())

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o')

    # Adding labels and title
    plt.xlabel('thresholds')
    plt.ylabel('risk')
    plt.title('risk plot')

    # Display the plot
    plt.grid(True)
    plt.show()

def find_min_value_key(dictionary):
        if not dictionary:
            return None, None
        max_key = min(dictionary, key=dictionary.get)
        return max_key, dictionary[max_key]