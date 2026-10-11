import matplotlib.pyplot as plt

bandwidths = ["0.5", "1.0", "2.0", "4.0", "6.0", "8.0", "10.0"]
accuracies = [0.9958300102540731, 0.9941209980631195, 0.9947818161102882, 0.994394440013672, 0.9926626409935058, 0.9890167483194713, 0.9860088868633928]

bars = plt.bar(bandwidths, accuracies)

plt.ylabel("Accuracy")
plt.xlabel("Bandwidth")
plt.ylim(0.95, 1.02)
plt.title("Accuracies for bandwidth constants")

for bar, val in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        val + 0.01,
        f"{val*100:.1f}%",
        ha="center",
        va="bottom"
    )

plt.savefig("other_stuff/bandwidth_accuracy_1D.png")