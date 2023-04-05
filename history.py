import matplotlib.pyplot as plt


def plot_history(history, metrics=None):
    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        metrics = [x for x in history.history.keys() if x[:4] != 'val_']
    if len(metrics) == 0:
        print('No metrics to display.')
        return

    x = history.epoch

    rows = 1
    cols = len(metrics)
    count = 0

    plt.Figure(figsize=(12 * cols, 8))
    for metric in sorted(metrics):
        count += 1
        plt.subplot(rows, cols, count)
        plt.plot(x, history.history[metrics], label='Train')
        val_metric = f'val_{metric}'
        if val_metric in history.history.keys():
            plt.plot(x, history.history[val_metric], label='Validation')
        plt.title(metric.capitalize())
        plt.legend()
    plt.show()


def add_history():


