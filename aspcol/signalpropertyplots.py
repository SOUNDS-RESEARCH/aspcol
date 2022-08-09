import numpy as np
import matplotlib.pyplot as plt


def linear_and_circ_correlation(signal):
    if signal.ndim == 1:
        signal = signal[None, :]
    num_channels = signal.shape[0]
    numSamples = signal.shape[-1]
    spectrum = np.fft.fft(signal, axis=-1)

    fig, axes = plt.subplots(num_channels, 2)
    fig.tight_layout(pad=2)
    if not isinstance(axes[0], (list, np.ndarray)):
        axes = [axes]

    for i, ax in enumerate(axes):
        ax[0].plot(np.correlate(signal[i, :], signal[i, :], mode="full"))
        circcorr = np.fft.ifft(spectrum[i, :] * spectrum[i, :].conj()).real
        ax[1].plot(
            np.arange(-numSamples / 2 + 1, numSamples / 2 + 1),
            np.fft.fftshift(circcorr),
            ".-",
        )

        ax[0].set_title("Linear Autocorrelation")
        ax[1].set_title("Circular Autocorrelation")

    for rows in axes:
        for ax in rows:
            ax.grid(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    plt.show()


def inter_channel_correlation(signal):
    num_channels = np.min((signal.shape[0], 6))
    fig, axes = plt.subplots(num_channels, num_channels)
    fig.tight_layout()
    plt.suptitle("Linear Correlation", fontsize=16)

    if not isinstance(axes, (list, np.ndarray)):
        axes = [[axes]]

    for i, rows in enumerate(axes):
        for j, ax in enumerate(rows):
            ax.plot(np.correlate(signal[i, :], signal[j, :], mode="full"))

            ax.set_title("Channel " + str(j) + " and " + str(i))
            ax.grid(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
    plt.subplots_adjust(top=0.85)
    plt.show()


def inter_channel_circ_correlation(signal):
    num_channels = np.min((signal.shape[0], 6))
    numSamples = signal.shape[1]
    fig, axes = plt.subplots(num_channels, num_channels)

    plt.suptitle("Circular Correlation", fontsize=16)
    fig.tight_layout()
    if not isinstance(axes, (list, np.ndarray)):
        axes = [[axes]]

    spectrum = np.fft.fft(signal, axis=-1)
    for i, rows in enumerate(axes):
        for j, ax in enumerate(rows):
            circcorr = np.fft.ifft(spectrum[i, :] * spectrum[j, :].conj()).real
            ax.plot(
                np.arange(-numSamples / 2 + 1, numSamples / 2 + 1),
                np.fft.fftshift(circcorr),
                ".-",
            )
            ax.set_title("Channel " + str(j) + " and " + str(i))
            ax.margins(0.1, 0.1)
            ax.grid(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
    plt.subplots_adjust(top=0.85)
    plt.show()




def analyze_filter(A, Afreq, pos, freqs, numMics):
    for a in range(numMics):
        for b in range(numMics):
            fig, axes = plt.subplots(2, 1)
            axes[0].plot(np.abs(np.fft.fft(A[a, b, :])))
            axes[0].plot(Afreq[:, a, b])
            # axes[0].plot(np.abs(np.fft.fft(irls)))
            axes[0].legend(("freqsampling", "original"))
            axes[0].set_title("amplitude")

            axes[1].plot(np.unwrap(np.angle(np.fft.fft(A[a, b, :]))))
            axes[1].plot(np.unwrap(np.angle(Afreq[:, a, b])))
            # axes[1].plot(np.unwrap(np.angle(np.fft.fft(irls))))
            axes[1].legend(("freqsampling", "original"))
            axes[1].set_title("phase")
            plt.show()