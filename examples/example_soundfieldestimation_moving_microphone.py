"""Moving microphone example."""

import json
from pathlib import Path

import aspcol.plot as aspplot
import aspcol.soundfieldestimation as sfe
import aspcore.fouriertransform as ft
import aspcore.pseq as pseq
import matplotlib.pyplot as plt
import numpy as np

import aspsim.diagnostics.diagnostics as dg
import aspsim.room.region as reg
import aspsim.room.trajectory as traj
import aspsim.signal.sources as sources
import aspsim.signal.sources as src
from aspsim.processor import AudioProcessor
from aspsim.simulator import SimulatorSetup

RT60 = 0.2
RIRLEN = 1000
SAMPLERATE = 2000


def main():
    """Run the moving microphone example."""
    # Choose where figures should be saved and create a SimulatorSetup object
    fig_path = Path(__file__).parent.joinpath("figs")
    fig_path.mkdir(exist_ok=True, parents=True)
    setup = SimulatorSetup(fig_path)

    # Adjust config values
    setup.sim_info.tot_samples = SAMPLERATE
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.samplerate = SAMPLERATE
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.8, 0.2, 0.1]
    setup.sim_info.rt60 = RT60
    setup.sim_info.max_room_ir_length = RIRLEN
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = RIRLEN
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "pdf"
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    setup.sim_info.highpass_cutoff = 20

    # Setup sources and microphones
    source_sig = pseq.create_pseq(RIRLEN)
    sound_src = src.Sequence(source_sig, amp_factor=1, end_mode="repeat")
    setup.add_free_source(
        "ls",
        np.array([[2.5, 0, 0]]),
        sound_src,
    )
    setup.add_mics(
        "mic",
        traj.LinearTrajectory(
            [[1.5, 0, 0], [-1.5, 0, 0]], 1, setup.sim_info.samplerate
        ),
    )
    sim = setup.create_simulator()

    # Choose which signals should be saved to files
    sim.diag.add_diagnostic(
        "ls_sig", dg.RecordSignal("ls", sim.sim_info, export_func="npz")
    )
    sim.diag.add_diagnostic(
        "mic_sig", dg.RecordSignal("mic", sim.sim_info, export_func="npz")
    )

    sim.run_simulation()

    # Load the signals from files
    ls_sig = np.load(sim.folder_path / f"ls_sig_{sim.sim_info.tot_samples}.npz")[
        "ls_sig"
    ]
    mic_sig = np.load(sim.folder_path / f"mic_sig_{sim.sim_info.tot_samples}.npz")[
        "mic_sig"
    ]

    # Below is an example of sound field estimation for moving microphones. No more information about
    # the simulator is shown.

    # Decide which positions and RIRs should be estimated
    # Here we will estimate RIRs along the trajectory of the microphone.
    eval_sampling_rate = 10
    eval_idxs = np.arange(
        0, sim.sim_info.tot_samples, sim.sim_info.samplerate // eval_sampling_rate
    )
    pos_eval = np.squeeze(sim.arrays["mic"].pos_all[eval_idxs, ...], axis=1)
    rir_eval_true = np.squeeze(
        sim.arrays.rir_all["ls"]["mic"][eval_idxs, ...], axis=(1, 2)
    )

    # Extract a single period from the loudspeaker signal.
    # Not necessary in principle, but it is currently how krr_moving_mic is implemented.

    period_len = RIRLEN
    ls_sig = np.squeeze(ls_sig, axis=0)  # only one source
    ls_sig_single_period = ls_sig[:period_len]
    assert np.all(
        [
            np.allclose(
                ls_sig[i * period_len : (i + 1) * period_len], ls_sig_single_period
            )
            for i in range(ls_sig.shape[-1] // period_len)
        ]
    )

    reg_param = 1e-5
    rir_estimates_freq = sfe.krr_moving_mic(
        mic_sig,
        np.squeeze(sim.arrays["mic"].pos_all, axis=1),
        pos_eval,
        ls_sig_single_period,
        sim.sim_info.samplerate,
        sim.sim_info.c,
        reg_param,
    )

    # The convenience function to plot comparisons accepts estimates in the frequency domain.
    rir_eval_true_freq = ft.rfft(rir_eval_true)
    freqs = ft.get_real_freqs(RIRLEN, sim.sim_info.samplerate)
    aspplot.soundfield_estimation_comparison(
        pos_eval, rir_estimates_freq, rir_eval_true_freq, freqs, sim.folder_path
    )

    # Inspect the time-varying room impulse responses
    rirs = sim.arrays.rir_all["ls"]["mic"]
    rirs = np.squeeze(rirs, axis=(1, 2))  # Because we have only one source and one mic

    time = np.arange(ls_sig.shape[-1]) / sim.sim_info.samplerate

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax[0].plot(time, np.squeeze(ls_sig), label="Source", linewidth=1.2)
    ax[1].plot(time, np.squeeze(mic_sig), label="Microphone", linewidth=1.2)
    ax[0].set_title("Source and microphone signals")
    ax[1].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    clr = ax.imshow(
        np.log10(np.abs(rirs.T) + 1e-6),
        aspect="auto",
        origin="lower",
        extent=(0, rirs.shape[0], 0, rirs.shape[1] / sim.sim_info.samplerate),
    )
    ax.set_title("Time-varying RIR magnitude (log scale)")
    ax.set_xlabel("Time index")
    ax.set_ylabel("RIR time [s]")
    plt.colorbar(clr, ax=ax, label="Log magnitude")
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    early_len = RIRLEN // 5
    clr = ax.imshow(
        rirs[:, :early_len].T,
        aspect="auto",
        origin="lower",
        extent=(0, rirs.shape[0], 0, early_len / sim.sim_info.samplerate),
    )
    ax.set_title("Early RIR samples (zoomed)")
    ax.set_xlabel("Time index")
    ax.set_ylabel("RIR time [s]")
    plt.colorbar(clr, ax=ax, label="Amplitude")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
