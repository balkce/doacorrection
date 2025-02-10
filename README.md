# doacorrection
DOA error correction through feedback of the speech enhancement quality.

## Basis of operation

![Diagram of the whole system](/images/proposedsystem.png?raw=true)

Three [ROS2](https://docs.ros.org/) nodes are provided:

- `demucs`: a variation of the original Demucs Denoiser model, that uses a [location-based strategy](https://github.com/balkce/demucstargetsel) for target selection. It subscribes to the `jackaudio` topic that is published by the [`beamform2`](https://github.com/balkce/beamform2) ROS2 node, and publishes the `jackaudio_filtered` topic that is the result of enhancing the speech from the beamforming output.

- `online_sqa`: a [SQUIM](https://pytorch.org/audio/main/tutorials/squim_tutorial.html)-based online speech quality estimator. It subscribes to the `jackaudio_filtered` topic, and publishes the `SDR` topic that is the speech quality estimation from the enhanced speech.

- `doaoptimizer`: it aims to optimize the speech quality by correcting the direction of arrival (DOA) that is fed to a beamformer, based on the [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer. It subscribes to the `SDR` topic, and publishes a the `theta` topic that is the corrected DOA.

The `theta` topic is then subscribed to by the [`beamform2`](https://github.com/balkce/beamform2) node, closing the feedback loop.

## Before running

1. Install and configure [`jackaudio`](https://jackaudio.org/).

2. Clone and compile the [`beamform2`](https://github.com/balkce/beamform2) ROS2 node.

3. Configure the `beamform_config.yaml` of `beamform2` so that it matches your microphone setup.

4. Configure the `rosjack_config.yaml` of `beamform2` so that:

   - Its output is fed through a ROS2 topic: `output_type` should be either `0` or `2`.
   - Its sampling rate matches the one that `demucs` was trained with: `ros_output_sample_rate` should be `16000`.

5. Install the python requirements of all the nodes:

   `pip install -r requirements.txt`

6. [Create a ROS2 package](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-Your-First-ROS2-Package.html), place the `demucs`, `doaoptimizer` and `online_sqa` directories inside the package's `src` directory, and run `colcon build`.

## To run:

1. Start the `jackaudio` server.

2. Start the `phase` beamformer from `beamform2`:

   `ros2 launch beamform2 phase.launch`

3. Run `demucs`:

   `ros2 run demucs demucs`

4. Start `jack_write` from `beamform2` to listen to the result:

   `ros2 launch beamform2 rosjack_write.launch`

5. Run `online_sqa`:

   `ros2 run online_sqa online_sqa`

6. Run `doaoptimizer`:

   `ros2 run doaoptimizer doaoptimizer`

The `jackaudio_filtered` topic provides the DOA-corrected enhanced speech.

## Hyperparameters:

All the following hyperparamaters can be set using the `--ros-args -p` argument, such as:

`ros2 run module submodule  --ros-args -p hyperparameter1:=value1 -p hyperparameter2:=value2`

Here is the list of modules and their hyperparameters:

- `demucs`:
  - `input_length`: length (in seconds) of input window (default: 0.512). The higher, the better quality, but the greater response time.

- `online_sqa`:
  - `hop_secs`: time hop (in seconds) between SDR estimates (default: 1.5).
  - `win_len_secs`: length (in seconds) of input window (default: 3.0).
  - `smooth_weight`: smoothing weight to apply to SDR estimate output (default: 0.9).

- `doaoptimizer`:
  - `init_doa`: initial DOA estimate (in degrees) of the source of interest (default: 0.0).
  - `eta`: adaptation rate of the Adam variation optimizer (default: 0.3).
  - `wait_for_sdr`: time (in seconds) to wait for SDR estimate after a new DOA correction is published (default: 1.5). It is highly recommended to use the same value as the `online_sqa`'s `hop_secs` hyperparameter.
  - `opt_correction`: use new optimization mechanism (default: True). This is explained in the following section.

## New optimization mechanism:

The `doaoptimizer` module can run using the original optimization mechanism that is based on a variation of the Adam optimizer. It can also run using a new optimizer correction mechanism. This new mechanism is run by default, but the original mechanism can be used by running:

   `ros2 run doaoptimizer doaoptimizer  --ros-args -p opt_correction:=False`

## Citation:

If you end up using this software, please credit it as:

```BibTex
@article{rascon2024direction,
  title={Direction of Arrival Correction through Speech Quality Feedback},
  author={Rascon, Caleb},
  journal={Digital Signal Processing},
  pages={104960},
  year={2024},
  publisher={Elsevier}
}
```
You can also have a look at its arxiv version:

```BibTex
@article{rascon2024direction,
  title={Direction of Arrival Correction through Speech Quality Feedback},
  author={Rascon, Caleb},
  journal={arXiv preprint arXiv:2408.07234},
  year={2024}
}
```

