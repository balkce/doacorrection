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

If you have an initial DOA estimate of the source of interest (SOI), you can provide it to `doaoptimizer` through the `init_doa` parameter. For example, if the SOI is located at 20 degrees, run `doaoptimizer` with:

`ros2 run doaoptimizer doaoptimizer  --ros-args -p init_doa:=20.0`

Also, the default learning rate is set at 0.15, since it provided good results in our tests. However, if you want to change it at run-time, you can do so through the `eta` parameter. For example, if the SOI is located at 20 degrees, and you want to have a learning rate of 0.1, run `doaoptimizer` with:

`ros2 run doaoptimizer doaoptimizer  --ros-args -p init_doa:=20.0 -p eta:=0.1`

The `jackaudio_filtered` topic provides the DOA-corrected enhanced speech.

## Citation:

If you end up using this software, please credit it as:

```BibTex
@article{rascon2024direction,
  title={Direction of Arrival Correction through Speech Quality Feedback},
  author={Rascon, Caleb},
  journal={arXiv preprint arXiv:2408.07234},
  year={2024}
}
```

