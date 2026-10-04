# Latest robot trajectory

To replot the saved data in this folder, run:

```bash
python latest_trajectory/plot_trajectories.py
```

`plot_trajectories.py` only reads `trajectories.json` beside it and writes the PNG and PDF beside it. It needs Matplotlib; it does not load policy models, ROS, or run logs. Both plot axes are in feet.

To replace the saved data with the newest logged run, execute `python refresh_latest_trajectory.py` from the repository root. This separate step uses the models and current arena to compute the ideal open-loop policy path from the logged initial pose. For a specific run, use `python refresh_latest_trajectory.py --run-id=<run-id>`.
