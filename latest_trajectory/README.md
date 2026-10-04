# Latest robot trajectory

From the repository root, run:

```bash
python latest_trajectory/plot.py
```

The script selects the newest `logs/<run-id>/` containing both `motion.jsonl` and `policy.jsonl`, then saves `latest_trajectory.png`, `.pdf`, and a summary in this folder. It plots the full measured path against an ideal unicycle rollout of the same policies and controller from the logged initial pose. Both plot axes are in feet. The open-loop curve is a new policy rollout, so its sampled actions may differ from those taken during the physical run.

For a specific run, use `python latest_trajectory/plot.py --run-id=<run-id>`. The script requires that `configs/arena.json` still matches the logged arena and settings, plus the two checkpoints in `models/`.
