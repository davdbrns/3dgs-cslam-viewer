# 3DGS C-SLAM viewer

A [viser](https://viser.studio) viewer for multi-agent 3D Gaussian Splatting C-SLAM maps. It shows several maps at once with per-layer toggles, agent trajectories and loop closures. You can follow an agent's path, snap the camera to the nearest agent view, and level the view to gravity.

## Run

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run cslam-viewer [run folders | folders of runs | .ply files ...]
```

Or use the launcher, which picks runs from `runs/`, finds a free port and opens the browser (Linux: uses `ss`, `curl`, `xdg-open`, and `fzf` if installed):

```bash
./view.sh                    # pick runs with fzf (Tab marks several)
./view.sh all                # every run in runs/
./view.sh euroc replica      # runs whose name contains these words
./view.sh office --max-splats 1000000   # options after the runs go to cslam-viewer
```

Open http://127.0.0.1:8080. To load more data, drop files or whole run folders anywhere on the window, or paste a path into **Load > Path**.

Options: `--port 8080`, `--upload-port` (default port + 1, receives dropped files), `--max-splats 3000000`, `--min-opacity 0.05`, `--host 127.0.0.1`.

## Input

Put run folders in `runs/` (ignored by git) or pass them on the command line. A run folder looks like this:

```
<run>/
  manifest.json      # lists maps; optional
  trajectories.npz   # agent_ids, agent{i}_kf_ids, agent{i}_optimized_kf_c2w (K,4,4), same world frame as the maps
                     # optional ground truth: agent{i}_gt_kf_c2w (K,4,4), agent{i}_gt_c2w (all frames, for drawing)
  loops.npz          # src_pos, tgt_pos, accepted, intra, ... ; optional
  maps/*.ply         # standard 3DGS PLY (binary little endian)
```

A loose `.ply` becomes its own layer. If there is no manifest, `*_yup.ply` copies are skipped, because they are in a different frame than the poses.

## Controls

| Action | Input |
| --- | --- |
| Orbit / pan / zoom | left drag / right drag / scroll |
| Fly | WASD, Q/E, arrow keys |
| Set orbit center | Ctrl+click on the map |
| Frame everything | View > Frame all |
| Gravity down | View > Gravity down (auto: mean camera up of all agents, snapped to the nearest world axis; override with *Up axis*) |
| Follow an agent | Follow agent > pick agent, move the *Keyframe* slider or press *Play* |
| Best-quality view | Follow agent > Snap to nearest agent view (closest keyframe pose to the current camera) |

With *Camera follows agent* off, a white frustum marks the agent's pose instead of moving your camera. *Level horizon* replaces the agent's roll with gravity up.

Each run in **Layers** has toggles for its maps, trajectories, keyframe frustums, accepted loops (magenta inter-agent, yellow intra-agent) and rejected loops (grey).

Runs with ground truth also get:

- **Ground truth**: the GT paths, in a lighter shade of each agent's color.
- **GT error lines**: dark lines from each estimated keyframe to its GT pose.
- **Align GT to estimate**: one rigid (Umeyama, no scale) fit over all agents together. Off means GT is drawn in its stored frame, which for MAGiC-SLAM runs is already the map frame, since each agent starts at its first GT pose.
- **ATE RMSE** per agent, raw or aligned to match the toggle. The joint fit minimizes the error over all agents, so a single agent's aligned ATE can be higher than its raw one.

## Notes

- Large maps are pruned on load. Splats below *Min opacity* are dropped, then at most *Max splats / map* are kept, ranked by opacity × projected area. The layer hint shows how many are shown. Change the settings and press *Reload maps*.
- Only the base color (SH degree 0) is rendered; viser's splat renderer has no view-dependent color.
- Browsers do not reveal local file paths, so dropped files are uploaded to a temp folder that is removed on exit. For very large local runs, the command line or the *Path* field avoids that copy.
- Drag and drop uses viser's internal `RunJavascriptMessage`, which is why `viser` is pinned.
- If the viewer runs on a remote machine, forward both the viewer port and the upload port.

## License

MIT, see [LICENSE](LICENSE).
