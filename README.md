# Introduction

One of SLEAP's biggest strengths is that, given extensive training data, it is able to learn to accurately predict animal poses. While SLEAP excels at generating accurate pose predictions, it struggles in tracking identities of individuals. While it allows users to track identity with different algorithms (in particular, "simple" and "flow" tracking), these methods struggle in situations where animal body parts might be occluded from one frame to the next. These errors are particularly prevalent in social situations or situations where the animals become occluded, and they require a great amount of human correction to fix them.

idTracker.ai is a video-tracking system that attempts to overcome the difficulties associated with identity tracking. It uses representation learning to learn how to distinguish two different individuals from each other: a network is trained so that images of the same animal land close together and images of different animals land far apart, and the identities are then found by clustering. However, the software *does not track pose*, and also requires users to manually threshold where animals on the screen might be. The framework relies on this thresholding in order to find "Blobs" on the screen, which it then crops as examples to use in contrastive learning. Crucially, SLEAP gives us enough information to be able to make these crops ourselves: its keypoints tell us where each individual is, which way it is facing, and roughly what its outline looks like, and its tracks already tell us which instances in consecutive frames belong to the same animal.

Thus, this project takes SLEAP predictions as its input and assigns a consistent identity to every instance, so that SLEAP's pose estimation comes with high quality identity tracking. It is important to note that this is not an integration of SLEAP and idTracker.ai: none of idTracker.ai's code is run. Rather, it is an adaptation of the techniques used in idTracker.ai (contrastive learning on pairs of fragments, a ResNet-18 embedding, and k-means clustering seeded from a global fragment), rebuilt around SLEAP's own data. The remainder of this document will describe how to run the program, the architecture used by the model, and how each of the individual modules works.

# Software

## Configure

Everything the program needs to know about your data lives in one YAML file. To create a starter file that lists the nodes of your skeleton, run:

```bash
sleap-id init-config --slp_file mice.predictions.slp --config roles.yaml
```

Open `roles.yaml` and fill in every setting marked `REQUIRED`: the number of animals, the keypoint roles that orient each crop, and the training target (see [Required User Inputs](#required-user-inputs)). A blank template generated from a 15-node mouse skeleton is in `examples/sample_config.yaml`.

## Run

A single command runs every stage, from the video to a new `.slp` file with identities:

```bash
sleap-id run --config roles.yaml --slp_file mice.predictions.slp -o session/
```

Useful options:

```
--video PATH         the video, when the path stored in the .slp is wrong on this machine
--frames START:END   only process part of the video (END exclusive, e.g. 0:3000)
--cache_gb N         disk budget for the training crops (default 2)
--workers N          video decoding processes (default: chosen from your CPUs and RAM)
--overwrite          replace a session that was made from different inputs or settings
```

The same run can be started from Python. Call it under `if __name__ == "__main__":`, because the video is decoded by separate processes that re-import the script that started them:

```python
import sleap_id

if __name__ == "__main__":
    sleap_id.run("mice.predictions.slp", "roles.yaml", output_path="session/")
```

**Running the same command again continues where the last run stopped.** Stages that are already done are skipped, and training resumes from its last checkpoint. On a cluster with short time limits, the same command can simply be submitted again until it reports `done`.

Everything is written into the session folder:

```
session/
  selection.h5                              one row per SLEAP instance: keypoints, scores, crop status, fragment
  crops.h5                                  the training crops (uncompressed, for fast random reads)
  checkpoints/latest.pt                     everything needed to continue training
  checkpoints/best.pt                       the best weights, used for prediction
  training_log.csv                          the loss and learning rate of every training batch
  predictions.h5                            embedding, cluster probabilities and vote of every instance
  identities.h5                             the identity of every instance and fragment, and why when it has none
  <name>.sleap_id_predictions.slp           the original .slp with identities as tracks
  run.log, prepare.log, run_summary.json
```

The last file is the readout: open it in the SLEAP GUI. SLEAP's own file is left unchanged.

# Architecture

## Architecture Overview

In order to run the program, we assume that the user already has a file that contains SLEAP predictions **with tracks** (from any SLEAP tracker). We go through the following steps:

1. Check the configuration against the skeleton, and the `.slp` against the video
2. Create crops that could be used as training examples using SLEAP keypoint information: rectangular images of uniform size that contain individuals oriented in the exact same way
3. Split SLEAP's tracks into fragments, where we can be sure that every instance belongs to the same animal
4. Choose a training sample of crops that fits in the disk budget, and stream the video to write it
5. Contrastive learning model training
6. Find the global fragments, and choose the one that seeds the clusters
7. Prediction: embed every instance and cluster the embeddings with k-means
8. Identity assignment, fragment by fragment
9. Write the identities into a copy of the original SLEAP file as `filename.sleap_id_predictions.slp` that we can open in the SLEAP GUI

### Required User Inputs

All of the inputs are saved in the YAML file (see `config.py` for every setting and its default).

1. Number of animals (`n_animals`)
2. Key nodes (`centroid_node`, `top_node`, `bottom_node`, and optionally `top_node_fallback`)
3. The outline of the body: `surrounding_keypoints` (nodes on the outline) and `bounded_keypoints` (nodes inside it)
4. The training target (`training.target_loss`), and optionally how long to wait before stopping (`training.patience_batches`, default 500) and before halving the learning rate (`training.lr_patience_batches`, default 250)
5. Optionally, a checkpoint from an earlier run to start training from (`training.resume_from`)

The remaining settings (crop size limit, color mode, which SLEAP instances to use, and so on) have defaults that match what idTracker.ai and the previous version of this project used.

### 1. Checks

Handled by `prepare.py`, `config.py`, `slp.py`, `video.py`

Before any heavy work starts, we check everything that could make the run fail hours later: that every node named in the configuration exists in the skeleton, that the video has as many frames as the `.slp` expects, that no frame holds more instances than `n_animals`, that the instances have SLEAP tracks, and that our frame numbering matches sleap-io's on a handful of decoded frames. `sleap-id prepare --dry_run` runs these checks alone and prints the plan without writing anything.

### 2. Image Normalization

Handled by `geometry.py`

In order to extract information from the video to generate examples that could be used by the training model, we go through the following algorithm:

1. For each instance, rotate the frame about `centroid_node` so that `top_node` sits directly above it (falling back to `top_node_fallback` when the top node is missing).
2. Measure the size of every instance in this rotated frame, and use the median over all instances in the video to set one crop size for the whole video.
3. Mask out everything outside the convex hull of `surrounding_keypoints`. When one of those nodes is missing, the outline is rebuilt with a Graham scan over the surrounding and bounded keypoints that are present.
4. Cut every instance with a single affine warp, so that rotation, scaling and cropping happen in one interpolation.

 *Please note: instances whose orientation or outline cannot be determined (no centroid, no top node, or too few outline nodes) get no crop. They are never dropped: every instance keeps a row with the reason in `crop_status`.*

In alignment with idTracker.ai, if the crop would exceed 6400 pixels (`max_crop_area`), all of the images are uniformly downscaled until they fit in a box of at most that size. For the 5-mouse video we tested on, the crops are 57 × 111 pixels.

### 3. Fragmentation

Handled by `selection.py`

When we try to find consecutive sequences of frames that we could attribute to the same individual animal, we create fragments. Rather than detecting crossings ourselves, we take advantage of the fact that SLEAP already separates individuals from each other: a fragment is a run of consecutive frames in which one SLEAP track holds a croppable instance. A fragment ends wherever its track skips a frame, because inside a gap we do not know where the animal went. Frames where a track holds two instances, and instances with no track, are left out of every fragment.

### 4. Training Sample and HDF5 Creation

Handled by `prepare.py`, `plan.py`, `stream.py`, `files.py`

Decoding a long video by jumping to random frames is very slow, so the video is always read **in order**. The frames are split into contiguous ranges, one per worker process. Each worker seeks once, decodes forward, and cuts the crops of its frames; a single writer process fills the HDF5 file, because HDF5 does not allow several processes to write to one file. The number of workers is chosen from the CPUs, RAM and free disk of the machine it runs on.

Only a **training sample** of crops is stored, so the disk never limits the length of the video: evenly spaced crops from every fragment of at least `min_fragment_length` frames, capped so that the whole sample fits in `--cache_gb` and in a quarter of the free disk. Two files are written:

- `selection.h5`: one row per SLEAP instance (frame, instance index, track, scores, keypoints, crop status, fragment), compressed
- `crops.h5`: the selected crops, uncompressed and contiguous, so that random training batches can be read quickly

Both are written under a temporary name and renamed only once complete, and both are stamped with a `cache_id` (a hash of the inputs, settings and selection), so a run can never mix crops from different inputs.

### 5. Training

Handled by `network.py`, `contrastive.py`, `train.py`

The network is a standard ResNet-18, built the way idTracker.ai builds its contrastive network: the first convolution accepts the crops' channel count, and the last layer maps each crop to 8 coordinates. Because ResNet-18 averages over space before its last layer, it accepts **rectangular inputs** of the size computed in the image normalization stage, with no padding to a square.

Training uses two facts that the tracking already gives us, without knowing any identity:

- **positive pair**: two crops of the same fragment are the same animal
- **negative pair**: crops of two fragments that appear in the same frame are different animals

Each batch holds 400 pairs of each kind. With $d$ the distance between the two crops' coordinates, the loss pulls positive pairs to within distance 1 and pushes negative pairs out to distance 10:

$$\text{loss}_{\text{negative}} = \max(0,\ 10 - d)^2 \qquad \text{loss}_{\text{positive}} = \max(0,\ d - 1)^2$$

As in idTracker.ai, pairs that still have loss are drawn more often in later batches.

Every batch's loss is written to `training_log.csv`. Every 50 batches, the mean loss of those batches is compared with the best so far, and two checkpoints are saved, each replacing the one before it: `latest.pt` (everything needed to continue) and `best.pt` (the best weights). If the loss has not improved for `lr_patience_batches` batches, the learning rate is halved. Training stops once the best mean loss is at or below `target_loss` **and** it has not improved for `patience_batches` batches.

### 6. Global Fragments and Seeding

Handled by `globalfrag.py`

There is another important fragment created called a *global fragment*. A frame belongs to a global fragment if it contains exactly $n$ instances with fragments, where $n$ is the number of animals that the user initially configured. For each such frame we take the **sorted list** of its $n$ fragment ids, and a global fragment is a run of consecutive frames with an identical list:

```
frame     fragments present     sorted list
 100      12, 7, 30             [7, 12, 30]   -+
 101      7, 30, 12             [7, 12, 30]    |  one global fragment: frames 100-102
 102      30, 12, 7             [7, 12, 30]   -+
 103      7, 30                 (only 2 of 3)     not part of any global fragment
 104      41, 7, 30             [7, 30, 41]       a new global fragment starts
```

Comparing the lists exactly matters: comparing a sum of the ids instead can give a false match, for example $\{1, 2, 6\}$ and $\{1, 3, 5\}$ both sum to 9.

One global fragment seeds the clusters, because its $n$ members are certainly $n$ different animals. For each member fragment $i$, let $n_i$ be its number of instances and $\sigma_i$ the standard deviation of their SLEAP instance scores. We pick the global fragment with the highest value of

$$\min_i \ \sigma_i \cdot n_i$$

Multiplying by $n_i$ weighs long fragments in full, and taking the minimum means that *every* member must be long, since the weakest member limits how well its animal is represented. If the video has no global fragment, the clusters start from k-means++ instead.

### 7. Prediction and Identity Clustering

Handled by `predict.py`

Every instance that belongs to a fragment is embedded. When `crops.h5` holds only the training sample, the video is streamed again in the same way as in step 4, and each batch of crops is embedded as soon as it is cut and then dropped, so only the embeddings are stored.

Identity clustering uses k-means with $k = n$, as in idTracker.ai. Each cluster starts from the mean embedding of one member of the seeding global fragment (at most 500 of its crops). A crop's probability of belonging to cluster $j$, at distance $d_j$ from its centre, is

$$p_j \propto \left(\frac{1}{d_j + 0.01}\right)^7$$

and its vote is the most probable cluster.

### 8. Identity Assignment

Handled by `assign.py`

We do not use idTracker.ai's P1/P2 cascade. In floating point, its products collapse to exact zeros, which left whole fragments without an identity in our earlier tests. Instead, each fragment is assigned an identity from its votes:

1. Count each fragment's votes over its crops.
2. Order the fragments by vote margin (the top count minus the runner-up), largest first, so that long, confident fragments decide first.
3. In that order, each fragment takes its most-voted identity that no already-assigned fragment in the same frames holds, falling back to its next choice.
4. If the free identities tie on votes, or none is free, the fragment gets no identity.

```
fragment   votes (1, 2, 3)   margin   taken by fragments in the same frames   identity
   L       [600,   0,   0]     600     -                                       1
   M       [  0,   0, 600]     600     1                                       3
   S       [  5,   0,   0]       5     1, 3                                    2  (only one left)
```

Every instance of a fragment gets the fragment's identity. `identities.h5` records, for every instance and fragment, its identity, or why it has none.

### 9. Importing into SLEAP

Handled by `export.py`

Since we used the indexes that were defined in the original SLEAP file, we can hand each instance identity back to the instance that it corresponds to in SLEAP. We write a copy of the input file in which the tracks are the identities (`1`, `2`, ..., `n`), keeping every instance, point and score, and store it as `.sleap_id_predictions.slp` so that it can be opened in the SLEAP GUI. The file is reloaded and checked instance by instance before it replaces any earlier copy.
