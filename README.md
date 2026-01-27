<h1>Epoch-Driven Feature Permutation for Time Series</h1>

<p>
This experiment explores whether dynamically reordering input features during
training can improve model performance in noisy, short window time series.
Rather than fixing feature ordering or relying solely on model adaptation,
this work treats permutation itself as a learnable, epoch-driven decision process.
</p>

<h2>Core Idea</h2>
<pre>
Baseline Phase →
Permutation Proposal →
Adaptation Phase →
Evaluation →
Accept / Reject →
Temperature Controller Update
</pre>

<p>
The underlying hypothesis is that the optimal feature order is not static:
temporally unstable or weak signals may be better captured if the model
can explore different feature arrangements and adapt in between.
</p>

<h2>Permutation Experiments</h2>
<p>
The script implements multiple permutation driven branches, each with its
own adaptive temperature controller:
</p>

<ul>
  <li>Learnable feature permutation layers using Sinkhorn normalization</li>
  <li>Gradient-based proposals for small permutation updates</li>
  <li>Epoch-aggregated entropy and repulsion metrics to guide updates</li>
  <li>Acceptance checks based on relative validation improvement and stability</li>
</ul>

<p>
Permutations are applied only during the proposal phases; model adaptation
occurs only after a proposed permutation, preventing unstable gradient interference.
</p>

<h2>Advanced Permutation Control</h2>
<p>
At the center of the experiment is a <strong>temperature controller</strong> for
each permutation branch, which:
</p>

<ul>
  <li>Regulates the "softness" of the permutation via Sinkhorn temperature</li>
  <li>Adapts in response to validation loss and entropy metrics</li>
  <li>Encodes uncertainty, allowing branches to freeze when stable</li>
</ul>

<p>
This design separates learning the feature arrangement from learning the
prediction function, explicitly encoding exploration vs exploitation.
</p>

<h2>Model Architecture</h2>
<ul>
  <li>Four parallel branches with GRU, LSTM, Conv1D, and TimeDistributed dense layers</li>
  <li>Advanced gated readout combines multiple temporal summaries</li>
  <li>Feature permutations applied per-branch before temporal modeling</li>
  <li>Adaptive weighting of branches through learned per timestep gating</li>
</ul>

<h2>Permutation Search Cycle</h2>
<p>
The high-level training cycle alternates between baseline training,
proposing permutations, and adaptation. Acceptance is determined
by relative improvement on validation loss and stability windows.
</p>

<ul>
  <li>Baseline phase: train with fixed permutations</li>
  <li>Propose phase: update permutation logits via gradient or custom step</li>
  <li>Adaptation phase: train model to adjust to proposed permutation</li>
  <li>Evaluation & acceptance: relative improvement check, stability verification</li>
  <li>Temperature controller update: adjusts exploration softness per branch</li>
</ul>

<h2>What This Pushes Against</h2>
<ul>
  <li>Static feature ordering assumptions</li>
  <li>One-shot permutation or brute-force feature selection</li>
  <li>Rigid temporal modeling without adaptive exploration</li>
</ul>

<h2>What This Is Not</h2>
<ul>
  <li>A production ready forecasting framework</li>
  <li>A guaranteed path to state-of-the-art performance</li>
  <li>A demonstration of “easy” feature reordering</li>
</ul>

<h2>Why Explore This?</h2>
<p>
In short-horizon time series, signals are often fleeting, noisy, and
regime-dependent. The feature arrangement can dramatically influence
how temporal patterns are extracted. By allowing the model to explore
and adapt feature permutations in an epoch driven manner, we probe
whether structural flexibility can improve learning robustness.
</p>

<blockquote>
When features are weakly informative, how you order them can matter as much as how you model them.
</blockquote>
