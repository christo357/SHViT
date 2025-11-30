1.  Data-efficiency analysis (core new contribution)   
    
    1. learning curves across datasets: 
        - Seperate for each dataset.
            •   X-axis: train fraction
            •	Y-axis: best test_acc1
            •	Curves: shvit vs deit_tiny

        Report : 
        1. for each model & dataset,  data @ 90% : min. fraction of data where acc.of fracton >=90%. 

        Problems answered:
        1. How many labeled examples each model needs to reach a given accuracy.
        2. Whether SHViT is more data-efficient than DeiT-Tiny in low-data regimes on CIFAR / EuroSAT / MedMNIST.

    2. Area Under Data–Accuracy Curve (AUDC)
        For each (model, dataset):
        •	You have (fraction, best_acc).
        •	Numerically integrate (trapezoidal) over fractions:

        \text{AUDC = \int_0^{1 \text{Acc}(\text{fraction}) \, d(\text{fraction})

        (approx via discrete sum).

        Interpretation:
            •	Higher AUDC = better overall data efficiency (strong performance even at low data).
            •	You can make a table:
            dataset, model , audc



        details:
            It’s one scalar per (model, dataset) that summarizes the entire learning curve over fractions, not a value “for each fraction”.

            So:
                •	X-axis: train fraction (e.g. 0.1, 0.325, 0.55, 0.775, 1.0)
                •	Y-axis: best test_acc1 achieved at that fraction
                •	AUDC = area under that curve

            Concretely

                For each dataset D and model M:
                    1.	Collect points:
                (\text{fraction}_i, \text{acc}_i), \quad i=1,\dots,k
                where:
                    •	fraction_i ∈ {0.1, 0.325, …, 1.0}
                    •	acc_i = best test_acc1 across epochs at that fraction
                    2.	Sort by fraction_i.
                    3.	Compute a discrete integral, e.g. trapezoidal rule:

                AUDC(M, D) \approx \sum_{i=1}^{k-1} [ (acc_i + acc_{i+1})/{2} * (fraction_{i+1} - fraction_i) ]

                That gives you one number:
                    •	AUDC_shvit_cifar
                    •	AUDC_deit_cifar
                    •	AUDC_shvit_eurosat
                    •	etc.

                You don’t compute an AUDC “for each fraction”; the fractions are the x-coordinates you integrate over.



2.  Optimization dynamics & generalization (within training)

    1. Convergence speed (epochs to X% of final accuracy)
        - epochs to reach 95 % acc.
        curve:
        - X-axis: train fraction
	    - Y-axis: Epochs@95%final
	    - Curves: SHViT vs DeiT-Tiny

    2. Generalization gap vs data fraction

        For each epoch:

        GenGap = train loss - testloss

        For each (model, dataset, fraction):
            •	Compute GenGap at:
            •	The epoch with best test_acc1, or
            •	Final epoch.

        Then plot:
            •	X-axis: train fraction
            •	Y-axis: GenGap
            •	Curves: SHViT vs DeiT-Tiny

        What it shows:
            •	Does one model overfit more at small fractions?
            •	Is SHViT more “regularized” / DeiT more prone to overfitting in low-data regimes, or vice versa?

        can even form table:
        Dataset, Fraction, Model , Best test_acc1, GenGap at best epoch
        CIFAR,   10%.    , SHViT , ….             ,…
        CIFAR,   10%.    , Deit-tiny , ….             ,…


3. Cross-domain comparison of data efficiency (inductive-bias-ish)

        Now combine the above across datasets.

    3.1 “Data@X%” comparisons across domains

        For each model, dataset:
            •	Compute Data@70%, Data@80%, Data@90% of full-data accuracy.

        Now you can ask:
            •	On CIFAR (natural-ish images), does SHViT need less data than DeiT to reach 80% of its own max performance?
            •	On EuroSAT (satellite), maybe SHViT needs more data than DeiT?
            •	On MedMNIST (medical), maybe both need much more data to reach the same relative performance.

        Table formed:
        Dataset, Model, Full acc, Data@80%, Data@90%
        CIFAR  , SHViT, X%.     , a%      , b%
        CIFAR  , Deit-tiny, X%.     , a%      , b%


        qns answered:
        1. 	If SHViT needs less data on certain domains, that suggests its architecture has a favorable inductive bias there (e.g., maybe for structured textures in EuroSAT).
        2.	If DeiT dominates on another domain, that’s also interesting.

    
    3.2 Normalized performance across domains

        Sometimes SHViT and DeiT might simply have different capacity. To see inductive bias more cleanly, normalize by each model’s full-data accuracy:

        RelAcc(fraction) = Acc(fraction) / Acc(100\%)

        Now:
            •	Plot RelAcc vs fraction, per dataset, for each model.
            •	This says: “Given the best this model can do on this dataset, how quickly does it get there?”

        You can then compare:
            •	Does SHViT ramp faster (RelAcc is higher at small fractions) on EuroSAT than DeiT?
            •	Does DeiT ramp faster on MedMNIST?

        That is a very clean way to talk about data efficiency as part of inductive bias, and it’s not in the SHViT paper




### Test that can be conducted: ( mainly all on cifar)

1( and 2), 3, 7, 

1. Corruption / Noise Robustness (per-domain) ( mainly on cifar)( maybe on eurosat and medmnist for cross domain)

    •	CIFAR:
        •	Gaussian noise
        •	Motion blur
        •	Brightness/contrast change
        •	JPEG compression
	•	EuroSAT:
        •	Gaussian noise (sensor noise)
        •	Downsample+upsample (low resolution)
        •	Light fog / haze (brightness + contrast tweak)
	•	MedMNIST:
        •	Gaussian noise
        •	Slight rotation
        •	Contrast change


        1.1 CIFAR → “CIFAR-C style” corruptions

        What: Evaluate each trained model on corrupted versions of CIFAR (you can use CIFAR-10-C/100-C if you’re on CIFAR-100; or build your own with torchvision.transforms).

        Corruptions to cover (a small but meaningful set):
            •	Noise: Gaussian, shot, impulse
            •	Blur: motion blur, defocus blur
            •	Color/brightness: brightness, contrast
            •	Digital: JPEG compression, pixelation

        How:

        For each (model, fraction):
            1.	Take the best checkpoint you already saved.
            2.	Build a test loader where you apply one corruption to each image (at a fixed severity).
            3.	Compute test_acc1.
            4.	Repeat for each corruption type and maybe 2–3 severities.

        Metrics / plots:
            •	Per-corruption robustness: acc_corr / acc_clean (relative robustness).
            •	Corruption family mean: average over noise / blur / color / digital.
            •	Plot: bar plot of robustness per corruption family for SHViT vs DeiT; optionally lines vs fraction (robustness learning curves).

        1.2 EuroSAT → domain-specific corruptions

            EuroSAT is satellite imagery, so use “remote-sensing-ish” corruptions:
                •	Atmospheric noise: Gaussian + speckle noise
                •	Resolution changes: strong downsampling + upsampling
                •	Cloud / haze simulation: light fog / brightness shifts (ColorJitter, RandomAffine with low contrast)

            How: same routine as CIFAR:
                •	For each checkpoint, evaluate on several corrupted test versions.
                •	Compare robustness vs clean accuracy.

            Question you answer:

            Does SHViT’s single-head design behave differently from DeiT when spatial resolution or atmospheric effects are degraded?

        1.3 MedMNIST → medical-style perturbations

            MedMNIST is grayscale/small medical images. Good corruptions:
                •	Additive noise (simulating sensor noise).
                •	Contrast changes (over/under-exposure).
                •	Small rotations / flips (pose / acquisition changes).
                •	Random erasing / occlusion (missing tissue or artifacts).

            Same evaluation pattern:
                •	Accuracy drops vs clean.
                •	Compare SHViT vs DeiT across corruptions and fractions.

2.  “Efficiency vs Robustness” Curves (using your fractions)

    This is the nice extension of your existing learning curves.

    For each (dataset, model):
        1.	For each fraction:
        •	Evaluate the checkpoint on clean test and on corrupted test (choose 1–2 “summary” corruptions or an average over several).
        2.	Plot:

        •	X-axis: train fraction
        •	Y-axis: accuracy
        •	Two curves per model: clean vs corrupted

    This tells you:
        •	How robustness scales with data for each architecture.
        •	Whether SHViT “needs” more data than DeiT to become robust, or whether its robustness is more data-efficient.


3. Geometric & Color Invariance Tests( cifar)

    These are easy, no extra dataset needed:

    Build test loaders with only:
        •	Geometric: rotations (±15°, ±30°), horizontal flips, small crops & resizes.
        •	Color: grayscale conversion, heavy color jitter.

    For each (model, fraction):
        •	Evaluate on:
        •	clean test
        •	rotated test
        •	grayscale test, etc.

    Metrics:
        •	Invariance score: acc_aug / acc_clean.
        •	Compare SHViT vs DeiT to see which is more stable to geometry vs color changes on each domain.

4. Calibration & Confidence (from logits)

    From the checkpoints you can also look at how well-calibrated the models are.

    For each (model, dataset, fraction):
        1.	Save softmax probabilities on the clean test set.
        2.	Compute:
        •	Expected Calibration Error (ECE)
        •	Brier score
        3.	Make reliability diagrams (predicted prob vs empirical accuracy in bins).

    You can also do this on corrupted test sets:
        •	Does SHViT become overconfident on corruptions more than DeiT?
        •	Does calibration degrade more or less with lower train fractions?

    This gives you a “trustworthiness” dimension beyond raw accuracy.

5. Simple OOD Detection Test( cifar )

    You can use your existing models as OOD detectors using just confidence:
        1.	For a model trained on *CIFAR*:
        •	In-distribution: CIFAR test.
        •	OOD: something like EuroSAT resized, or noise images, or MedMNIST upsampled.
        2.	Compute:
        •	Max softmax probability for each sample.
        •	Plot histograms for ID vs OOD.
        •	Compute AUROC / AUPR for “is this ID?” based on confidence.

    Repeat for SHViT vs DeiT:
        •	Which model gives more separation between ID and OOD confidence?
        •	Does that change with training fraction?

    You can do the same trick swapping “domains” (EuroSAT as ID, CIFAR/MedMNIST as OOD).

6. Representation Analysis via Probing (lightweight)

    Without retraining from scratch, you can:
        1.	Freeze the backbone (SHViT vs DeiT) trained on, say, full CIFAR fraction.
        2.	Train a linear layer on top for another dataset (e.g., DTD, FGVC-Aircraft, or even EuroSAT/MedMNIST).

    Compare:
        •	Linear-probe accuracy of SHViT vs DeiT with the same training data.
        •	That shows which backbone learned more transferable features in each domain.

    You can also compare:
        •	Linear probe performance when the backbone was trained at 10% vs 100% fraction (data-efficiency of representation learning).

7. Qualitative: Saliency / Attention Maps (for thesis figures)

    Finally, for a couple of nice qualitative plots:
        •	Use Grad-CAM or attention rollout to visualize:
        •	Where SHViT looks vs where DeiT looks on the same image (CIFAR, EuroSAT).
        •	Show:
        •	An example where SHViT is robust to noise but DeiT fails (and vice versa).
        •	How the focus changes under corruption.

    This is great for a short “interpretability / inductive bias” visual subsection.






ood test:
Nice catch—that’s exactly the kind of thing people get stuck on, and the good news is: different class counts are not a problem for OOD tests. 🌈

Because in OOD detection, we don’t care what class the OOD image “is”, we only care how confident the model is that it belongs to some in-distribution class.

⸻

1. Key idea: OOD doesn’t need matching labels

Take this setup:
	•	Model trained on CIFAR-100 → 100 classes (ID domain)
	•	You want to use EuroSAT (10 classes) or MedMNIST (9, 7, etc. classes) as OOD.

When you feed a EuroSAT image into the CIFAR-trained model:
	•	The model still outputs a probability vector of size 100 (over CIFAR classes).
	•	Those class names are wrong for EuroSAT—but that’s fine.
	•	For OOD, we never use the EuroSAT labels at all.

We only use:
	•	ID: CIFAR test images → confidences from CIFAR model
	•	OOD: EuroSAT/MedMNIST images → confidences from the same CIFAR model

Then we ask:

“Are the confidence values different between ID and OOD?”

So class mismatch doesn’t matter.

⸻

2. Concrete workflow with different class counts

Let’s fix:
	•	ID domain = CIFAR (model trained on CIFAR with C_id classes)
	•	OOD domain = EuroSAT (10 classes) or MedMNIST

Step 1 – Run the model on ID & OOD

import torch
import torch.nn.functional as F

model.eval()

# --- ID data (CIFAR test) ---
all_logits_id = []
all_labels_id = []  # for sanity / accuracy check (optional)
for x, y in cifar_test_loader:
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)       # shape [N_batch, C_id]
    all_logits_id.append(logits.cpu())
    all_labels_id.append(y)

logits_id = torch.cat(all_logits_id, dim=0)   # [N_id, C_id]
labels_id = torch.cat(all_labels_id, dim=0)   # [N_id]

probs_id = F.softmax(logits_id, dim=1)
conf_id, preds_id = probs_id.max(dim=1)       # [N_id], [N_id]

# --- OOD data (EuroSAT / MedMNIST) ---
all_logits_ood = []
for x, _ in ood_loader:  # ignore OOD labels
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)       # still [N_batch, C_id]
    all_logits_ood.append(logits.cpu())

logits_ood = torch.cat(all_logits_ood, dim=0)  # [N_ood, C_id]
probs_ood = F.softmax(logits_ood, dim=1)
conf_ood, preds_ood = probs_ood.max(dim=1)     # [N_ood], [N_ood]

Notice:
	•	EuroSAT might “really” have 10 classes, but the model just forces them into 100 CIFAR classes.
	•	We ignore the semantics and only look at conf_ood vs conf_id.

⸻

3. How to turn this into an OOD detection metric

3.1 Confidence histograms (most intuitive)

For each model (SHViT, DeiT):
	•	Plot two histograms:
	•	ID confidences = conf_id
	•	OOD confidences = conf_ood

You want to see:
	•	ID confidences mostly high (e.g. 0.7–1.0)
	•	OOD confidences lower (e.g. 0.0–0.4)

Compare SHViT vs DeiT:
	•	If SHViT’s OOD confidences are lower / more separated, you can say:
“SHViT is more cautious on out-of-domain inputs than DeiT-Tiny.”

3.2 AUROC: single scalar score

Build a binary classification problem: “is this sample ID or OOD?”

import torch
from sklearn.metrics import roc_auc_score

# Scores: higher = more likely ID
scores_id = conf_id        # [N_id]
scores_ood = conf_ood      # [N_ood]

scores = torch.cat([scores_id, scores_ood]).numpy()
labels_bin = torch.cat([
    torch.ones_like(scores_id),     # 1 for ID
    torch.zeros_like(scores_ood)    # 0 for OOD
]).numpy()

auroc = roc_auc_score(labels_bin, scores)
print("AUROC ID vs OOD:", auroc)

	•	AUROC ~1.0 → confidence perfectly distinguishes ID vs OOD.
	•	AUROC ~0.5 → confidence is useless for OOD (completely overlapping).

Again: no need for class-label compatibility. We never use EuroSAT/MedMNIST labels in this metric.

⸻

4. How to use multiple datasets cleanly

You can repeat the same idea with each dataset as ID in its own experiment:
	1.	ID = CIFAR → OOD = EuroSAT, MedMNIST
	2.	ID = EuroSAT → OOD = CIFAR, MedMNIST
	3.	ID = MedMNIST → OOD = CIFAR, EuroSAT

For each, you have:
	•	Train a model on ID dataset
	•	Use other datasets only as blind OOD inputs
	•	Always use the ID-trained model (class count = #ID classes)
	•	Only look at confidence distributions and AUROC

You never need the labels to match across datasets.

⸻

5. What you don’t do in OOD

Just to be super clear:
	•	You do not compute “accuracy on EuroSAT using CIFAR labels” → meaningless.
	•	You do not relabel EuroSAT into CIFAR classes.
	•	You do not change the model’s number of output classes when switching OOD.

You only:
	•	Train model on its own dataset (CIFAR, EuroSAT, or MedMNIST).
	•	At test time, use that trained model to score both ID test and foreign OOD images.
	•	Compare confidence behavior.

⸻

If you want, I can help you pick one clean OOD setup (e.g., “CIFAR as ID, EuroSAT as OOD, full-data SHViT vs DeiT”) and write a short, self-contained eval_ood.py script around this idea.