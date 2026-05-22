Natural language autoencoder complete project documentation


1. Overview

This project is a small scale implementation of a natural language autoencoder inspired by the anthropic paper on natural language autoencoders that produce unsupervised explanations of language model activations.

The goal is to test a simple but important question.

Can we describe a language model's internal activations in plain english and then use that english description to reconstruct the original activations?

When a language model reads text it does not only process words. Inside the model each piece of text is represented as a high dimensional vector of numbers. These internal numerical representations are called activations or hidden states.

If we can translate those activations into natural language and then reconstruct the activations from that language it suggests that the explanation contains meaningful information about what the model internally represented.

In simple terms the project tests this loop.

Text goes into a language model which produces an activation vector. That activation vector is converted into an english explanation. The english explanation is then used to reconstruct an activation vector. The reconstructed vector is then compared against the original to measure quality.

This is useful for artificial intelligence interpretability because it explores whether natural language can help us understand what a neural network is internally encoding.


2. High level workflow

The full pipeline works as follows.

The first stage is the text dataset. It contains 2000 short passages from wikipedia.

The second stage is the small language model. The model used is smollm2 360m which has 360 million parameters. Each text is processed by the model to produce a hidden activation vector of size 960.

The third stage produces the activation vectors. These have a shape of 2000 texts by 960 dimensions and are saved as data/activations.npy.

The fourth stage is the activation verbalizer which works in three steps.
- Step one groups the 2000 activation vectors into 32 clusters using kmeans so that similar activations are placed together
- Step two selects the 5 most representative texts for each cluster
- Step three asks smollm2 to summarize what those texts have in common

The result is 32 natural language explanations one per cluster. An example output would be a sentence like this activation is related to music tours and anniversary releases.

The fifth stage is the sentence embedder. A sentence transformer model converts each english explanation into a numerical embedding. For example the sentence this activation is related to music would become a 384 dimensional sentence embedding.

The sixth stage is the activation reconstructor. A small neural network is trained to map explanation embeddings back to the original activation vectors. The input is a 384 dimensional sentence embedding and the output is a 960 dimensional reconstructed activation vector. The data is split into 1600 training examples 200 validation examples and 200 test examples. The trained model is saved as checkpoints/reconstructor.pt.

The seventh stage is evaluation. The original activation vector is compared with the reconstructed activation vector. The main results are as follows.
- Fraction of variance explained equals 0.938
- Mean squared error equals 5.78
- Cosine similarity equals 0.994

Results are saved as results/eval_results.json and as result plots.


3. Repository structure

The project is organized as a reproducible research repository.

natural-language-autoencoder-small/

config.yaml
README.md
run_all.sh
run_all.ps1

src/
utils.py
data.py
model_utils.py
activations.py
verbalizer.py
llm_verbalizer.py
reconstructor.py
metrics.py
plotting.py

scripts/
01_prepare_data.py
02_collect_activations.py
03b_llm_verbalize.py
04_train_reconstructor.py
05_evaluate.py
06_control_shuffled.py

data/
texts.jsonl
activations.npy
clusters_llm.json
verbalizations.jsonl
splits.json

checkpoints/
reconstructor.pt

results/
eval_results.json
control_shuffled.json
per_cluster_results.csv
qualitative_examples.csv
fve_histogram.png
cosine_histogram.png
fve_per_cluster.png
training_loss.png

app/
streamlit_app.py


4. Configuration file

The config.yaml file acts as the main control panel for the project. It defines important settings such as the model name the dataset size the activation layer the number of clusters the training epochs the batch size the random seed and the output paths.

All scripts read from this file. This makes the experiment easy to modify without editing code in multiple places. For example to change the number of clusters from 32 to 16 only the config file needs to be updated.


5. Source code explanation

The src/ folder contains reusable python modules used by the command line scripts.

utils.py handles general helper functions. It loads the config file sets random seeds handles json input and output and selects cpu or gpu.

data.py downloads and prepares the wikitext dataset. It filters out texts that are too short or too long and selects 2000 examples.

model_utils.py loads the smollm2 model and tokenizer. It also determines which model layer should be used for activation extraction.

activations.py runs text through smollm2 and extracts hidden activations from layer 21. It averages across token positions to produce one 960 dimensional vector per text.

verbalizer.py implements a tfidf baseline verbalizer. It clusters texts and creates keyword based template explanations.

llm_verbalizer.py implements the language model based verbalizer. It asks smollm2 to write a natural language description for each activation cluster.

reconstructor.py defines and trains the multilayer perceptron reconstructor. It maps 384 dimensional sentence embeddings to 960 dimensional activation vectors.

metrics.py computes fraction of variance explained and mean squared error and cosine similarity. It also includes unit tests to verify that the metric implementations are correct.

plotting.py generates the result visualizations and saves them as png files.


6. Script execution order

The project is designed to be run step by step.

Step one is 01_prepare_data.py which downloads and filters the dataset and produces data/texts.jsonl containing 2000 wikipedia passages one per line.

Step two is 02_collect_activations.py which loads smollm2 and passes each text through the model. It extracts hidden activations from layer 21 and saves one 960 dimensional activation vector per text as data/activations.npy. The final matrix has a shape of 2000 by 960.

Step three is 03b_llm_verbalize.py which clusters the activation vectors into 32 groups using kmeans. For each cluster the script selects representative texts and asks smollm2 to describe what they have in common. The outputs are data/clusters_llm.json and data/verbalizations.jsonl.

Step four is 04_train_reconstructor.py which trains the activation reconstructor. The reconstructor takes a sentence embedding as input and predicts the original activation vector. The architecture has an input of 384 dimensions then a hidden layer of 512 dimensions then a hidden layer of 1024 dimensions and finally an output of 960 dimensions. The training split is 1600 training examples 200 validation examples and 200 test examples. The output is checkpoints/reconstructor.pt.

Step five is 05_evaluate.py which evaluates the trained reconstructor on the test set and computes fraction of variance explained and mean squared error and cosine similarity and per cluster scores and per sample scores. The outputs are results/eval_results.json and results/per_cluster_results.csv and results/qualitative_examples.csv and results/fve_histogram.png and results/cosine_histogram.png and results/fve_per_cluster.png and results/training_loss.png.

Step six is 06_control_shuffled.py which runs a sanity check experiment. It randomly shuffles the explanation to activation pairs and trains and evaluates the same reconstructor. The purpose is to test whether the explanations actually carry useful information. The output is results/control_shuffled.json.


7. Evaluation metrics

The project uses three main evaluation metrics.


7.1 Fraction of variance explained

Fraction of variance explained is the main metric. It measures how much of the original activation variance is recovered by the reconstructed activation.

The formula is fraction of variance explained equals one minus reconstruction error divided by baseline error.

The reconstruction error is the error between the original activation and the reconstructed activation. The baseline error is the error from simply guessing the average activation.

A score of 1.0 means perfect reconstruction. A score of 0.0 means no better than guessing the average activation. A score below 0.0 means worse than guessing the average activation.

The project result is 0.938 which means the reconstructor recovered approximately 93.8 percent of the variance in the original activation vectors.


7.2 Mean squared error

Mean squared error measures the average squared difference between the original and reconstructed activation values. Lower is better.

The project result is 5.78. This means the reconstructed 960 dimensional vectors differ from the original vectors by an average squared error of 5.78. The square root of this value is approximately 2.4 which means individual activation dimensions are off by around 2.4 units on average.


7.3 Cosine similarity

Cosine similarity measures whether two vectors point in the same direction. It ignores vector magnitude and focuses only on direction.

A value of 1.0 means the same direction. A value of 0.0 means no directional relationship. A value of negative 1.0 means opposite direction.

The project result is 0.994. At first glance this looks extremely strong. However cosine similarity is misleading in this experiment because even the shuffled control baseline achieved a cosine similarity of approximately 0.990. This suggests that many smollm2 activations already point in broadly similar directions. Therefore cosine similarity alone is not a reliable metric here. Fraction of variance explained is the more informative and honest metric.


8. Main results

The key result is that the real explanations strongly outperform the shuffled explanations on fraction of variance explained and mean squared error.

For fraction of variance explained the real explanations score 0.938 while the shuffled explanations score negative 0.048 giving a difference of 0.986.

For mean squared error the real explanations score 5.78 while the shuffled explanations score 97.57.

For cosine similarity the real explanations score 0.994 while the shuffled explanations score 0.990 giving a difference of only 0.004.

The shuffled control model collapses to negative fraction of variance explained meaning it performs worse than simply predicting the average activation. This shows that the explanations contain real information about the activation clusters.


9. Shuffled control experiment

The shuffled control is the most important sanity check in this project. The question was what happens if the explanations are deliberately assigned to the wrong activations.

To test this the same reconstructor architecture and training setup were used but the explanation to activation pairs were randomly scrambled. If the explanations were not meaningful the model would perform similarly to the real setup. That did not happen.

The real setup achieved a fraction of variance explained of 0.938. The shuffled setup achieved a fraction of variance explained of negative 0.048. This shows that the natural language explanations are carrying useful reconstruction signal.


10. Why the fraction of variance explained score is higher than the scores reported by anthropic

Anthropic reports fraction of variance explained values around 0.6 to 0.8 for their full scale experiments. This project achieved 0.938. However this does not mean this project outperformed the anthropic method.

The reason is that this implementation uses a simplified cluster level verbalizer. In this project all texts in the same cluster receive the same explanation. With 32 clusters there are only 32 possible explanations. This means the reconstructor can partially learn to map explanation 7 to the average activation vector for cluster 7. So the task is easier than the anthropic original setup.

In the anthropic approach each individual activation can receive its own unique natural language explanation. That requires the system to encode much more detailed information into language.

A useful way to understand the difference is this. In this project all sports related activations get explanation number 7 and the reconstructor learns the average vector for cluster 7. In the anthropic approach each activation receives its own detailed explanation and the reconstructor must recover activation details from richer language.

Therefore this project should be understood as a simplified baseline and not a full reproduction of the anthropic results. The current codebase includes infrastructure for a harder per sample mode but running that version is slower and may take approximately six hours on a typical laptop.


11. Per cluster results

The per cluster bar chart shows that most clusters reconstruct well. Many clusters achieve fraction of variance explained values between 0.80 and 0.99. These are well separated activation groups where a single explanation works reasonably well. However some clusters perform poorly. The most interesting failure case is cluster 25.

Cluster 25 contains 86 texts that are not semantically coherent. Examples include the biography of tina fey and aston villa football managers and a franciscan church in omaha and a yiddish character description and lithuanian national anthem law. These texts were grouped together because their activation vectors were numerically close under kmeans even though their topics were unrelated.

The language model was shown the most representative texts and generated an explanation about youth development and mentorship particularly in relation to autism speaks. That explanation was then assigned to all texts in the cluster including unrelated texts such as lithuanian legal text. The result was a fraction of variance explained of negative 0.05 for cluster 25.

This is an important failure mode. The issue is not necessarily that the language model wrote a bad description. Instead the clustering step created a semantically mixed group. This suggests that future versions should explore better clustering algorithms and more clusters and per sample verbalization and semantic filtering and manual cluster inspection.


12. Fraction of variance explained distribution

The fraction of variance explained histogram gives a more detailed view than the global average.

The median fraction of variance explained is 0.895. The mean fraction of variance explained is 0.781. The median is higher than the mean because a small number of poor reconstructions pull the average down. The distribution is left skewed meaning most examples reconstruct well and a smaller number of bad clusters create a long tail of poor scores.

This is why it is important to report more than just the global fraction of variance explained score. The histogram reveals failure cases that a single average number would hide.


13. Training curve

The training curve shows both training loss and validation loss over 40 epochs.

The training loss decreases steadily. The validation loss also decreases steadily. The two curves remain close together. This suggests that the reconstructor learned a useful mapping without severe overfitting. If the training loss had decreased while validation loss increased that would indicate overfitting. That did not happen here. The model converged successfully within 40 epochs.


14. Qualitative examples

The file results/qualitative_examples.csv contains selected examples from the test set. It includes strong examples and weak examples and average examples.

The best reconstruction example has a fraction of variance explained of 0.995. The original text is about the storm name graham being retired from the australian region basin. The assigned explanation is that this activation is related to the shared content of the passage describing australian geography. This example works well because the cluster is topically coherent and the explanation is relevant to the source text.

The worst reconstruction example has a fraction of variance explained of negative 0.16. The original text is about the law on the national anthem of the republic of lithuania signed by president valdas adamkus. The assigned explanation is about youth development and mentorship particularly in relation to autism speaks. This is a clear failure. The lithuanian legal text was assigned an explanation from an unrelated cluster. This demonstrates the main weakness of the cluster level verbalizer.


15. Streamlit dashboard

The project includes a streamlit dashboard for visual inspection of the results. To launch it run streamlit run app/streamlit_app.py and then open http://localhost:8501 in your browser. The dashboard is not part of the core methodology. It is provided as a convenient tool for exploring the experiment outputs.

The dashboard has the following pages.

The overview page shows the model name the dataset size the layer used and the configuration settings.

The metrics and control page displays fraction of variance explained and mean squared error and cosine similarity and the real versus shuffled comparison.

The training curve page shows the training and validation loss over 40 epochs.

The plots page displays the saved result visualizations.

The qualitative examples page shows selected examples with color coded reconstruction quality.

The cluster explanations page lists all 32 clusters with their language model generated descriptions and representative texts.

The custom text demo page lets the user type a sentence and then extract its activation and find the nearest cluster and view the assigned explanation.


16. Running the full pipeline

On windows run the following command.

cd "d:\sir tariq\natural-language-autoencoder-small"
.\run_all.ps1

On linux or macos or colab run the following command.

bash run_all.sh

The approximate runtime on a normal cpu only laptop is around 20 minutes total. Activation extraction takes approximately 13 minutes. Language model verbalization takes approximately 3.5 minutes. All other steps take under 1 minute.


17. Output files

data/texts.jsonl contains 2000 wikipedia passages one json object per line.

data/activations.npy contains a 2000 by 960 matrix of activation vectors.

data/clusters_llm.json contains 32 cluster descriptions generated by smollm2.

data/verbalizations.jsonl contains each text paired with its assigned natural language explanation.

data/splits.json contains the train validation and test split indices.

checkpoints/reconstructor.pt contains the trained multilayer perceptron reconstructor weights.

results/eval_results.json contains the main evaluation metrics including fraction of variance explained and mean squared error and cosine similarity.

results/control_shuffled.json contains the shuffled control experiment results.

results/per_cluster_results.csv contains per cluster fraction of variance explained and cosine similarity and mean squared error.

results/qualitative_examples.csv contains selected best and worst and average examples.

results/fve_histogram.png is a histogram of per sample fraction of variance explained scores.

results/cosine_histogram.png is a histogram of per sample cosine similarity scores.

results/fve_per_cluster.png is a bar chart showing average fraction of variance explained for each cluster.

results/training_loss.png shows training and validation loss across 40 epochs.


18. Why this project matters

This project explores a central question in artificial intelligence interpretability.

Can natural language describe what a neural network is internally representing?

If this becomes reliable it could help researchers understand model internals and debug model failures and detect dangerous or unwanted internal representations and audit artificial intelligence reasoning more effectively and build more transparent artificial intelligence systems.

This project is only a small scale baseline but it demonstrates the complete experimental loop. Activations are extracted. Activations are described. Activations are reconstructed. Reconstruction quality is measured. Failures are analyzed.

The shuffled control experiment confirms that the english descriptions are not arbitrary. They contain meaningful information that helps reconstruct activation vectors.


19. Limitations

This project should be understood as a simplified baseline and not a full reproduction of the anthropic work.

The main limitations are as follows.

Only 2000 text samples were used. Only one small model was tested. Only one layer was analyzed. The verbalizer works at cluster level and not per individual activation. There are only 32 possible explanations. The reconstructor may learn cluster averages rather than deeply understanding language. Kmeans can create semantically mixed clusters. Cosine similarity is misleading because even shuffled controls score highly.

The most important limitation is the cluster level verbalizer. Because every example in a cluster receives the same explanation the task is easier than the full natural language autoencoder problem.


20. Future improvements

Possible next steps include the following.

Use per sample verbalization instead of cluster level verbalization. Test multiple model layers. Compare multiple open source language models. Increase the dataset size. Use better clustering methods than kmeans. Generate richer explanations. Add human evaluation of explanation quality. Train a stronger reconstructor. Compare cluster based and tfidf and language model based verbalizers. Run ablations for number of clusters. Evaluate whether explanations generalize across datasets.

The most important improvement would be moving from one explanation per cluster to one explanation per activation. That would make the experiment closer to the anthropic original methodology.


21. Summary

This project implemented a small scale natural language autoencoder pipeline.

We downloaded 2000 wikipedia passages and passed them through smollm2 360m which is a 360 million parameter language model. From layer 21 of the model we extracted a 960 dimensional activation vector for each text.

The activation vectors were grouped into 32 clusters using kmeans. For each cluster smollm2 generated a natural language explanation based on representative examples from that cluster. Each text then received the explanation associated with its cluster.

Next each explanation was converted into a 384 dimensional sentence embedding using minilm l6. A small multilayer perceptron reconstructor was trained to map those sentence embeddings back to the original 960 dimensional activation vectors.

On the held out test set the model achieved a fraction of variance explained of 0.938 and a mean squared error of 5.78 and a cosine similarity of 0.994.

A shuffled control experiment where explanations were randomly assigned to activations collapsed to a fraction of variance explained of negative 0.048. This confirms that the real explanations carry meaningful signal for reconstruction.

However the high fraction of variance explained should be interpreted carefully. Because this implementation uses only 32 cluster level explanations the reconstructor can partially learn cluster average activation vectors. This makes the task much easier than the anthropic full per activation setting.

Overall the project successfully demonstrates the core structure of a natural language autoencoder. Activations go in. Explanations come out. Activations are reconstructed. Quality is evaluated. It provides a clear reproducible baseline for further experimentation with natural language explanations of language model activations.
