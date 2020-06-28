# AI security vulnerabilities and threats	

- [AI security vulnerabilities and threats](#ai-security-vulnerabilities-and-threats)
  * [1. The first one of Attacks categories](#1-the-first-one-of-attacks-categories)
    + [1) Espionage](#1--espionage)
    + [2) Sabotage](#2--sabotage)
    + [3) Fraud](#3--fraud)
  * [2.The second one of Attacks categories](#2the-second-one-of-attacks-categories)
    + [1) Evasion](#1--evasion)
    + [2) Poisoning：Widespread.](#2--poisoning-widespread)
    + [3) Trojianing](#3--trojianing)
    + [4) backdooring](#4--backdooring)
    + [5) reprogramming (adversarial reprogramming)](#5--reprogramming--adversarial-reprogramming-)
    + [6) inference attack (Privacy attack)](#6--inference-attack--privacy-attack-)
  * [3.The third one of Attacks categories](#3the-third-one-of-attacks-categories)
    + [1) attacks on supervised learning (classification)](#1--attacks-on-supervised-learning--classification-)
    + [2) attacks on supervised learning (regression)](#2--attacks-on-supervised-learning--regression-)
    + [3) attacks on semi-supervised learning (generative models)](#3--attacks-on-semi-supervised-learning--generative-models-)
    + [4) attacks on unsupervised learning (clustering)](#4--attacks-on-unsupervised-learning--clustering-)
    + [5) attacks on unsupervised learning (dimensionality reduction)](#5--attacks-on-unsupervised-learning--dimensionality-reduction-)
    + [6) attacks on reinforcement learning](#6--attacks-on-reinforcement-learning)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


Power brings responsibility and AI, like any technology, is not immune to attacks.There are different categories of attacks on ML models depending on the actual goal of an attacker (Espionage, Sabotage, Fraud) and the stages of machine learning pipeline (training and production), or also can be called attacks on algorithm and attacks on a model respectively. They are Evasion, Poisoning, Trojaning, Backdooring, Reprogramming, and Inference attacks. Evasion, poisoning and inference are the most widespread now.
## 1. The first one of Attacks categories
### 1) Espionage
The objective is to glean insights about the system and utilize the received information for his or her own profit or plot more advanced attacks.
### 2) Sabotage
the objective is to disable functionality of an AI system.

There are some ways of sabotaging:
- Flooding AI with requests, which require more computation time than an average example.
- Flooding with incorrectly classified objects to increase manual work on false positives. In case this misclassification takes place, or there is a need to erode trust in this system. For example, an attacker can make the system of video recommendation recommend horror movies to comedy lovers.
- Modifying a model by retraining it with wrong examples so that the model outcome will let down. It only works if the model is trained online.
- Using computing power of an AI model for solving your own tasks. This attack is called adversarial reprogramming.
### 3) Fraud
misclassifying tasks. Attackers has two ways to do it--by interacting with a system at learning stage (Poisoning-poison data) or production stage (Evasion-exploit vulnerabilities to misoperation).
It is the most common attack on machine learning model. It refers to designing an input, which seems normal for a human but is wrongly classified by ML models.


## 2.The second one of Attacks categories
Evasion, Poisoning, Trojaning, Backdooring, Reprogramming, and Inference attacks
### 1) Evasion
Some restrictions should be taken into account before choosing the right attack method: goal, knowledge, and method restrictions.
- Goal restriction (Targeted vs Non-targeted vs Universal)
- Knowledge restriction (White-box, Black-box, Grey-box)
  - Black-box method — an attacker can only send information to the system and obtain a simple result about a class.
  - Grey-box methods — an attacker may know details about dataset or a type of neural network, its structure, the number of layers, etc.
  - White-box methods — everything about the network is known including all weights and all data on which this network was trained.
- Method restriction (l-0, l-1, l-2, l-infinity — norms)：pixel difference
### 2) Poisoning：Widespread.
There are four broad attack strategies for altering the model based on the adversarial capabilities:
- Label modification: Those attacks allows adversary to modify solely the labels in supervised learning datasets but for arbitrary data points. Typically subject to a constraint on total modification cost.
- Data Injection: The adversary does not have any access to the training data as well as to the learning algorithm but has the ability to augment a new data to the training set. It’s possible to corrupt the target model by inserting adversarial samples into the training dataset.
- Data Modification: The adversary does not have access to the learning algorithm but has full access to the training data. The training data can be poisoned directly by modifying the data before it is used for training the target model.
- Logic Corruption: The adversary has the ability to meddle with the learning algorithm. These attacks are referred as logic corruption.
### 3) Trojianing
As to Trojaning, an attacker still don’t have access to the initial dataset but have access to the model and its parameters and can retrain this model
### 4) backdooring
the main goal is not only inject some additional behavior but to do it in such a way that backdoor will operate after retraining the system
### 5) reprogramming (adversarial reprogramming)
 This is the most realistic scenario of a sabotage attack on the AI model. As the name implies, the mechanism is based on remote reprogramming of the neural network algorithms.
### 6) inference attack (Privacy attack)
Most studies currently cover inference attacks at the production stage, but there are still some during training.
- Model inversion attack: most common
- Model extraction attack: less common, is to know the exact model or even a model’s hyperparameters
- Membership inference attack: less frequent, guess if particular data example is in the training dataset
- Attribute inference: guessing a type of data

## 3.The third one of Attacks categories
This map illustrates high-level categories of ML tasks and methods.
 
### 1) attacks on supervised learning (classification)
First attacks on classification dated 2004 (Link: https://homes.cs.washington.edu/~pedrod/papers/kdd04.pdf)
### 2) attacks on supervised learning (regression)
Regression == prediction, and all technical aspects of regression can be divided into two categories: deep learning and machine learning.
### 3) attacks on semi-supervised learning (generative models)
Generative models are designed to simulate the actual data based on the previous decisions. An example was mentioned in the article titled “Adversarial examples for generative models”.
### 4) attacks on unsupervised learning (clustering)
### 5) attacks on unsupervised learning (dimensionality reduction)
### 6) attacks on reinforcement learning
latest one called “Vulnerability of Deep Reinforcement Learning to Policy Induction Attack”

