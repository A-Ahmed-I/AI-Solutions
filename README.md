# Early Detection of Parkinson's Disease Using Multimodal AI

> A multimodal artificial intelligence system for the early screening of Parkinson's disease through spiral drawing analysis, wave drawing analysis, and voice signal processing.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20In%20Progress-orange?style=flat-square)
![ML](https://img.shields.io/badge/Domain-Machine%20Learning%20%7C%20Medical%20AI-purple?style=flat-square)

---

## Table of Contents

- [Introduction](#-introduction)
- [Current Treatments](#-current-treatments)
- [Problem Statement](#-problem-statement)
- [Proposed AI Solution](#-proposed-ai-solution)
- [System Overview](#-system-overview)
- [Goal of the Project](#-goal-of-the-project)

---

## Introduction

### What is Parkinson's Disease?

**Parkinson's disease (PD)** is a progressive neurodegenerative disorder that primarily affects the motor system. It is the second most common neurodegenerative disease worldwide, after Alzheimer's disease, affecting approximately **10 million people globally** (Parkinson's Foundation, 2023).

### Neurological Basis

Parkinson's disease is caused by the **gradual loss of dopaminergic neurons** in a region of the brain called the _substantia nigra_, a component of the basal ganglia system responsible for coordinating smooth and controlled movement. Dopamine is a critical neurotransmitter that facilitates communication between the substantia nigra and the striatum, enabling fluid motor function. As dopamine-producing neurons degenerate, dopamine levels decline significantly, disrupting the neural circuits that regulate movement.

In addition to dopaminergic cell loss, research has identified the abnormal accumulation of a protein called **alpha-synuclein**, which forms toxic aggregates known as _Lewy bodies_ within neurons. These aggregates are considered a hallmark of Parkinson's pathology and contribute to neuronal dysfunction and death.

### Common Symptoms

Parkinson's disease manifests through a wide range of **motor and non-motor symptoms**, which worsen progressively over time:

**Motor Symptoms:**

- **Resting tremor** — involuntary shaking, typically beginning in one hand
- **Bradykinesia** — slowness of movement, making everyday tasks difficult
- **Muscular rigidity** — stiffness and resistance in limbs and trunk muscles
- **Postural instability** — impaired balance and coordination, increasing fall risk
- **Micrographia** — progressively smaller and more cramped handwriting
- **Hypophonia** — reduced voice volume and monotone speech

**Non-Motor Symptoms:**

- Cognitive impairment and dementia (in later stages)
- Depression and anxiety
- Sleep disturbances (e.g., REM sleep behavior disorder)
- Autonomic dysfunction (e.g., constipation, orthostatic hypotension)
- Olfactory dysfunction (loss of smell) — often one of the earliest signs

---

## Current Treatments

While there is currently **no cure** for Parkinson's disease, several treatment modalities are available to manage symptoms and improve quality of life:

### Pharmacological Treatments

- **Levodopa / Carbidopa (Sinemet):** The gold standard pharmacological treatment. Levodopa is converted to dopamine in the brain, temporarily alleviating motor symptoms. Carbidopa prevents peripheral conversion of levodopa, improving efficacy.
- **Dopamine Agonists** (e.g., pramipexole, ropinirole): Mimic dopamine's effects in the brain and are often used in early-stage disease.
- **MAO-B Inhibitors** (e.g., selegiline, rasagiline): Slow the breakdown of dopamine, extending its availability.
- **COMT Inhibitors** (e.g., entacapone): Reduce the breakdown of levodopa, prolonging its effect.
- **Anticholinergics:** Help control tremor in some patients, particularly younger individuals.

### Surgical and Interventional Treatments

- **Deep Brain Stimulation (DBS):** A neurosurgical procedure in which electrodes are implanted in specific brain regions (e.g., subthalamic nucleus) and connected to a pulse generator. DBS significantly reduces motor symptoms in advanced-stage patients.
- **Focused Ultrasound Thalamotomy:** A non-invasive procedure using focused ultrasound to ablate targeted brain tissue, providing tremor relief.

### Non-Pharmacological Therapies

- **Physiotherapy:** Targeted exercises to improve mobility, balance, and gait.
- **Speech and Language Therapy (LSVT LOUD):** Specialized voice training to address hypophonia and dysarthria.
- **Occupational Therapy:** Adaptive strategies and tools to support daily living activities.
- **Nutritional guidance and psychological support** to address non-motor symptoms.

---

## Problem Statement

### The Challenge of Early Diagnosis

One of the most critical challenges in Parkinson's disease management is the **significant delay between the onset of neurodegeneration and clinical diagnosis**. By the time a patient is formally diagnosed, it is estimated that approximately **60–80% of dopaminergic neurons** in the substantia nigra have already been irreversibly lost.

### Why Early Detection is Difficult

- **Gradual and insidious onset:** Early symptoms — such as reduced sense of smell, constipation, sleep disturbances, and subtle tremors — are non-specific and often attributed to aging or other conditions.
- **Lack of definitive biomarkers:** There is currently no single reliable blood test or imaging biomarker that can confirm Parkinson's disease in its early stages. Diagnosis remains primarily **clinical**, based on physician observation of motor symptoms.
- **Subjective clinical assessment:** Standard diagnostic tools, such as the Unified Parkinson's Disease Rating Scale (UPDRS), rely heavily on a neurologist's subjective judgment and require specialized expertise that is not universally accessible.
- **Limited access to specialists:** In many regions, particularly in low- and middle-income countries, access to movement disorder specialists and neurological diagnostic facilities is severely limited.
- **Symptom overlap:** Early-stage PD symptoms overlap with other conditions such as essential tremor, normal pressure hydrocephalus, and drug-induced parkinsonism, leading to frequent misdiagnoses.

### Consequences of Late Diagnosis

Late diagnosis means that patients begin treatment only after substantial neuronal damage has occurred, significantly limiting the effectiveness of neuroprotective interventions and reducing the potential window for disease-modifying therapies.

---

## Proposed AI Solution

### Multimodal AI-Based Early Screening

This project proposes a **multimodal artificial intelligence system** capable of detecting early indicators of Parkinson's disease by analyzing three complementary data modalities:

### 1. Spiral Drawing Test

The spiral drawing test is a well-established clinical tool for assessing motor dysfunction. Patients are asked to trace or draw an Archimedean spiral on a digitizing tablet or paper. In individuals with Parkinson's disease, the drawings exhibit characteristic abnormalities such as:

- Irregular spacing and tremor-induced oscillations
- Reduced drawing speed and pressure variation
- Loss of smoothness and symmetry

Deep learning-based image analysis (e.g., Convolutional Neural Networks) can extract subtle spatial features from spiral images that may not be perceptible to the human eye.

### 2. Wave Drawing Test

Similar to the spiral test, the wave (sinusoidal) drawing test requires patients to draw continuous wave patterns. Parkinson's-related motor impairments manifest as:

- Amplitude variability and frequency irregularities
- Tremor artifacts superimposed on the waveform
- Reduced linearity and stroke control

Feature extraction from wave drawings provides complementary information about fine motor control not fully captured by the spiral test alone.

### 3. Voice Recording Analysis

Parkinson's disease causes characteristic changes in vocal quality, including:

- **Jitter** (cycle-to-cycle variation in fundamental frequency)
- **Shimmer** (amplitude variation between consecutive cycles)
- **Harmonic-to-Noise Ratio (HNR)** reduction
- **Dysphonia** and breathiness

Acoustic feature extraction from sustained phonation recordings (e.g., the vowel /a/) and connected speech enables non-invasive assessment of vocal biomarkers strongly correlated with PD severity.

### Why Multimodal Fusion?

Each modality provides distinct and complementary information about the neuromotor system. Fusing all three modalities enables:

- Higher diagnostic accuracy compared to any single modality alone
- Greater robustness to noise and inter-subject variability
- A richer representation of the patient's neuromotor state

---

## System Overview

The proposed system follows a structured end-to-end pipeline:

```
┌──────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                          │
│                                                              │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│   │ Spiral Image │   │  Wave Image  │   │  Voice Recording │ │
│   └──────┬───────┘   └──────┬───────┘   └─────────┬────────┘ │
└──────────┼──────────────────┼─────────────────────┼──────────┘
           │                  │                     │
           ▼                  ▼                     ▼

┌──────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION                        │
│                                                              │
│  • CNN-based image features (shape, texture, tremor)         │
│  • Geometric & spatial descriptors (drawing dynamics)        │
│  • Acoustic features: MFCC, Jitter, Shimmer, HNR, Pitch      │
│    and Energy from voice signals                             │
└───────────────┬───────────────────────────────┬──────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼

┌──────────────────────────────────────────────────────────────┐
│                     MULTIMODAL FUSION                        │
│        Feature Concatenation / Attention-Based Fusion        │
└───────────────────────────────┬──────────────────────────────┘
                                ▼

┌──────────────────────────────────────────────────────────────┐
│                ML / DEEP LEARNING CLASSIFIER                 │
│     Random Forest | SVM | CNN | LSTM | Transformer           │
│                     Ensemble Models                          │
└───────────────────────────────┬──────────────────────────────┘
                                ▼

┌──────────────────────────────────────────────────────────────┐
│                     PREDICTION OUTPUT                        │
│                                                              │
│                   Healthy  | PD Detected                     │
│                                                              │
│                    + Confidence Score                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Pipeline Components

| Stage                  | Description                                           | Methods                                        |
| ---------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| **Input**              | Raw data collection from three modalities             | Digital tablet / smartphone camera, microphone |
| **Preprocessing**      | Image normalization, noise reduction, audio filtering | OpenCV, librosa, scipy                         |
| **Feature Extraction** | Modality-specific feature extraction                  | CNN, MFCC, geometric descriptors               |
| **Fusion**             | Combining multi-modal feature representations         | Concatenation, attention mechanisms            |
| **Classification**     | Binary or multi-class prediction                      | SVM, Random Forest, DNN, Ensemble              |
| **Output**             | Prediction with probability and explanation           | SHAP, Grad-CAM, LIME                           |

---

## Goal of the Project

The primary objective of this research project is to develop a **reliable, non-invasive, and accessible AI-powered screening tool** that can assist in the early detection of Parkinson's disease, prior to the onset of severe motor symptoms.

### Specific Objectives

- **Support early screening** by identifying neuromotor biomarkers in drawing and speech patterns that are indicative of early-stage Parkinson's disease.
- **Assist medical professionals** by providing a supplementary diagnostic tool that can be used alongside clinical evaluations, reducing diagnostic uncertainty.
- **Improve accessibility** by enabling screening through low-cost, widely available hardware (e.g., smartphones, standard microphones), making early detection feasible in resource-limited settings.
- **Advance explainability** by incorporating interpretable AI techniques to provide clinicians with transparent, actionable insights into model predictions.
- **Contribute to the research community** by creating and openly sharing benchmarked datasets, reproducible code, and documented experiments.

### Important Note

> This system is intended to serve as a **clinical decision support tool** and is **not designed to replace** the judgment of qualified medical professionals. All outputs should be interpreted by licensed neurologists or movement disorder specialists. The system does not provide a definitive medical diagnosis.
