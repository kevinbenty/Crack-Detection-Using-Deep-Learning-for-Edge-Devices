# Paper Summary: Image-based Non-invasive Bridge Crack Detection Model for Edge Devices

**RAICS 2025**

## Authors
- Athul Joe Palliparambil
- Kevin Benty
- Alan Cherian
- Rajeev Rajan

**Institution**: Government Engineering College, Idukki  
**University**: APJ Abdul Kalam Technological University, Thiruvananthapuram, India

**Contact**: rajeev@gecidukki.ac.in

---

## Abstract

Crack detection is a critical task for ensuring the structural integrity and safety of various infrastructures. This study evaluates the efficacy of three deep learning architectures—a custom 3-layer CNN, MobileNetV2, and Vision Transformer (ViT)—for automated multi-class crack detection on edge devices. The study demonstrates that these models can effectively capture complex patterns and important features, improving performance in distinguishing crack severity. The research provides actionable strategies for future directions in automated structural health monitoring.

---

## Problem Statement

Cracks in bridge and concrete structures are a widespread problem that regularly lead to:
- Compromised structural integrity
- Functional failures
- Safety hazards

Current challenges include:
- **Unbalanced datasets**: Difficulty in obtaining balanced samples across crack severity levels
- **Subtle differences**: Complex patterns differentiating between crack types
- **Manual inspection limitations**: Labor-intensive, time-consuming, and often unsafe

---

## Motivation

While recent research has focused on **binary crack detection** (crack vs. no crack), there is a significant gap in **multi-class classification** of crack severity. Additionally, deployment on **edge devices** (resource-constrained systems like Raspberry Pi) is underexplored.

This project addresses:
1. Multi-class crack severity classification
2. Edge device optimization and deployment
3. Comprehensive comparison of CNN, MobileNetV2, and ViT architectures

---

## Methodology

### Dataset
- **Total Images**: 40,030
  - Training: 20,020
  - Validation: 7,230
  - Testing: 12,780
- **Source**: Kaggle Concrete Crack Images Dataset + Real-world images
- **Classes** (4):
  - No Crack
  - Light Crack
  - Medium Crack
  - Heavy Crack
- **Preprocessing**:
  - Resized to 256×256 pixels
  - Normalized for efficient training
  - Data augmentation: rotation, flipping, zooming, brightness adjustment

### Models Evaluated

#### 1. Custom 3-Layer CNN
A lightweight baseline model consisting of:
- 3 convolutional layers (32, 64, 128 filters)
- Max pooling for dimensionality reduction
- Dense layers (512 neurons) with softmax output
- ~40K parameters

**Advantages**:
- Minimal computational requirements
- Fast inference on edge devices
- Easy to understand and modify

**Disadvantages**:
- Limited feature extraction capability
- Lower accuracy compared to advanced architectures

#### 2. MobileNetV2
Lightweight architecture designed for mobile/embedded applications:
- **Key Innovation**: Inverted residual bottleneck blocks
- **Structure**:
  - Depthwise separable convolutions (reduce parameters)
  - 1×1 convolutions for expansion/compression
  - Global average pooling
- **Efficiency**: Reduced computational cost while maintaining accuracy

**Advantages**:
- Efficient parameter usage
- Good accuracy-to-efficiency ratio
- Practical for edge deployment
- Successfully deployed on Raspberry Pi 4B

**Disadvantages**:
- Lower accuracy than ViT
- Limited global context understanding

#### 3. Vision Transformer (ViT)
State-of-the-art transformer-based architecture:
- **Approach**: Applies NLP transformer directly to image classification
- **Process**:
  1. Splits image into 16×16 patches
  2. Flattens and linearly projects each patch
  3. Adds positional embeddings for spatial information
  4. Passes through 12 transformer encoder layers
  5. Uses [CLS] token output for classification
- **Mechanism**: Multi-head self-attention captures global relationships

**Advantages**:
- Highest accuracy (90.03%)
- Captures global context effectively
- Excellent for fine-grained classification

**Disadvantages**:
- Large number of parameters (~86M)
- High computational requirements
- Impractical for resource-constrained edge devices
- Longer inference time

### Training Framework
- **Environment**: Google Colab TPU v4
- **Optimizer**: Adam
- **Loss Function**: Categorical cross-entropy
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with validation-based scheduler)
- **Epochs**: Up to 30
- **Regularization**:
  - Early stopping (patience: 5 epochs)
  - Model checkpoints based on validation accuracy

### Evaluation Metrics
- **Accuracy**: Overall correct classifications
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrices**: Per-class classification analysis

---

## Results

### Overall Performance Comparison

| Model | Accuracy | F1-Score | Model Size | Best Use Case |
|-------|----------|----------|------------|---------------|
| CNN | 84.88% | 0.78 | 1.2 MB | Minimal resources |
| MobileNetV2 | 86.99% | 0.79 | 18.7 MB | Edge deployment (balanced) |
| ViT | 90.03% | 0.77 | 350+ MB | Cloud/Server |

### Per-Class Performance Analysis

#### 3-Layer CNN
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Heavy Crack | 0.96 | 0.78 | 0.86 |
| Light Crack | 0.72 | 0.48 | 0.57 |
| Medium Crack | 0.57 | 0.90 | 0.70 |
| No Crack | 0.99 | 0.99 | 0.99 |
| **Average** | **0.81** | **0.79** | **0.78** |

#### MobileNetV2
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Heavy Crack | 0.91 | 0.87 | 0.90 |
| Light Crack | 0.63 | 0.52 | 0.58 |
| Medium Crack | 0.65 | 0.71 | 0.69 |
| No Crack | 0.99 | 0.99 | 0.99 |
| **Average** | **0.76** | **0.80** | **0.79** |

#### Vision Transformer (ViT)
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Heavy Crack | 0.68 | 0.86 | 0.76 |
| Light Crack | 0.49 | 0.62 | 0.55 |
| Medium Crack | 0.87 | 0.72 | 0.79 |
| No Crack | 1.00 | 1.00 | 1.00 |
| **Average** | **0.76** | **0.80** | **0.77** |

### Key Findings

1. **Accuracy Hierarchy**: ViT > MobileNetV2 > CNN
   - ViT achieves superior accuracy through global context via self-attention
   - MobileNetV2 offers better balance through efficient architecture
   - CNN sufficient for resource-critical deployments

2. **Class-Specific Performance**:
   - "No Crack" class: Excellent across all models (>0.99)
   - "Heavy Crack": Good performance (0.86-0.90 F1)
   - "Medium Crack": Moderate performance (0.69-0.79 F1)
   - "Light Crack": Most challenging class (0.55-0.58 F1)

3. **Feature Extraction Progression**:
   - Early layers: Basic features (edges, textures)
   - Deeper layers: Abstract, high-level patterns
   - ViT: Captures global context from early stages

4. **Data Augmentation Impact**:
   - Critical for handling class imbalance
   - Improved model generalization
   - Reduced overfitting

5. **Model Size vs. Accuracy Trade-off**:
   ```
   CNN (1.2 MB) ← Smallest
   MobileNetV2 (18.7 MB) ← Optimal balance
   ViT (350+ MB) ← Highest accuracy
   ```

---

## Edge Device Deployment

### Raspberry Pi 4B Implementation

**Hardware Specs**:
- 4GB RAM
- ARM Cortex-A72 CPU (1.5 GHz)
- No integrated GPU

**Optimization Techniques**:

1. **Quantization**: INT8 precision
   - Reduced MobileNetV2 from ~70 MB to 18.7 MB
   - Minimal accuracy loss (< 1%)
   - 95% size reduction

2. **Model Conversion**: TensorFlow → TensorFlow Lite
   - Format optimized for mobile/embedded
   - Reduced inference overhead

3. **Inference Optimization**:
   - Headless operation (no GUI)
   - Background task prioritization
   - Lightweight dependencies

**Performance Metrics**:
- **MobileNetV2 on Raspberry Pi 4B**:
  - Inference time: ~200-500 ms per image
  - Real-time FPS: 3-5 FPS
  - Model size: 18.7 MB
  - RAM usage: Minimal

- **With Google Coral TPU Accelerator**:
  - Inference time: ~30-50 ms per image
  - Real-time FPS: 30+ FPS
  - Significant speedup with hardware acceleration

**Viability Assessment**:
- ✅ Successfully deployed MobileNetV2 on Raspberry Pi 4B
- ✅ Acceptable inference time for real-world inspections
- ✅ Low power consumption suitable for portable systems
- ✅ Can be further optimized with TPU accelerators

---

## Comparative Analysis

### CNN vs. MobileNetV2 vs. ViT

**Complexity vs. Performance**:
- CNN: Simple architecture, moderate performance
- MobileNetV2: Balanced approach, practical efficiency
- ViT: Complex architecture, optimal accuracy

**Computational Efficiency**:
- MobileNetV2 uses depthwise separable convolutions (efficient)
- ViT requires matrix multiplications over all image patches (computationally expensive)
- CNN: Minimal overhead but limited expressiveness

**Feature Extraction Strategy**:
- CNN: Local spatial relationships
- MobileNetV2: Local + efficient feature refinement
- ViT: Global context through attention mechanisms

---

## Conclusions

### Key Takeaways

1. **Vision Transformer achieves highest accuracy** (90.03%) but is impractical for edge devices due to computational complexity

2. **MobileNetV2 emerges as optimal solution** for edge deployment:
   - 86.99% accuracy (only 3% less than ViT)
   - 18.7 MB size (50x smaller than ViT)
   - Successfully deployed on Raspberry Pi 4B
   - Practical for real-world on-site inspections

3. **3-Layer CNN** demonstrates that simple architectures can achieve acceptable performance (84.88%) for resource-critical scenarios

4. **Multi-class classification** more challenging than binary detection, especially for light cracks

5. **Data augmentation and class balancing** crucial for improved generalization

6. **Edge computing viable** for structural monitoring with appropriate optimizations

---

## Future Directions

The paper proposes actionable strategies for future research:

1. **Improved Accuracy Without Increased Complexity**:
   - Incorporate lightweight attention mechanisms
   - Hybrid CNN-Transformer architectures
   - Depthwise separable attention

2. **Advanced Data Augmentation**:
   - Domain-specific augmentation for crack images
   - Synthetic data generation (GANs)
   - Few-shot learning for rare crack types

3. **Model Optimization**:
   - Pruning (remove unnecessary parameters)
   - Quantization (INT4, binary networks)
   - Knowledge distillation
   - Neural architecture search (NAS)

4. **Edge Device Enhancements**:
   - Hardware accelerators (TPU, NPU)
   - Distributed inference
   - Online learning capabilities

5. **Real-world Deployment**:
   - Integration with IoT sensors
   - Multi-modal analysis (thermal, acoustic)
   - Automated reporting systems
   - Large-scale infrastructure monitoring

6. **Dataset Improvements**:
   - More diverse real-world images
   - Different concrete types and environmental conditions
   - Time-series crack progression data

---

## Practical Applications

This research enables:

1. **Automated Bridge Inspection**: Replace manual inspections with AI-powered systems
2. **Early Crack Detection**: Prevent catastrophic failures through timely intervention
3. **Maintenance Scheduling**: Prioritize repairs based on severity classification
4. **Remote Monitoring**: Deploy systems on portable edge devices for on-site analysis
5. **Large-scale Assessment**: Efficiently evaluate numerous structures
6. **Cost Reduction**: Reduce inspection labor and equipment costs
7. **Safety Improvement**: Improve public safety through systematic monitoring

---

## Research Impact

This work contributes to:
- **Computer Vision**: Architectural comparison for image classification
- **Edge Computing**: Practical deployment of deep learning on resource-constrained devices
- **Civil Engineering**: Automated structural health monitoring
- **Structural Engineering**: Data-driven maintenance strategies

---

## Reproducibility

All code, configurations, and training procedures are provided in the GitHub repository to ensure:
- Full reproducibility of results
- Transparency in methodology
- Community contribution and improvement
- Academic integrity

---

## References

[1] S. Dorafshan, R. J. Thomas, and M. Maguire, "Comparison of deep convolutional neural networks and edge detectors for image-based crack detection in concrete," Constr. Build. Mater., 2018.

[2] R. Ritzy, et al., "A review on non-invasive image-based deep learning paradigms for crack detection," in ICFCR, 2024.

[3] F. Moreh, et al., "Deep neural networks for crack detection inside structures," Sci. Rep., vol. 14, 2024.

[4] S. Jency and G. R. Kumar, "DeepCrack: A deep learning approach for image-based crack prediction using MobileNet," in IDCIoT, 2024.

---

## How to Cite This Paper

```bibtex
@inproceedings{palliparambil2025crack,
  title={Image-based Non-invasive Bridge Crack Detection Model for Edge Devices},
  author={Palliparambil, Athul Joe and Benty, Kevin and Cherian, Alan and Rajan, Rajeev},
  booktitle={RAICS 2025},
  year={2025},
  organization={APJ Abdul Kalam Technological University}
}
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Status**: Published at RAICS 2025
