<div align="center">

# InternVLA-A1: Unifying Understanding, Generation, and Action for Robotic Manipulation​

</div>

---

InternVLA-A1 is an end-to-end vision–language–action (VLA) framework unifing understanding, generation ,and action for robotic manipulation. It leverages predictive imagination of task evolution to guide execution, enabling enhanced manipulation in highly dynamic environments. 

## :fire: Highlights <a name="high"></a>
<img width="1000" alt="seer" src="assets/internvla_a1_framework.jpg">

- **Novel Model Archituecture**: A Mixture-of-Transformers architecture for unified understanding, generation, and action.
- **Hybrid Synthetic-Real Data Corpus**: A hybrid synthetic-real manipulation dataset integrating 5 heterogeneous robots, 15 skills, and 200+ scenes, emphasizing multi-robot collaboration under dynamic scenarios.
- **Impressive Real-World performance**: InternVLA-A1 demonstrates strong effectiveness and generalization in highly dynamic scenarios involving dynamic grasping of conveyor belts and multi-robot collaboration.

## 🤖 Real-World Robot Demonstrations

### **Package grabbing and flipping in conveyor belt**
<div align="center">
    <video src="https://github.com/user-attachments/assets/07ca1356-9956-4acb-a67e-1d2cf37c8587"
         controls autoplay muted playsinline loop width="720"></video>
  <p><em>In the conveyor belt scenario, faced with dynamic packages of various shapes, the model can track and accurately predict their movement trajectories in real time, and still achieve stable grasping at high speeds; it can also adaptively complete package flipping and express information identification based on the state of the delivery note.</em></p>
</div>


### **Rapid Adaptation**
<div align="center">
    <video src="https://github.com/user-attachments/assets/3048e21a-ebdb-4f4e-a96e-f31597412992"
         controls autoplay muted playsinline loop width="720"></video>
  <p><em>In the rotating hot pot table scenario, faced with a variety of ingredients running at high speed, the model can quickly identify and locate targets according to the needs of the finished dishes, and accurately complete the gripping, fully demonstrating its high adaptability to complex environments and task-oriented intelligence.</em></p>
</div>

### **Multi-Robot Collaboration**
<div align="center">
      <video src="https://github.com/user-attachments/assets/4435ced0-2cc2-4999-93f4-5ec1e2fbab8f"
         controls autoplay muted playsinline loop width="720"></video>
  <p><em>In multi-robot collaboration scenarios, relying on dynamic visual capture technology and the forward-looking prediction algorithm in the "task imagination module", the system can quickly adjust its movements and achieve high-precision real-time interaction.</em></p>
</div>


### **Long-horizon task**
<div align="center">
    <video src="https://github.com/user-attachments/assets/849e922f-f41c-4b1a-8b82-4e189fc41d43"
         controls autoplay muted playsinline loop width="720"></video>
  <p><em>In terms of long-horizon task such as microwave/oven operation. </em></p>
</div>


## 🚀 Quick Start

### **Prerequisites**
- Python ≥ 3.10
- torch ≥ 2.6.0
- CUDA ≥ 12.4

### **Installation**
```bash
# Clone repository
git clone https://github.com/InternRobotics/InternVLA-A1.git

# Create environment
conda create -f internvla_a1 python==3.10
conda activate internvla_a1

# Install dependencies
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124

# install other requirements
pip install -r requirements.txt

pip install numpy==1.26.4
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Lerobot](https://github.com/huggingface/lerobot)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [COSMOS](https://github.com/nvidia-cosmos)
- [Any4lerobot](https://github.com/Tavish9/any4lerobot/)
- [VAR](https://github.com/FoundationVision/VAR)
