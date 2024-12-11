# DeRelayLearning
Decentralizde Relay Learning: decentralized AI model training system using Blockchain


# DeRelayL Simulation

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Tengfei-Ma13206/DeRelayL_Simulation.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DeRelayL_Simulation
   ```
3. Create a new conda environment:
   ```bash
   conda create -n DeRelayL python=3.10
   ```
4. Activate the conda environment:
   ```bash
   conda activate DeRelayL
   ```
5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the Simulation

1. Execute the following scripts (ensure that your current directory is still `DeRelayL_Simulation`):
   ```bash
   python simulation/simulation_version.py
   python simulation/simulation_coin.py
   ```

2. After running the scripts, you will find two generated images in the directory `DeRelayL_Simulation/pic`.  
   These images should be identical to the following files:
   - `DeRelayL_Simulation/simulation/pic-coin/coins_per_participant_over_rounds.pdf`
   - `DeRelayL_Simulation/simulation/pic-version/version_distribution_per_participant_over_rounds.pdf`

Enjoy exploring the simulation results!


## Key Principles

In the era of large-scale models, fully utilizing personal data—especially data unavailable on the internet—requires a carefully designed incentive mechanism. This simulation focuses on the following key principles:

1. **Incentivizing Contributions**: Designing mechanisms that encourage individuals to willingly contribute their data and computational power to large-scale models, with the promise of improved model performance as a tangible benefit.

2. **Fundamental Assumptions for Sustainability**:
   - The primary goal for all participants is to achieve better-performing models.
   - The majority of participants act honestly. Dishonest behaviors are naturally filtered out, as the majority will ignore and not acknowledge contributions from dishonest participants.

## Simulation Focus

This simulation demonstrates the core concepts of DeRelayL by:
- Highlighting how individual contributions lead to collective model improvement.
- Showcasing the mechanisms that ensure dishonest participants are excluded from the system.
- Emphasizing the transparency of model versioning and reward distribution to maintain trust and sustainability.

## Conclusion

DeRelayL provides a sustainable framework for decentralized collaborative model training, leveraging individual contributions effectively while ensuring system integrity. By focusing on incentivizing honest participation and filtering out dishonest behavior, DeRelayL creates a robust ecosystem for achieving high-performing models collaboratively.
