from run_agent_logged import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS

def main():
    run_agent_main(
        model="data/VPT-models/foundation-model-1x.model",
        weights="train/MineRLBasaltFindCave.weights",
        env="MineRLBasaltFindCave-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=EVAL_MAX_STEPS,
        show=False,
        video=True,
        video_dir="videos"
    )

if __name__ == "__main__":
    main()