from argparse import ArgumentParser
import pickle

#import aicrowd_gym
import gym as aicrowd_gym
import minerl

from openai_vpt.agent import MineRLAgent

import videoio
import os
import datetime
import logging

# Configure base logging

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        console # Always log to console
    ]
)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)


def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=False, video=False, video_dir="videos"):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    logging.info(f"Running agent on {env.spec.id} for {n_episodes} episodes.")
    run_title = f"{env.spec.id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" 
    vid_run_dir = os.path.join(video_dir, run_title)
    log_run_dir = os.path.join(log_dir, run_title)
    os.makedirs(log_run_dir, exist_ok=True)

    
    for _ in range(n_episodes):
        obs = env.reset()
        if video:
            os.makedirs(vid_run_dir, exist_ok=True)
            # Specify subfolder with date and time
            video_path = os.path.join(vid_run_dir, f"{env.spec.id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
            recorder = videoio.VideoWriter(video_path, resolution=(640, 360), fps=20)
            recorder.write(obs["pov"])
        
        log_run_path = os.path.join(log_run_dir, f"{run_title}.log")
        file_handler = logging.FileHandler(log_run_path)
        file_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(file_handler)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.info(f"Starting run.")

        for step in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            logging.debug(f"Step {step}: {action}")
            action.setdefault("ESC", 0)  # Ensure ESC is present        

            if action.get('ESC'):
                # Agent has taken escape action
                logging.info(f'Value of ESC is {action["ESC"]}, escaping at step {step}!')
                break
            
            
            # If model isn't escaping, perform a step
            obs, _, done, _ = env.step(action)
            if video:
                recorder.write(obs["pov"])
            if show:
                env.render()
            if done:
                logging.info(f"Done from env after {step} steps!")
                break
        logging.info(f"Finished run.") 
        if video:
            recorder.close()
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--video", action="store_true", help="Record a video of the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)
