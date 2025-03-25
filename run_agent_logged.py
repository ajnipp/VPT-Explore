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

logging.basicConfig(level=logging.INFO)

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
    
    for _ in range(n_episodes):
        obs = env.reset()
        if video:
            os.makedirs(video_dir, exist_ok=True)
            # Specify subfolder with date and time
            video_path = os.path.join(video_dir, f"{env.spec.id}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
            recorder = videoio.VideoWriter(video_path, resolution=(640, 360), fps=20)
            recorder.write(obs["pov"])

        for step in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
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
