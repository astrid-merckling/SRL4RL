import torch
import numpy as np
import os, json
import cv2
from datetime import datetime
import argparse
import shutil

from bullet_envs.utils import env_with_goals, PY_MUJOCO

from SRL4RL.utils.utilsEnv import update_video, render_env
from SRL4RL.rl.utils.env_utils import make_env
from SRL4RL.rl.modules.agent_utils import process_inputs
from SRL4RL.utils.utils import str2bool, loadConfig, encoder_methods, give_name, saveConfig, createFolder
from SRL4RL.utils.nn_torch import set_seeds

RANDOM = False
seperate_csv = False

image_size = 588  # 588
color = True

class EmptyArgs: pass


def eval_agent(args, env, o_mean, o_std, g_mean, g_std, actor_network, runner, video_path="", image_path=''):
    numSteps = int(args.max_episode_steps // args.actionRepeat)
    mean_rewardProgress = 0
    if args.renders and args.env_name in PY_MUJOCO:
        env.render(mode='human')
    if args.env_name in ['TurtlebotEnv-v0', 'TurtlebotMazeEnv-v0']:
        "map view of the environment"
        camera_id_eval = 1
    else:
        "the default camera"
        camera_id_eval = -1 if args.highRes else 0

    camera_id = -1 if args.highRes else 0
    g, num_steps = 0, 0
    total_step = np.zeros((args.n_eval_traj))
    rewards = np.zeros((args.n_eval_traj, numSteps))
    rewardProgress = np.zeros((args.n_eval_traj))
    if video_path:
        if 'Turtlebot' in args.env_name:
            fps = 5
        elif args.actionRepeat > 1:
            fps = 40 // args.actionRepeat
        else:
            fps = 4
        im_width = image_size * 2 if args.method in encoder_methods else image_size
        video_out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps,
                                    frameSize=(im_width, image_size)) if args.color else \
            cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), fps=fps, frameSize=(im_width, image_size),
                            isColor=0)

    for ntraj in range(args.n_eval_traj):
        print('traj: {}'.format(ntraj + 1))

        observation = env.reset()
        if args.with_goal:
            state = observation['observation']
            g = observation['desired_goal']
        else:
            state = observation

        if video_path:
            "reset video"
            if args.method in encoder_methods:
                update_video(env, color=args.color, video_size=image_size,
                             video=video_out, fpv=args.fpv, camera_id=camera_id, concatIM=runner.last_input * 255,downscaling=not args.highRes)
            else:
                update_video(env, im=None, color=args.color, video_size=image_size,
                             video=video_out, fpv=args.fpv, camera_id=camera_id,downscaling=not args.highRes)
        if image_path:
            im_high_render = render_env(env, 588, False, camera_id_eval, args.color,downscaling=not args.highRes)
            cv2.imwrite(image_path + 'ob_{:05d}'.format(num_steps) + '.png', im_high_render[:, :, ::-1].astype(np.uint8))

        for step in range(numSteps):
            num_steps += 1

            if not RANDOM:
                input_tensor = process_inputs(state, g, o_mean, o_std, g_mean, g_std, args.__dict__)
            else:
                input_tensor = None
            with torch.no_grad():
                pi = runner.forward(actor_network, input_tensor, eval=True, random = RANDOM)

            observation_new, reward, done, info = env.step(pi)
            if video_path:
                "update video"
                if args.method in encoder_methods:
                    update_video(env, color=args.color, video_size=image_size,
                                 video=video_out, fpv=args.fpv, camera_id=camera_id, concatIM=runner.last_input * 255,downscaling=not args.highRes)
                else:
                    update_video(env, im=None, color=args.color, video_size=image_size,
                                 video=video_out, fpv=args.fpv, camera_id=camera_id,downscaling=not args.highRes)
            if image_path:
                im_high_render = render_env(env, 588, False, camera_id_eval, args.color, downscaling=not args.highRes)
                cv2.imwrite(image_path + 'ob_{:05d}'.format(num_steps) + '.png',
                            im_high_render[:, :, ::-1].astype(np.uint8))

            assert step < env.maxSteps, "wrong max_episode_steps"
            if args.env_name in env_with_goals:
                if args.with_goal:
                    state_new = observation_new['observation']
                    g = observation_new['desired_goal']
                else:
                    state_new = observation_new
                    info['is_success'] = reward + 1
                if ((info['is_success'] == 1.) or (step == env.maxSteps)):
                    rewards[ntraj, step] = info['is_success']
                    num_steps += 1
                    if info['is_success'] == 0.:
                        print('\ntraj {} fails, elapsed_steps {}'.format(ntraj + 1, step + 1))
                    break
            else:
                state_new = observation_new
                rewards[ntraj, step] = reward
                if 'Pendulum' in args.env_name and video_path:
                    if np.mean(rewards[ntraj, step + 1 - fps * 10: step + 1]) > 3.9:
                        rewards[ntraj, step + 1:] = rewards[ntraj, step]
                        break
            if video_path: print('step [{}] reward {}'.format(step, reward))
            state = state_new

        if args.env_name in env_with_goals:
            if info['is_success'] != 0.:
                total_step[ntraj] = step + 1
        else:
            total_step[ntraj] = step + 1
        if args.env_name in ['AntBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0',
                             'Walker2DBulletEnv-v0']:
            rewardProgress[ntraj] = env.rewardProgress

    mean_rewards = np.mean(np.sum(rewards, axis=1))
    if args.env_name in ['AntBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0']:
        mean_rewardProgress = np.mean(rewardProgress)
    average_steps = args.max_episode_steps if np.isnan(np.mean(total_step)) else np.mean(total_step)
    if video_path:
        "Release everything if job is finished"
        video_out.release()
        cv2.destroyAllWindows()
        if args.env_name in env_with_goals:
            strR = '%03d' % int(mean_rewards * 100)
            strR = strR[0] + ',' + strR[1:]
        else:
            strR = 'R%04d' % int(mean_rewards)
        if args.env_name in ['AntBulletEnv-v0', 'HalfCheetahBulletEnv-v0', 'HopperBulletEnv-v0',
                             'Walker2DBulletEnv-v0']:
            strR += '-RP%04d' % int(mean_rewardProgress)
        destination = video_path[:-4] + '-' + strR + '.mp4'
        print('destination', destination)
        shutil.move(video_path, destination)

    return mean_rewards, mean_rewardProgress, average_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str)
    parser.add_argument('--save_video', type=str2bool, default=False, help='Record video from evaluation trajectories')
    parser.add_argument('--save_image', type=str2bool, default=False, help='Record images from evaluation trajectories')
    parser.add_argument('--model_type', type=str, default="model_best", choices=(['model_last', 'model_best']), help='Whether to load the policy with best average return, or the last saved policy')
    parser.add_argument('--demo_length', type=int, default=2, help='The demo length')
    parser.add_argument('--n_eval_traj', type=int, default=0, help='The number of trajectories to compute the average episode returns')
    parser.add_argument('--renders', type=str2bool, default=False, help='Tune entropy')
    parser.add_argument('--highRes', type=str2bool, default=True, help='Record high-resolution images, if True, do not downscale images')
    parser.add_argument('--cuda', type=str2bool, default=True, help='If False, do not use cuda if available')
    args_init = parser.parse_args()

    if args_init.n_eval_traj > 0: assert not (args_init.save_video or args_init.save_image)

    demo_length = args_init.demo_length
    print('\nproj_path: ', args_init.dir)
    args_init.dir = args_init.dir[:-1] if args_init.dir[-1] == '/' else args_init.dir
    all_proj_opt, file = os.path.split(args_init.dir)
    try:
        config = loadConfig(args_init.dir)
    except:
        print('\nNeed remove folder: %s\n' % args_init.dir)
        exit()

    env_params = config['env_params']

    args = EmptyArgs()
    args.__dict__.update(config)
    "change args with args_init"
    args.renders = args_init.renders
    args.dir = args_init.dir
    args.highRes = args_init.highRes

    args.seed = datetime.now().microsecond
    print('\nSeed is : \n', args.seed)
    "IMPORTANT TO USE FOR CUDA MEMORY"
    set_seeds(args.seed)

    """
    Load the controller
    """
    o_mean, o_std, g_mean, g_std = 0, 0, 0, 0
    "Load RL model"
    try:
        o_mean, o_std, g_mean, g_std, actor_network, _ = torch.load(
            args_init.dir + '/{}.pt'.format(args_init.model_type), map_location=lambda storage, loc: storage)
    except:
        assert args_init.model_type == 'model_best', 'model_last.pt missing!'
        print('\nNot model.pt saved')
        exit()

    actor_network.eval()


    if args.env_name in env_with_goals:
        # because RL do not need to see target, it sees the target's position
        args.display_target = False if args_init.n_eval_traj > 0 else True
        if 'TurtlebotMazeEnv' in args.env_name:
            args.n_eval_traj = 5
        elif 'ReacherBulletEnv' in args.env_name:
            args.n_eval_traj = 20
        saved_steps_per_episode = args.max_episode_steps
    elif 'Pendulum' in args.env_name:
        args.n_eval_traj = 5
        saved_steps_per_episode = 100 * 4
    else:
        args.n_eval_traj = 1
        saved_steps_per_episode = args.max_episode_steps


    if args_init.save_video or args_init.save_image or args_init.renders:
        assert args_init.n_eval_traj == 0
        # Change max_episode_steps to define the Average step
        args.max_episode_steps = saved_steps_per_episode
    elif args_init.n_eval_traj > 0:
        args.n_eval_traj = args_init.n_eval_traj



    """
    Create the environment with the SRL model wrapper
    """
    args.srl_path = args_init.dir
    args.demo = True
    # args.distractor = True
    # args.noise_type = 'noisyObs'
    env, _, runner = make_env(args.__dict__)
    env.seed(args.seed)

    # Create video folder
    if args_init.save_video:
        video_path = args.dir + '/piEval-best-E%s.mp4' % config[
            'best_elapsed_epochs'] if args_init.model_type == "model_best" else args.dir + '/piEval-last-E%s.mp4' % config[
            'last_elapsed_epochs']

    else:
        video_path = ""
    if args_init.save_image:
        image_path = args.dir + '/piEval-best-E%s/' % config[
            'best_elapsed_epochs'] if args_init.model_type == "model_best" else args.dir + '/piEval-last-E%s/' % config[
            'last_elapsed_epochs']
        createFolder(image_path, image_path + " already exist")
    else:
        image_path = ""

    # Create recorder:
    model_name = config['method']
    model_name = give_name(config)
    prefix = 'best_' if args_init.model_type == "model_best" else ''

    mean_rewards, mean_rewardProgress, average_steps = eval_agent(args, env, o_mean, o_std, g_mean, g_std,
                                                                  actor_network, runner,
                                                                  video_path=video_path,image_path=image_path)
    if args_init.n_eval_traj > 0:
        print('the average total reward is: {}, the average total steps is: {}'.format(mean_rewards, average_steps))
        config[prefix + 'avg-reward'] = mean_rewards
        config[prefix + 'avg-progress'] = mean_rewardProgress if mean_rewardProgress != 0 else ''
        config[prefix + 'avg-steps'] = average_steps
        saveConfig(config, save_dir=args.dir)
        with open(os.path.join(args.dir,'config.json'), 'w') as outfile:
            json.dump(config, outfile)
