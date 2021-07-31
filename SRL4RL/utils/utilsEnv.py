import numpy as np
import torch
import cv2
import pickle

from bullet_envs.utils import PY_MUJOCO, env_with_distractor, env_with_fpv

from SRL4RL.utils.nn_torch import numpy2pytorch, pytorch2numpy
from SRL4RL.utils.utils import createFolder
from SRL4RL import user_path


CWH2WHC = lambda x: x.transpose(1, 2, 0)
NCWH2WHC = lambda x: x.transpose(0, 2, 3, 1)[0]
tensor2image = lambda x: NCWH2WHC(pytorch2numpy(x))


def giveEnv_name(config):
    param_env = ''
    param_env += 'fpv ' if config['fpv'] else ''
    param_env += 'wallDistractor ' if config['wallDistractor'] else ''
    param_env += 'withDistractor ' if config['distractor'] else ''
    param_env += config['noise_type'] + ' ' if config['noise_type'] != 'none' else ''
    param_env += 'flickering-{} '.format(config['flickering']) if config['flickering']>0 else ''
    if 'randomExplor' in config:
        randomExplor = config['randomExplor']
    else:
        randomExplor = True
    param_env += 'randomExplor ' if randomExplor else ''
    return param_env[:-1]

def assert_args_envs(args):
    if args.env_name != 'TurtlebotMazeEnv-v0':
        assert not args.wallDistractor, 'wallDistractor with not TurtlebotMazeEnv'

    if args.distractor:
        assert args.env_name in env_with_distractor, 'distractor not implemented'

def update_args_envs(args):
    if 'Turtlebot' in args.env_name:
        args.fpv = True

    if 'Turtlebot' in args.env_name:
        args.bumpDetection = True
    else:
        args.bumpDetection = False

    if args.env_name in PY_MUJOCO:
        if 'with_reset' in args.__dict__:
            args.with_reset = True
        args.actionRepeat = 4
        if args.env_name in ['HopperBulletEnv-v0', 'Walker2DBulletEnv-v0']:
            args.actionRepeat = 2
        elif args.env_name in ['ReacherBulletEnv-v0']:
            args.actionRepeat = 1
    else:
        args.actionRepeat = 1

    args.color = True
    args.new_env_name = args.env_name[:-3] if '-v' in args.env_name else args.env_name
    args.image_size = 64

    args.n_stack = 1
    # TODO: uncomment below line
    args.n_stack = 3 if args.actionRepeat > 1 else 1
    # TODO: comment below line
    if args.actionRepeat == 1 and args.method!= 'XSRL': args.n_stack = 3

    return args


def reset_stack(obs, config):
    if config['n_stack'] > 1:
        nc = 3 if config['color'] else 1
        shape = list(obs.shape)
        if len(shape) > 3:
            shape[1] *= config['n_stack']
            observation_stack = np.zeros((shape), np.float32)
            for step_rep in range(config['n_stack']):
                observation_stack[:, step_rep * nc: (step_rep + 1) * nc] = obs
        elif len(shape) == 3:
            shape[0] *= config['n_stack']
            observation_stack = np.zeros((shape), np.float32)
            for step_rep in range(config['n_stack']):
                observation_stack[step_rep * nc : (step_rep +1) * nc] = obs
        return observation_stack


def render_env(env, image_size, fpv, camera_id, color=True, downscaling=True):
    image = env.render(mode='rgb_array', image_size=image_size,  color=color, fpv=fpv,
                       downscaling=downscaling,camera_id=camera_id)
    return image


def renderPybullet(envs, config, tensor = True):
    """Provides as much images as envs"""
    if type(envs) is list:
        obs = [env_.render(mode='rgb_array', image_size=config['image_size'], color=config['color'], fpv=config['fpv'],
                           camera_id=0) for env_ in envs]
        obs = np.array(obs).transpose(0, 3, 1, 2) / 255.
    else:
        obs = envs.render(mode='rgb_array', image_size=config['image_size'], color=config['color'], fpv=config['fpv'],
                          camera_id=0)
        obs = obs.transpose(2, 0, 1) / 255.
        if tensor:
            obs = obs[None]
    return obs


def update_video(env=None, im=None, step=0, color=True, camera_id=0, video_size=588, video=None, save_images=False,
                 fpv=None,
                 save_dir='', concatIM=None, downscaling=True):
    if save_dir == '':
        save_dir = user_path + 'Downloads/' + env.__class__.__name__ + '/'
        if save_images:
            createFolder(save_dir, '')
    if im is None:
        im = env.render(mode='rgb_array', image_size=video_size, color=color, camera_id=camera_id, fpv=fpv,
                        downscaling=downscaling)
        assert im.shape[0] == video_size, 'im.shape[0] is not in good size'
    else:
        if im.shape[0] != video_size:
            im = im.astype(np.uint8)
            im = cv2.resize(im, dsize=(video_size, video_size), interpolation=cv2.INTER_CUBIC)
    im = im[:, :, ::-1].astype(np.uint8) if color else im.astype(np.uint8)
    if type(concatIM) is np.ndarray:
        "concatIM is between 0 and 255"
        concatIM = concatIM.astype(np.uint8)
        concatIM = cv2.resize(concatIM, dsize=(video_size, video_size), interpolation=cv2.INTER_CUBIC)
        concatIM = concatIM[:, :, ::-1].astype(np.uint8) if color else concatIM.astype(np.uint8)
        new_im = np.hstack([im, concatIM])
    else:
        new_im = im

    if save_images:
        cv2.imwrite(save_dir + 'ob_%05d' % (step) + '.png', new_im)
    if video is not None:
        video.write(new_im)


def load_cifar10(file):
    """load the cifar-10 data"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        im = dict[b'data'].reshape(-1, 3, 32, 32)
        im = im.transpose(0, 3, 2, 1)
    return im


def cutout(img, n_holes, length):
    """
    Args:
        img (Tensor): Tensor image of size (C, H, W).
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
    """
    mask = np.ones((img.shape), np.float32)
    if len(img.shape) > 4:
        h = img.shape[-2]
        w = img.shape[-1]
        n_obs = img.shape[:2]
    elif len(img.shape) == 4:
        h = img.shape[-2]
        w = img.shape[-1]
        if img.shape[0] > 1:
            n_obs = img.shape[0]
        else:
            n_obs = None
    else:
        assert img.shape[-1] in [1, 3]
        h = img.shape[0]
        w = img.shape[1]
        n_obs = None

    for n in range(n_holes):
        y = np.random.randint(h, size=(n_obs))
        x = np.random.randint(w, size=(n_obs))

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        if len(img.shape) == 4:
            if img.shape[0] > 1:
                for j in range(n_obs):
                    mask[j, :, y1[j]: y2[j], x1[j]: x2[j]] = 0.
            else:
                mask[0, :, y1: y2, x1: x2] = 0.
        elif len(img.shape) > 4:
            for ep in range(n_obs[0]):
                for j in range(n_obs[1]):
                    mask[ep, j, :, y1[ep, j]: y2[ep, j], x1[ep, j]: x2[ep, j]] = 0.
        elif n_obs is None:
            mask[y1: y2, x1: x2] = 0.

    img = img * mask

    return img


def add_noise(x, noise_adder, config):
    if config['with_noise'] or config['flickering'] > 0.:
        is_tensor = False
        if torch.is_tensor(x):
            is_tensor = True
            device = x.device
            x = pytorch2numpy(x)
        if config['with_noise']:
            out = noise_adder(observation=x)
        else:
            out = x
        if config['flickering'] > 0.:
            if len(x.shape) > 3:
                flickerings = np.random.uniform(0, 1, size=(out.shape[0])) < config['flickering']
                out[flickerings] = out[flickerings] * 0
            else:
                if np.random.uniform(0, 1) < config['flickering']:
                    out = out * 0
        if is_tensor:
            out = numpy2pytorch(out, differentiable=False, device=device)
    else:
        out = x
    return out


