import gc
import os

import cv2
import numpy as np
import torch

from SRL4RL import SRL4RL_path
from SRL4RL.rl.utils.runner import StateRunner
from SRL4RL.utils.nn_torch import numpy2pytorch, pytorch2numpy, save_model
from SRL4RL.utils.utils import loadPickle
from SRL4RL.utils.utilsEnv import CWH2WHC, tensor2image, update_video
from SRL4RL.utils.utilsPlot import plot_xHat, plotEmbedding

np2torch = lambda x, device="cpu": numpy2pytorch(x, differentiable=False, device=device)


def decoder_last_layer(x):
    return torch.sigmoid(x)


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def AE_nextObsEval(
    encoder,
    decoder,
    config,
    save_dir,
    gradientStep=None,
    saved_step=None,
    suffix="last",
    debug=False,
):
    device = torch.device(config["device"])
    evaluate = suffix == "evaluate"
    actionRepeat = config["actionRepeat"]

    datasetEval_path = "testDatasets/testDataset_{}".format(config["new_env_name"])
    if actionRepeat > 1:
        datasetEval_path += "_noRepeatAction"
    elif config["distractor"]:
        datasetEval_path += "_withDistractor"
    datasetEval_path += ".pkl"

    datasetEval_path = os.path.join(SRL4RL_path, datasetEval_path)
    dataset = loadPickle(datasetEval_path)
    observations, measures = dataset["observations"], dataset["measures"]
    if debug:
        last_index = actionRepeat * 200
        observations, measures = observations[:-last_index], measures[:-last_index]

    "observations[1:][actionRepeat:] -> remove reset obs and first actionRepeat time steps"
    observations = observations[1:][actionRepeat:]
    measures = measures[1:][actionRepeat:][::actionRepeat]
    "force the Garbage Collector to release unreferenced memory"
    del dataset
    gc.collect()
    loss_log = 0

    print("AE_nextObsEval (predicting obs with PIeval_dataset) ......")
    "+1 in order AE has nextObs in input"
    eval_steps = None
    if config["new_env_name"] == "TurtlebotMazeEnv":
        xHat_nextObsEval_step = 84
        eval_steps = [87, 88, 101, 115, 117, 439, 440]
    elif config["new_env_name"] == "HalfCheetahBulletEnv":
        xHat_nextObsEval_step = 119
    elif config["new_env_name"] == "InvertedPendulumSwingupBulletEnv":
        xHat_nextObsEval_step = 45
    elif config["new_env_name"] == "ReacherBulletEnv":
        xHat_nextObsEval_step = 42
        eval_steps = [14, 25, 396]

    if config["n_stack"] == 1 and config["actionRepeat"] > 1:
        "when uncomment on # args.n_stack = 3 if args.actionRepeat > 1 else args.n_stack"
        xHat_nextObsEval_step -= 1

    video_path = os.path.join(save_dir, "piEval_%s.mp4" % suffix)
    if config["new_env_name"] == "TurtlebotMazeEnv":
        fps = 5
    elif actionRepeat > 1:
        fps = 20 // actionRepeat
    else:
        fps = 5

    video_out = (
        cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(int(588 * 2), 588),
        )
        if config["color"]
        else cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps=fps,
            frameSize=(int(588 * 2), 588),
            isColor=0,
        )
    )

    "init state with obs without noise"
    if config["n_stack"] > 1:
        nc = 3
        obs_shape = list(observations[0].shape)
        obs_shape[0] *= config["n_stack"]
        observation = np.zeros((obs_shape), np.float32)
        if config["n_stack"] > 1 and config["actionRepeat"] == 1:
            "when uncomment # if args.env_name == 'TurtlebotMazeEnv-v0': args.n_stack = 3"
            for step_rep in range(config["n_stack"]):
                observation[step_rep * nc : (step_rep + 1) * nc] = observations[0]

    loss_fn = lambda x, y: torch.nn.MSELoss(reduction="sum")(x, y) / (
        x.shape[0] * config["n_stack"]
    )

    "reset var"
    elapsed_steps = 0
    len_traj = (len(observations)) // config["actionRepeat"]
    assert len_traj == len(measures), "wrong division in len_traj"
    all_states = np.zeros([len_traj, config["state_dim"]])
    step_rep = 0
    for step, obs in enumerate(observations):
        ## Make a step
        "update obs"
        if config["n_stack"] > 1 and config["actionRepeat"] == 1:
            "when uncomment # if args.env_name == 'TurtlebotMazeEnv-v0': args.n_stack = 3"
            observation[-2 * nc :] = observation[: 2 * nc]
            observation[:nc] = obs
        elif config["n_stack"] > 1:
            if (step_rep + 1) > (config["actionRepeat"] - config["n_stack"]):
                observation[(step_rep - 1) * nc : ((step_rep - 1) + 1) * nc] = obs
        elif (step_rep + 1) == config["actionRepeat"]:
            observation = obs
        step_rep += 1
        if (step + 1) % config["actionRepeat"] == 0:
            step_rep = 0

            "decode observation"
            with torch.no_grad():
                stateExpl = encoder(np2torch(observation, device).unsqueeze(0))
                xHat = decoder_last_layer(decoder(stateExpl))
                "update loss"
                loss_log += pytorch2numpy(
                    loss_fn(xHat, np2torch(observation, device).unsqueeze(0))
                )

            "update video"
            update_video(
                im=255 * CWH2WHC(observation[-3:, :, :]),
                color=config["color"],
                video_size=588,
                video=video_out,
                fpv=config["fpv"],
                concatIM=255 * tensor2image(xHat[:, -3:, :, :]),
            )

            if type(eval_steps) is list:
                saveIm = elapsed_steps in [xHat_nextObsEval_step] + eval_steps
                name = "xHat_nextObsEval{}".format(step)
            else:
                saveIm = elapsed_steps == xHat_nextObsEval_step
                name = "xHat_nextObsEval"
            if saveIm:
                "plot image to check the image prediction quality"
                if config["n_stack"] > 1:
                    "saving other frames"
                    for step_r in range(config["n_stack"]):
                        name = "xHat_nextObsEval{}_frame{}".format(step, step_r)
                        plot_xHat(
                            CWH2WHC(observation[step_r * nc : (step_r + 1) * nc]),
                            tensor2image(xHat[:, step_r * nc : (step_r + 1) * nc]),
                            figure_path=save_dir,
                            with_nextObs=False,
                            name=name,
                            gradientStep=gradientStep,
                            suffix=suffix,
                            evaluate=evaluate,
                        )
                else:
                    plot_xHat(
                        CWH2WHC(observation[-3:, :, :]),
                        tensor2image(xHat[:, -3:, :, :]),
                        figure_path=save_dir,
                        with_nextObs=False,
                        name=name,
                        gradientStep=gradientStep,
                        suffix=suffix,
                        evaluate=evaluate,
                    )
                if elapsed_steps == xHat_nextObsEval_step:
                    if saved_step is not None:
                        plot_xHat(
                            CWH2WHC(observation[-3:, :, :]),
                            tensor2image(xHat[:, -3:, :, :]),
                            figure_path=os.path.join(save_dir, "xHat_nextObsEval"),
                            with_nextObs=False,
                            name="xHat_nextObsEval",
                            gradientStep=gradientStep,
                            saved_step=saved_step,
                        )

            # save states
            all_states[elapsed_steps] = pytorch2numpy(stateExpl)[0]
            elapsed_steps += 1

    "Release everything if job is finished"
    video_out.release()
    cv2.destroyAllWindows()

    nextObsEval = loss_log / len_traj
    print(" " * 100 + "done: nextObsEval = {:.3f}".format(nextObsEval))

    plotEmbedding(
        "UMAP",
        measures.copy(),
        all_states,
        figure_path=save_dir,
        gradientStep=gradientStep,
        saved_step=saved_step,
        proj_dim=3,
        suffix=suffix,
        env_name=config["env_name"],
        evaluate=evaluate,
    )
    plotEmbedding(
        "PCA",
        measures,
        all_states,
        figure_path=save_dir,
        gradientStep=gradientStep,
        saved_step=saved_step,
        proj_dim=3,
        suffix=suffix,
        env_name=config["env_name"],
        evaluate=evaluate,
    )

    "force the Garbage Collector to release unreferenced memory"
    del observations, video_out, stateExpl, observation, xHat, measures, all_states
    gc.collect()
    return nextObsEval


class AERunner(StateRunner):
    def __init__(self, config):
        super().__init__(config)
        srl_path = config["srl_path"] if config["srl_path"] else config["my_dir"]
        if srl_path:
            print("Load: encoder")
            self.encoder = torch.load(
                os.path.join(srl_path, "state_model.pt"),
                map_location=torch.device("cpu"),
            )
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

    def save_state_model(self, save_path):
        print("Saving models ......")
        save_model(self.encoder, save_path + "state_model")

    def train(self, training=True):
        self.encoder.train(training)

    def to_device(self, device="cpu"):
        torchDevice = torch.device(device)
        self.encoder.to(torchDevice)
