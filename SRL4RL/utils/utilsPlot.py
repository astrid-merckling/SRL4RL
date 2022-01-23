import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import colorsys
from PIL import Image
from sklearn import decomposition
import cv2
import umap  # to import after sklearn!

from bullet_envs import bullet_envs_path


def plotter(
    loss_log,
    save_path,
    name="loss",
    title=None,
    xlabel="gradient steps",
    ylabel="Mini-Batch Loss",
    backprop_per_eval=None,
    text_file=None,
):

    if text_file is not None:
        with open(text_file) as f:
            loss_log = np.array(f.read().split("\n"), dtype=np.float32)
    if backprop_per_eval is None:
        x = np.arange(0, len(loss_log))
    elif type(backprop_per_eval) is list:
        x = np.cumsum(backprop_per_eval)
    else:
        x = np.arange(
            backprop_per_eval,
            (len(loss_log) + 1) * backprop_per_eval,
            backprop_per_eval,
        )
    plt.plot(x, loss_log, "-k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=7, pad=2)
    plt.savefig(
        os.path.join(save_path, name + ".pdf"),
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def visualizeMazeExplor(
    env_name, robot_pos=None, robot_pos_dir="", save_dir="", name=""
):
    my_dpi = 150
    if robot_pos_dir:
        robot_pos = np.loadtxt(robot_pos_dir)
    else:
        assert robot_pos is not None, "no robot_pos"

    if env_name == "TurtlebotMazeEnv-v0":
        real_limit = 1.50 * 4
        img = Image.open(os.path.join(bullet_envs_path, "turtlebot/maze.png"))
    elif env_name == "TurtlebotEnv-v0":
        real_limit = 2
        img = Image.open(os.path.join(bullet_envs_path, "turtlebot/arena.png"))

    "center and scale the positions"
    dimX, dimY = img.size
    unit = dimX / 5  # 4*units + 1 unit for margins
    scale = (dimX - unit) / real_limit
    robot_pos[:, 0] = -robot_pos[:, 0] + real_limit
    scaledPos = (unit / 2 + robot_pos * scale).copy()

    fig, ax = plt.subplots()
    ax.imshow(img)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    t = np.arange(len(robot_pos))
    "blue -> green -> yellow -> red"
    ax.scatter(scaledPos[:, 0], scaledPos[:, 1], c=t, alpha=0.5, s=10)
    ax.plot(scaledPos[:, 0], scaledPos[:, 1], color="k", linewidth=0.5, alpha=0.7)

    try:
        plt.savefig(
            os.path.join(save_dir, name + ".png"),
            bbox_inches="tight",
            pad_inches=0.0,
            format="png",
            dpi=my_dpi,
        )
        plt.close()
    except:
        print("  Bug while saving in visualizeMazeExplor")
        plt.close()


def plot_xHat(
    img,
    img_hat,
    imgTarget=None,
    im_high_render=None,
    imLabel=None,
    figure_path="",
    name="xHat",
    gradientStep=None,
    saved_step=None,
    with_noise=False,
    with_nextObs=False,
    suffix="last",
    eval=False,
):
    im_size = 256
    color = img.shape[-1] == 3
    if img.shape[0] != im_size:
        "Resize to a square image, important that rgb_array values between 0 and 255"
        img = (img * 255).astype(np.uint8) if color else (img * 255).astype(np.uint8)
        img = (
            cv2.resize(
                img, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC
            ).astype(np.float32)
            / 255.0
        )
    if img_hat.shape[0] != im_size:
        img_hat = (
            (img_hat * 255).astype(np.uint8)
            if color
            else (img_hat * 255).astype(np.uint8)
        )
        img_hat = (
            cv2.resize(
                img_hat, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC
            ).astype(np.float32)
            / 255.0
        )
    if (imgTarget is not None) and (imgTarget.shape[0] != im_size):
        imgTarget = (
            (imgTarget * 255).astype(np.uint8)
            if color
            else (imgTarget * 255).astype(np.uint8)
        )
        imgTarget = (
            cv2.resize(
                imgTarget, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC
            ).astype(np.float32)
            / 255.0
        )
    if (im_high_render is not None) and (im_high_render.shape[0] != im_size):
        im_high_render = (
            (im_high_render * 255).astype(np.uint8)
            if color
            else (im_high_render * 255).astype(np.uint8)
        )
        im_high_render = (
            cv2.resize(
                im_high_render, dsize=(im_size, im_size), interpolation=cv2.INTER_CUBIC
            ).astype(np.float32)
            / 255.0
        )

    my_dpi = 100

    show_target = with_nextObs or with_noise

    ncol = 2
    nrow = 2 if show_target else 1

    fig, axs = plt.subplots(
        nrow,
        ncol,
        facecolor=(0.88, 0.88, 0.88),
        figsize=(ncol * im_size / my_dpi, nrow * im_size / my_dpi),
    )
    "to set the spacing between subplots"
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.04, wspace=0.18)

    if gradientStep is not None:
        plt.suptitle("gradient step {}".format(gradientStep), fontsize=10, y=0.0)
    axs = axs.flatten()

    if im_high_render is not None:
        axId = 0
        axs[axId].axis("off")
        axs[axId].imshow(im_high_render) if color else axs[axId].imshow(
            np.dstack([im_high_render, im_high_render, im_high_render])
        )
        axs[axId].set_title(imLabel, fontsize=10, pad=1.5)
    else:
        axs[0].axis(False)

    if show_target:
        # remove the top left frame
        axId = 1
        axs[axId].axis("off")
        axs[axId].imshow(imgTarget) if color else axs[axId].imshow(
            np.dstack([imgTarget, imgTarget, imgTarget])
        )
        if with_nextObs:
            axs[axId].set_title(r"next obs: $\mathbf{o}_{t+1}$", fontsize=10, pad=1.5)
            prediction_name = r"pred: $\hat{\mathbf{o}}_{t+1}$"
        elif with_noise:
            axs[axId].set_title(r"target obs: $\mathbf{o}_{t}$", fontsize=10, pad=1.5)
            prediction_name = r"pred: $\hat{\mathbf{o}}_{t}$"
    else:
        prediction_name = r"pred: $\hat{\mathbf{o}}_t$"

    axId = 2 if show_target else 0
    axs[axId].axis("off")
    axs[axId].imshow(img) if color else axs[axId].imshow(np.dstack([img, img, img]))
    axs[axId].set_title(r"input: $\mathbf{o}_t$", fontsize=10, pad=1.5)  # 'observation'

    axId = 3 if show_target else 1
    axs[axId].axis("off")
    axs[axId].imshow(img_hat) if color else axs[axId].imshow(
        np.dstack([img_hat, img_hat, img_hat])
    )
    axs[axId].set_title(prediction_name, fontsize=10, pad=1.5)

    if gradientStep:
        pad = 0.05
        arrow_x = 171.5
        arrow_y = 104
    else:
        pad = 0
        arrow_x = 168
        arrow_y = 85
    # Create the arrow
    if show_target:
        axs[2].annotate(
            "",
            xytext=(arrow_x, arrow_y),
            xy=(arrow_x + 32, arrow_y),
            xycoords="figure points",
            arrowprops=dict(
                arrowstyle="simple",
                color="k",
            ),
        )
    else:
        axs[0].annotate(
            "",
            xytext=(arrow_x, arrow_y),
            xy=(arrow_x + 32, arrow_y),
            xycoords="figure points",
            arrowprops=dict(
                arrowstyle="simple",
                color="k",
            ),
        )

    if saved_step is not None:
        epoch_ = "-%06d" % saved_step
    else:
        epoch_ = "_%s" % suffix

    plt.savefig(
        os.path.join(figure_path, name + epoch_ + ".png"),
        bbox_inches="tight",
        pad_inches=pad,
        format="png",
        dpi=my_dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close()

    if eval:
        "save images separately"
        cv2.imwrite(
            os.path.join(figure_path, "{}_img.png".format(name)), 255 * img[:, :, ::-1]
        )
        cv2.imwrite(
            os.path.join(figure_path, "{}_img_hat.png".format(name)),
            255 * img_hat[:, :, ::-1],
        )
        if imgTarget is not None:
            cv2.imwrite(
                os.path.join(figure_path, "{}_imgTarget.png".format(name)),
                255 * imgTarget[:, :, ::-1],
            )
        if im_high_render is not None:
            cv2.imwrite(
                os.path.join(figure_path, "{}_{}.png".format(name, imLabel)),
                255 * im_high_render[:, :, ::-1],
            )


epsilon = 1e-6  # Avoid NaN


def angle2Dhsv(gt, normalizeAngle=False):
    "HSV colors with proj_dim=2"
    Xmax, Xmin = np.max(gt[:, 0]), np.min(gt[:, 0])
    if (np.mean(gt[:, 0]) - Xmin) / (Xmax - Xmin) < 0.5:
        gt[:, 0] = -gt[:, 0] + np.max(gt[:, 0])
        Xmax, Xmin = Xmax - Xmin, 0
    if normalizeAngle:
        "normalize angles between [0,2*pi]"
        angleMax, angleMin = np.max(gt[:, -1]), np.min(gt[:, -1])
        gt[:, -1] = 2 * np.pi * (gt[:, -1] - angleMin) / (angleMax - angleMin)
        XminC = (
            Xmin - (Xmax - Xmin) / 2
        )  # in order to better visualize different colors
    else:
        XminC = (
            Xmin - (Xmax - Xmin) / 10
        )  # in order to better visualize different colors
    # not needed because when modulo2PI was not done, angle is limited: gt[:, -1] = np.mod(gt[:, -1], (2 * np.pi))
    HSV = [(x[-1] * 1.0 / (2 * np.pi), (x[0] - XminC) / (Xmax - XminC), 1) for x in gt]
    RGB = [colorsys.hsv_to_rgb(*HSV_) for HSV_ in HSV]
    colors = np.array(RGB)
    HSVx = ((gt[:, 0] - Xmin) / (Xmax - Xmin) + 0.2) * np.cos(gt[:, -1])
    HSVy = ((gt[:, 0] - Xmin) / (Xmax - Xmin) + 0.2) * np.sin(gt[:, -1])
    gt2plot = np.array((HSVx, HSVy)).T
    return gt2plot, colors


def angle3Dhsv(gt, normalizeAngle=False):
    "HSV colors with proj_dim=3"
    Xmax, Xmin, Ymax, Ymin = (
        np.max(gt[:, 0]),
        np.min(gt[:, 0]),
        np.max(gt[:, 1]),
        np.min(gt[:, 1]),
    )
    if (np.mean(gt[:, 0]) - Xmin) / (Xmax - Xmin) < 0.5:
        gt[:, 0] = -gt[:, 0] + np.max(gt[:, 0])
        Xmax, Xmin = Xmax - Xmin, 0
    if (np.mean(gt[:, 1]) - Ymin) / (Ymax - Ymin) < 0.5:
        gt[:, 1] = -gt[:, 1] + np.max(gt[:, 1])
        Ymax, Ymin = Ymax - Ymin, 0
    if normalizeAngle:
        "normalize angles between [0,2*pi]"
        angleMax, angleMin = np.max(gt[:, -1]), np.min(gt[:, -1])
        gt[:, -1] = 2 * np.pi * (gt[:, -1] - angleMin) / (angleMax - angleMin)
        YminC = (
            Ymin - (Ymax - Ymin) / 2
        )  # in order to better visualize different colors
        XminC = (
            Xmin - (Xmax - Xmin) / 2
        )  # in order to better visualize different colors
    else:
        YminC = (
            Ymin - (Ymax - Ymin) / 2
        )  # in order to better visualize different colors
        XminC = (
            Xmin - (Xmax - Xmin) / 10
        )  # in order to better visualize different colors
    # not needed because when modulo2PI was not done, angle is limited: gt[:, -1] = np.mod(gt[:, -1], (2 * np.pi))
    HSV = [
        (
            x[-1] * 1.0 / (2 * np.pi),
            (x[0] - XminC) / (Xmax - XminC),
            (x[1] - YminC) / max((Ymax - YminC), epsilon),
        )
        for x in gt
    ]
    RGB = [colorsys.hsv_to_rgb(*HSV_) for HSV_ in HSV]
    colors = np.array(RGB)  # between 0 and 1
    "for equal axis scales"
    HSVx = ((gt[:, 0] - Xmin) / (Xmax - Xmin) + 0.2) * np.cos(gt[:, -1])
    HSVy = ((gt[:, 0] - Xmin) / (Xmax - Xmin) + 0.2) * np.sin(gt[:, -1])
    HSVz = (gt[:, 1] - Ymin) / max((Ymax - Ymin), epsilon)
    gt2plot = np.array((HSVx, HSVy, HSVz)).T
    return gt2plot, colors


def obs2torus(states):
    R = 1
    P = 0.3
    states = np.vstack(
        [
            (R + P * np.cos(states[:, 1])) * np.cos(states[:, 0]),
            (R + P * np.cos(states[:, 1])) * np.sin(states[:, 0]),
            P * np.sin(states[:, 1]),
        ]
    ).transpose(1, 0)
    return states


def plotEmbedding(
    method,
    measures,
    all_states,
    figure_path="",
    gradientStep=None,
    saved_step=None,
    proj_dim=2,
    n_neighbors=15,
    metric="euclidean",
    suffix="",
    env_name="TurtlebotMazeEnv-v0",
    eval=False,
):
    # matplotlib.rcParams['backend'] = 'TkAgg'
    gt2plotParts = None
    print("  {}D {} projection of ground truth ......".format(proj_dim, method))
    withHSV = False
    min_dist = 0.8  # default is 0.1
    assert proj_dim <= all_states.shape[-1]
    if env_name == "TurtlebotMazeEnv-v0" and measures.shape[-1] == 2:
        gt4colors = measures.copy()
        gt4colors[:, 0] = gt4colors[:, 0] + (-2 * gt4colors[:, 0]) * (
            gt4colors[:, 1] < 1.04987811
        )
        colors = gt4colors[:, 0] + gt4colors[:, 1]
        gt2plot = measures
    elif env_name == "TurtlebotMazeEnv-v0":
        withHSV = True
        proj_dim = 3
        measures[:, 0] = (
            -measures[:, 0] + 1.5 * 4
        )  # in order to better visualize the maze
        gt2plot, colors = angle3Dhsv(measures)
    elif env_name == "InvertedPendulumSwingupBulletEnv-v0":
        withHSV = True
        proj_dim = 2
        gt2plot, colors = angle2Dhsv(measures)
    elif "Reacher" in env_name:
        torus = True
        colors = measures[:, 0] % (2 * np.pi)
        gt2plot = obs2torus(measures)
    else:
        withHSV = True
        gt2plotParts, colorsParts = [], []
        measures = measures.transpose(1, 0, 2)
        proj_dimParts = [3] * measures.shape[0]
        proj_dimParts[0] = 2
        for i, partMeasure in enumerate(measures):
            if i == 0:
                "torso part"
                partMeasure = partMeasure[:, 1:]
                gt2plot, colors = angle2Dhsv(partMeasure, normalizeAngle=True)
            else:
                gt2plot, colors = angle3Dhsv(partMeasure, normalizeAngle=True)
            gt2plotParts.append(gt2plot)
            colorsParts.append(colors)

    if withHSV:
        cmap = None
    else:
        cmap = "gist_rainbow"

    if method == "UMAP":
        embedding = umap.UMAP(
            n_components=proj_dim,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        ).fit_transform(all_states)
    elif method == "PCA":
        pcaTransform = decomposition.PCA(n_components=proj_dim)
        embedding = pcaTransform.fit_transform(all_states)

    GTlegend = "Ground truth"

    if gt2plotParts is not None:
        embeddingParts = {"%s" % proj_dim: embedding}
        if proj_dim == 3:
            proj_dimLast = 2
        else:
            proj_dimLast = 3
        if method == "UMAP":
            embeddingParts["%s" % proj_dimLast] = umap.UMAP(
                n_components=proj_dimLast,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
            ).fit_transform(all_states)
        elif method == "PCA":
            pcaTransform = decomposition.PCA(n_components=proj_dimLast)
            embeddingParts["%s" % proj_dimLast] = pcaTransform.fit_transform(all_states)
        for i, (gt2plot, colors, proj_dim) in enumerate(
            zip(gt2plotParts, colorsParts, proj_dimParts)
        ):
            embedding = embeddingParts["%s" % proj_dim]
            if suffix != "":
                namePart = suffix + "-part%02d" % i
            else:
                namePart = "part%02d" % i
            createFig(
                method,
                gt2plot,
                embedding,
                colors,
                cmap,
                GTlegend + " robot part %01d" % i,
                figure_path=figure_path,
                gradientStep=gradientStep,
                saved_step=saved_step,
                proj_dim=proj_dim,
                suffix=namePart,
                eval=eval,
            )
    else:
        createFig(
            method,
            gt2plot,
            embedding,
            colors,
            cmap,
            GTlegend,
            figure_path=figure_path,
            gradientStep=gradientStep,
            saved_step=saved_step,
            proj_dim=proj_dim,
            suffix=suffix,
            eval=eval,
        )


def createFig(
    method,
    gt2plot,
    embedding,
    colors,
    cmap,
    GTlegend,
    figure_path="",
    gradientStep=None,
    saved_step=None,
    proj_dim=2,
    suffix="",
    eval=False,
):
    linewidth = 0
    size = 8
    "Create figure"
    if proj_dim == 2:
        fig, axs = plt.subplots(1, 2, figsize=(7 * 2, 7 * 1))
        axs = axs.flatten()

        axs[0].scatter(
            *gt2plot.T,
            s=size,
            c=colors,
            cmap=cmap,
            alpha=1.0,
            edgecolors="black",
            linewidth=linewidth
        )
        plt.setp(axs[0], xticks=[], yticks=[])
        axs[0].set_title(GTlegend)

        axs[1].scatter(
            *embedding.T,
            s=size,
            c=colors,
            cmap=cmap,
            alpha=1.0,
            edgecolors="black",
            linewidth=linewidth
        )
        plt.setp(axs[1], xticks=[], yticks=[])
        axs[1].set_title("2D %s projection of Learned State Space" % method)

    elif proj_dim == 3:
        ## Ground Truth
        Axes3D
        fig = plt.figure(figsize=(7 * 2, 7 * 1))
        if gt2plot.shape[-1] == 2:
            ax = fig.add_subplot(121)
            ax.scatter(
                *gt2plot.T,
                s=size,
                c=colors,
                cmap=cmap,
                edgecolors="black",
                linewidth=linewidth
            )
            plt.setp(ax, xticks=[], yticks=[])
            ax.set_title("Ground truth", y=1.05)
        else:
            ax = fig.add_subplot(1, 2, 1, projection="3d")
            ax.scatter(
                *gt2plot.T,
                s=size,
                c=colors,
                cmap=cmap,
                alpha=1.0,
                edgecolors="black",
                linewidth=linewidth
            )
            plt.setp(ax, xticks=[], yticks=[], zticks=[])
            resize_3Dfig(ax, gt2plot)
            remove_background_3Dfig(ax)
            ax.set_title(GTlegend, y=1.05)

        # Plot results
        ax = fig.add_subplot(1, 2, 2, projection="3d")
        ax.scatter(
            *embedding.T,
            s=size,
            c=colors,
            cmap=cmap,
            alpha=1.0,
            edgecolors="black",
            linewidth=linewidth
        )
        plt.setp(ax, xticks=[], yticks=[], zticks=[])
        resize_3Dfig(ax, embedding)
        remove_background_3Dfig(ax)
        ax.set_title("3D %s projection of Learned State Space" % method, y=1.05)

    "to set the spacing between subplots"
    fig.tight_layout()

    if gradientStep is not None:
        plt.suptitle("gradient step {}".format(gradientStep), fontsize=14)

    if suffix != "":
        suffix = "-" + suffix
    else:
        suffix = ""
    if saved_step is not None:
        epoch_ = "-%06d" % saved_step
        plt.savefig(
            os.path.join(
                figure_path,
                "{}proj/{}proj-dim{}{}".format(method, method, proj_dim, suffix)
                + epoch_
                + ".png",
            ),
            bbox_inches="tight",
            format="png",
        )  # pad_inches=0.1,
    # plt.savefig(path +".pdf", bbox_inches='tight', pad_inches=0.3, format='pdf')
    plt.savefig(
        os.path.join(
            figure_path, "{}proj-dim{}{}.pdf".format(method, proj_dim, suffix)
        ),
        bbox_inches="tight",
        format="pdf",
    )  # pad_inches=0.1,
    plt.close()


def resize_3Dfig(ax, data):
    data2plotXmin, data2plotXmax = min(data[:, 0]), max(data[:, 0])
    data2plotYmin, data2plotYmax = min(data[:, 1]), max(data[:, 1])
    data2plotZmin, data2plotZmax = min(data[:, 2]), max(data[:, 2])

    ax.set_xlim(data2plotXmin, data2plotXmax)
    ax.set_ylim(data2plotYmin, data2plotYmax)
    ax.set_zlim(data2plotZmin, data2plotZmax)


def remove_background_3Dfig(ax):
    # Get rid of colored axes planes
    # First remove fill
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    # ax.xaxis.pane.set_edgecolor('black')
    # ax.yaxis.pane.set_edgecolor('black')
    # ax.zaxis.pane.set_edgecolor('black')
