import shutil
import pandas as pd
from training import run_training
import argparse
from enum import Enum
import torch
from typing import Union, List, Tuple
import time
import yaml
import os
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gradient_descent_the_ultimate_optimizer import gdtuo
from sam import SAM
import datetime
import sys
from multiprocessing import Queue, Process, Semaphore
import gc
import logging
logging.basicConfig(format="%(message)s %(asctime)s")


class GlobalQueueActor:
    def __init__(self):
        self.q = Queue()
        self.accuracy_history = Queue()
        self.uar_history = Queue()
        self.f1_history = Queue()
        self.train_loss_history = Queue()
        self.valid_loss_history = Queue()
        self.run_name_history = Queue()
        self.semaphore = Semaphore(1)
        self.counter = Queue(1)
        self.counter.put(0)

    def set_permutation(self, permutation):
        [self.q.put(i) for i in permutation]
        self.total_experiments = len(permutation)

    def get_next(self):
        current_experiment = self.total_experiments - self.q.qsize() + 1
        if self.q.qsize() > 0:
            next_p = self.q.get()
            return next_p, current_experiment, self.total_experiments
        return None, None, None

    def update(self, accuracy, uar, f1, train_loss, valid_loss, run_name, actor_num):
        with self.semaphore:
            self.accuracy_history.put(accuracy)
            self.uar_history.put(uar)
            self.f1_history.put(f1)
            self.train_loss_history.put(train_loss)
            self.valid_loss_history.put(valid_loss)
            self.run_name_history.put(run_name)
            current_counter = self.counter.get() + 1
            self.counter.put(current_counter)
            log_str = f"\nFinished {current_counter}/{self.total_experiments} runs.\nFinished by Actor{actor_num}.\nTimestamp:"
            logging.log(level=logging.CRITICAL, msg=log_str)

    def failed(self):
        with self.semaphore:
            current_counter = self.counter.get() + 1
            self.counter.put(current_counter)

    def get_metrics(self):
        self.accuracy_history = GlobalQueueActor.empty_queue(
            self.accuracy_history)
        self.uar_history = GlobalQueueActor.empty_queue(
            self.uar_history)
        self.f1_history = GlobalQueueActor.empty_queue(
            self.f1_history)
        self.train_loss_history = GlobalQueueActor.empty_queue(
            self.train_loss_history)
        self.valid_loss_history = GlobalQueueActor.empty_queue(
            self.valid_loss_history)
        self.run_name_history = GlobalQueueActor.empty_queue(
            self.run_name_history)
        return self.accuracy_history, self.uar_history, self.f1_history, self.train_loss_history, self.valid_loss_history, self.run_name_history

    @classmethod
    def empty_queue(cls, q):
        res = []
        while q.qsize() > 0:
            n = q.get()
            res.append(n)
        return np.array(res)


class ParallelActor:
    def __init__(self, q: GlobalQueueActor, actor_num, data_root, device, run_name, results_path, features, feature_dir, pretrained_dir, custom_feature_path, state, base_folder, disable_progress_bar) -> None:
        self.q = q
        self.actor_num = actor_num

        self.data_root = data_root
        self.device = device
        self.run_name = run_name
        self.results_path = results_path
        self.features = features
        self.feature_dir = feature_dir
        self.pretrained_dir = pretrained_dir,
        self.custom_feature_path = custom_feature_path
        self.state = state
        self.base_folder = base_folder
        self.disable_progress_bar = disable_progress_bar

    def run_parallel(self):
        while True:
            experiment, current_experiment, total_experiments = self.q.get_next()
            if experiment is None:
                return
            with open(os.path.join(self.base_folder, self.results_path, "actor"+str(self.actor_num)+".txt"), "a") as sys.stdout:
                print("Running experiment", str(current_experiment) +
                      "/" + str(total_experiments))
                print("")
                print(torch.cuda.get_device_properties(self.actor_num).name)
                run = NeuralBench(
                    data_root=self.data_root,
                    feature_dir=self.feature_dir,
                    pretrained_dir = self.pretrained_dir,
                    device=self.device,
                    run_name=self.run_name,
                    results_path=self.results_path,
                    features=self.features,
                    custom_feature_path=self.custom_feature_path,
                    state=self.state,
                    # TODO: Change info here!
                    approach=experiment[0],
                    category=experiment[1],
                    dataset=experiment[2],
                    batch_size=experiment[3],
                    epochs=experiment[4],
                    learning_rate=experiment[5],
                    seed=experiment[6],
                    optimizer=experiment[7],
                    sheduler_wrapper=experiment[8],
                    exclude_cities=experiment[9],
                    base_folder=self.base_folder,
                    disable_progress_bar=self.disable_progress_bar,
                )
                try:
                    accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history, run_name_history = run.run()
                    self.q.update(accuracy_history, uar_history,
                                  f1_history, train_loss_history, valid_loss_history, run_name_history, self.actor_num)
                    del accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history, run_name_history
                except torch.cuda.OutOfMemoryError:
                    self.q.failed()
                    log_str = f"\nCUDA: Out of Memory in run {current_experiment}/{total_experiments} runs.\nProduced by Actor{self.actor_num}.\nTimestamp:"
                    logging.log(level=logging.CRITICAL, msg=log_str)
                except BaseException as e:
                    self.q.failed()
                    log_str = f"\nException in run {current_experiment}/{total_experiments} runs.\nProduced by Actor{self.actor_num}.\nError: {e}\nTimestamp:"
                    logging.log(level=logging.CRITICAL, msg=log_str)
                del run
                gc.collect()
                torch.cuda.empty_cache()


class ModelEnum(Enum):
    CNN14 = "cnn14"
    CNN10 = "cnn10"
    SINCNET = "sincnet"
    ResNet50 = "resnet50"
    Efficientnet_b0 = "efficientnet-b0"
    Efficientnet_b4 = "efficientnet-b4"


class DataEnum(Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    TRANSPORTATION = "transportation"
    NONE = None


class OptimizerEnum(Enum):
    SGD = "SGD"
    ADAM = "Adam"
    RMSPROP = "RMSprop"
    SAM = "SAM"


def generate_plot_base(title, accuracy_type="accuracy", loss=True):
    lw = 1.2
    f_title = 24
    f_labels = 20
    fig = plt.figure("Grid Search", figsize=(22, 10))
    fig.suptitle(title, fontsize=f_title+4)
    if loss:
        acc_plot, loss_plot = fig.subplots(1, 2, squeeze=True)
        loss_plot.spines.top.set(visible=False)
        loss_plot.spines.right.set(visible=False)
        loss_plot.set_title("Loss Plot", fontdict={"fontsize": f_title})
        loss_plot.set_ylabel("Loss", fontsize=f_labels)
        loss_plot.set_xlabel("Epochs", fontsize=f_labels)
    else:
        acc_plot = fig.subplots(1, 1, squeeze=True)
        loss_plot = None
    acc_plot.spines.top.set(visible=False)
    acc_plot.spines.right.set(visible=False)
    acc_plot.set_title(accuracy_type+" Plot", fontdict={"fontsize": f_title})
    acc_plot.set_ylabel(accuracy_type, fontsize=f_labels)
    acc_plot.set_xlabel("Epochs", fontsize=f_labels)
    return fig, acc_plot, loss_plot, lw


class Timer:
    def __init__(self) -> None:
        self.time_log = []
        self.start_time = None

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self) -> None:
        if self.start_time is None:
            raise Exception("Timer not yet started!")
        run_time = time.time() - self.start_time
        self.start_time = None
        self.time_log.append(run_time)

    def get_time_log(self) -> list:
        return self.time_log

    def get_mean_seconds(self) -> float:
        return sum(self.time_log)/len(self.time_log)

    def pretty_time(self) -> str:
        pretty_time = datetime.timedelta(seconds=int(self.get_mean_seconds()))
        return str(pretty_time)


class OptimizerWrapper:
    def __init__(self, optimizer_type: callable, **optimizer_kwargs) -> None:
        """Wrapper Class for Optimizer

        Args:
            optimizer_type (callable): Optimizer class to be used.
        """
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs

    def create(self, model: torch.nn.Module, lr: float, device=None) -> torch.optim.Optimizer:
        """Creates an optimizer with set parameters

        Args:
            model (torch.nn.Module): Model to be used for the optimizer.
            lr (float): Learning rate to be used for the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        self.optimizer_kwargs["lr"] = lr
        if self.get_name() in ["KFACOptimizer", "EKFACOptimizer"]:
            return self.optimizer_type(model, **self.optimizer_kwargs)
        return self.optimizer_type(model.parameters(), **self.optimizer_kwargs)

    def get_params(self) -> dict:
        """Return non default parameter set of optimizer

        Returns:
            dict: Dictionary of parameters.
        """
        return self.optimizer_kwargs

    def get_name(self) -> str:
        """Return optimizer name.

        Returns:
            str: Name.
        """
        return self.optimizer_type.__name__

class SAMWrapper:
    def __init__(self, optimizer_type: callable, **optimizer_kwargs) -> None:
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs

    def create(self, model: torch.nn.Module, lr: float, device) -> gdtuo.ModuleWrapper:
        self.optimizer_kwargs["lr"] = lr
        # if self.get_name() in ["KFACOptimizer", "EKFACOptimizer"]:
            #return self.optimizer_type(model, **self.optimizer_kwargs)
        return SAM(model.parameters(), self.optimizer_type, **self.optimizer_kwargs)
        #return self.optimizer_type(model.parameters(), **self.optimizer_kwargs)
        
    def get_params(self) -> dict:
        """Return non default parameter set of optimizer

        Returns:
            dict: Dictionary of parameters.
        """
        return self.optimizer_kwargs

    def get_name(self) -> str:
        """Return optimizer name.

        Returns:
            str: Name.
        """
        return "SAM"



class GDTUOWrapper:
    def __init__(self, stack: List[Tuple[gdtuo.Optimizable, dict, bool]]) -> None:
        self.optimizer_stack, self.parameters, self.set_lr = zip(*stack[::-1])
        self.optimizer_stack = list(self.optimizer_stack)
        self.parameters = list(self.parameters)
        self.set_lr = list(self.set_lr)
        self.wrapper = None

    def create(self, model: torch.nn.Module, lr: float, device) -> gdtuo.ModuleWrapper:
        full_optimizer = gdtuo.NoOpOptimizer()
        for optimizer, params, set_lr in zip(self.optimizer_stack, self.parameters, self.set_lr):
            if set_lr:
                params["alpha"] = lr
            params["optimizer"] = full_optimizer
            full_optimizer = optimizer(**params)

        wrapper = gdtuo.ModuleWrapper(
            model.to(device), optimizer=full_optimizer)
        wrapper.initialize()
        self.start_params = self._create_param_dict(wrapper)
        self.wrapper = wrapper
        return self.wrapper

    def _create_param_dict(self, wrapper: gdtuo.ModuleWrapper, round_to=10) -> None:
        def _param_default(k, v):
            return v.detach().cpu().item()

        def _param_rmsprop(k, v):
            if k == "alpha":
                v = torch.square(v)
            elif k == "gamma":
                v = (v.tanh() + 1.) / 2.
            return v.detach().cpu().item()

        def _param_rmsprop_alpha(k, v):
            if k == "alpha":
                v = torch.square(v)
            return v.detach().cpu().item()

        def _param_adam(k, v):
            if k in ["beta1", "beta2"]:
                v = (v.tanh() + 1.) / 2.
            return v.detach().cpu().item()

        parameters = []
        opt = wrapper.optimizer
        while not isinstance(opt, gdtuo.NoOpOptimizer):
            if isinstance(opt, (gdtuo.SGD, gdtuo.AdaGrad, gdtuo.AdamBaydin)):
                _param_convert = _param_default
            elif isinstance(opt, gdtuo.RMSProp):
                _param_convert = _param_rmsprop
            elif isinstance(opt, gdtuo.RMSPropAlpha):
                _param_convert = _param_rmsprop_alpha
            elif isinstance(opt, gdtuo.Adam):
                _param_convert = _param_adam
            parameters.append(
                {k: round(_param_convert(k, v), round_to) for (k, v) in opt.parameters.items()})
            opt = opt.optimizer
        return parameters

    def get_params(self) -> List[dict]:
        if self.wrapper is None:
            return self.parameters
        return {
            "start_params": self.start_params,
            "end_params": self._create_param_dict(self.wrapper)
        }

    def get_name(self) -> str:
        optimizer_name = ["GDTUO"]
        for optimizer in self.optimizer_stack:
            optimizer_name.append(optimizer().__class__.__name__)
        return "-".join(optimizer_name)




class ShedulerWrapper:
    def __init__(self, sheduler_type: callable, **sheduler_kwargs) -> None:
        """Wrapper Class for Learning Rate Sheduler

        Args:
            sheduler_type (callable): Sheduler class to be used.
            sheduler_kwargs (**kwargs): Additional keyword arguments to be passed to the sheduler class.
        """
        self.sheduler_type = sheduler_type
        self.sheduler_kwargs = sheduler_kwargs

    def create(self, optim: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """Creates a sheduler with set parameters.

        Args:
            optim (torch.optim.Optimizer): Optimizer to be used for the sheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Sheduler.
        """
        return self.sheduler_type(optim, **self.sheduler_kwargs)

    def get_params(self) -> dict:
        """Return non default parameter set of sheduler.

        Returns:
            dict: Dictionary of parameters.
        """
        return self.sheduler_kwargs

    def get_name(self) -> str:
        """Return sheduler name.

        Returns:
            str: Name.
        """
        return self.sheduler_type.__name__


class ExcludeSearch:
    def __init__(self,
                 dataset: str = "ignore",
                 approach: str = "ignore",
                 category: DataEnum = "ignore",
                 batch_size: int = "ignore",
                 epochs: int = "ignore",
                 learning_rate: float = "ignore",
                 pretrained: bool = "ignore",
                 seed: int = "ignore",
                 optimizer: Union[OptimizerEnum,
                                  OptimizerWrapper, GDTUOWrapper] = "ignore",
                 sheduler_wrapper: Union[ShedulerWrapper,
                                         List[ShedulerWrapper]] = "ignore"
                 ) -> None:
        """Create a configuration that should be excluded from the grid search.
        If two or more parameters match in a configuration it will be ignored.
        "ignore" ignores the configuration and is seen as a wildcard.
        "all" rejects every configuration that is not None.

        Args:
            approach (ModelEnum, optional): Approach to be excluded. Defaults to "ignore".
            category (DataEnum, optional): Category to be excluded. Defaults to "ignore".
            batch_size (int, optional): Batch size to be excluded. Defaults to "ignore".
            epochs (int, optional): epochs to be excluded. Defaults to "ignore".
            learning_rate (float, optional): Learning rate to be excluded. Defaults to "ignore".
            seed (int, optional): Seed to be excluded. Defaults to "ignore".
            optimizer (Union[OptimizerEnum, OptimizerWrapper, GDTUOWrapper], optional): Optimizer to be excluded. Defaults to "ignore".
            sheduler_wrapper (Union[ShedulerWrapper, List[ShedulerWrapper]], optional): Sheduler or List of Shedulers to be excluded. Defaults to "ignore".
        """
        self.match_cases = [
            dataset,
            approach,
            category,
            batch_size,
            epochs,
            learning_rate,
            pretrained,
            seed,
            optimizer,
            sheduler_wrapper
        ]

    def match(self, permutations: List[tuple]) -> List[tuple]:
        """Matches a list of permutations to be excluded

        Args:
            permutations (List[tuple]): List of permutations to be matched.

        Returns:
            List[tuple]: Permutations that are valid.
        """
        new_permutations = []
        for permutation in permutations:
            matches = 0
            for i, case in enumerate(self.match_cases):
                if case != "ignore" and case == permutation[i]:
                    matches += 1
                elif case == "all" and permutation[i] != None:
                    matches += 1
            if matches < 2:
                new_permutations.append(permutation)
        return new_permutations


class NeuralBench():
    def __init__(self,
                 data_root: str,
                 device: str,
                 features: str,
                 feature_dir: str = "",
                 pretrained_dir: str = "",
                 run_name: str = None,
                 results_path: str = None,
                 custom_feature_path: str = None,
                 state: str = None,
                 approach: str = "cnn10",
                 category: DataEnum = DataEnum.NONE,
                 dataset: str = "DCASE2020",
                 batch_size: int = 32,
                 epochs: int = 60,
                 learning_rate: float = 0.001,
                 pretrained: bool = False,
                 seed: int = 0,
                 optimizer: Union[OptimizerEnum,
                                  OptimizerWrapper, GDTUOWrapper] = OptimizerEnum.SGD,
                 sheduler_wrapper: Union[ShedulerWrapper,
                                         List[ShedulerWrapper]] = None,
                 exclude_cities: List[List[str]] = None,
                 base_folder: str = None,
                 disable_progress_bar=False
                 ) -> None:
        """Create a Training Configuration

        Args:
            data_root (str): Path data has been extracted.
            device (str): CUDA-enabled device to use for training.
            features (str): Path to features.
            run_name (str, optional): Path where results are to be stored, if None the name is generated automatically. Defaults to None.
            custom_feature_path (str, optional): Custom .npy location of features. Defaults to None.
            state (str,optional): Optional initial state file path. Defaults to None.
            approach (ModelEnum, optional): Model to be used for training. Defaults to ModelEnum.CNN10.
            category (DataEnum, optional): Data category to be used for training. Defaults to DataEnum.NONE.
            dataset (str, optional): Dataset/Task
            batch_size (int, optional): Batch size to be used for training. Defaults to 32.
            epochs (int, optional): Training epochs. Defaults to 60.
            learning_rate (float, optional): Learning rate for optimizer. Defaults to 0.001.
            seed (int, optional): Seed to be used for training. Defaults to 0.
            optimizer (Union[OptimizerEnum, OptimizerWrapper, GDTUOWrapper], optional): Optimizer to be used for training. Defaults to OptimizerEnum.SGD.
            sheduler (Union[ShedulerWrapper, list], optional): Wrapped Sheduler or list of Shedulers to be used for training. Defaults to None.
            exclude_cities (List[str], optional): List of cities to exclude from training data. Defaults to None.
            base_folder (str, optional): Base folder for "data_root", "run_name", "features" and "custom_feature_path". Defaults to None.
            disable_progress_bar (bool, optional): Disable tqdm progress bar while training. Defaults to False.
        """
        if base_folder is None:
            base_folder = ""
        else:
            base_folder += "/"

        self.train_timer = Timer()
        self.valid_timer = Timer()

        self.args = argparse.Namespace()
        self.args.data_root = base_folder+data_root
        self.args.device = device
        torch.cuda.set_device(
            int(device[-1])) if isinstance(device, str) else torch.cuda.set_device(device)
        self.args.features = base_folder+features
        self.args.feature_dir = feature_dir
        self.args.pretrained_dir = pretrained_dir
        self.args.results_path = results_path
        self.args.custom_feature_path = base_folder+custom_feature_path
        self.args.state = state
        self.args.approach = approach
        self.args.category = category.value
        self.args.dataset = dataset
        self.args.batch_size = batch_size
        self.args.epochs = epochs
        self.args.learning_rate = learning_rate
        self.args.pretrained = pretrained
        print("Neural BenchS")
        print(pretrained)
        self.args.seed = seed
        self.args.optimizer = optimizer.value if isinstance(
            optimizer, OptimizerEnum) else optimizer
        self.args.optimizer_name = optimizer.value if isinstance(
            optimizer, OptimizerEnum) else optimizer.get_name() 
        self.args.sheduler_wrapper = sheduler_wrapper
        if exclude_cities is None:
            exclude_cities = ["None"]
        self.args.exclude_cities = exclude_cities
        self.args.train_timer = self.train_timer
        self.args.valid_timer = self.valid_timer
        self.args.disable_progress_bar = disable_progress_bar

        if isinstance(self.args.sheduler_wrapper, list):
            self.args.sheduler_name = "-".join(
                [s.get_name() for s in self.args.sheduler_wrapper])
        elif isinstance(self.args.sheduler_wrapper, ShedulerWrapper):
            self.args.sheduler_name = self.args.sheduler_wrapper.get_name()
        else:
            self.args.sheduler_name = "None"

        if run_name is None:
            run_name = str(self.args.dataset)+"_" +\
                str(self.args.approach)+"_" +\
                "pretrained-" + str(self.args.pretrained)+"_" +\
                str(self.args.category)+"_" +\
                str(self.args.optimizer_name)+"_" +\
                str(self.args.learning_rate).replace(".", "-")+"_" +\
                str(self.args.batch_size)+"_" +\
                str(self.args.epochs)+"_" +\
                str(self.args.seed)+"_" +\
                str(self.args.sheduler_name)+"_" +\
                str("-".join(self.args.exclude_cities))
        # print("-"*50)
        print("pretrained")
        print(self.args.pretrained)
        self.run_name = run_name
        if self.args.results_path == None:
            self.args.results_path = ""
        self.args.results_root = os.path.join(
            base_folder, self.args.results_path, run_name)

    def plot_accuracy(self, title=None, file_name=None):
        if title == None:
            title = self.run_name
        if file_name == None:
            file_name = "eval_plots"
        fig, acc_plot, loss_plot, lw = generate_plot_base(title)
        colors = matplotlib.colormaps["rainbow"](
            np.linspace(0, 1, 5)) * 0.8

        acc_plot.plot(range(1, len(self.accuracy_history)+1), self.accuracy_history,
                      label="Accuracy", c=colors[0], linewidth=lw)
        acc_plot.plot(range(1, len(self.uar_history)+1), self.uar_history,
                      label="UAR", c=colors[1], linewidth=lw)
        acc_plot.plot(range(1, len(self.f1_history)+1), self.f1_history,
                      label="F1", c=colors[2], linewidth=lw)
        loss_plot.plot(range(1, len(self.train_loss_history)+1), self.train_loss_history,
                       label="Train Loss", c=colors[3], linewidth=lw)
        loss_plot.plot(range(1, len(self.valid_loss_history)+1), self.valid_loss_history,
                       label="Valid Loss", c=colors[4], linewidth=lw)

        acc_plot.legend(prop={"size": 10})
        loss_plot.legend(prop={"size": 10})
        fig.savefig(os.path.join(self.args.results_root, file_name+".png"), transparent=False, dpi=100,
                    bbox_inches="tight", pad_inches=1)
        plt.clf()

    def export_metadata(self):
        print("Save Metdata")
        metadata = {
            "min train loss": min(self.train_loss_history),
            "min valid loss": min(self.valid_loss_history),
            "max valid acc": max(self.accuracy_history),
            "max valid uar": max(self.uar_history),
            "max valid f1": max(self.f1_history),
            "mean train time": self.train_timer.pretty_time(),
            "mean valid time": self.valid_timer.pretty_time(),
            "device": self.args.device if isinstance(
                self.args.device, str) else torch.cuda.get_device_name(self.args.device),
            "state": self.args.state,
            "approach": self.args.approach,
            "category": self.args.category,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "learning_rate": self.args.learning_rate,
            "seed": self.args.seed,
            "optimizer": self.args.optimizer_name,
            "exclude_cities": self.args.exclude_cities,
        }

        if not isinstance(self.args.optimizer, str):
            metadata["optimizer params"] = self.args.optimizer.get_params()

        metadata["sheduler"] = self.args.sheduler_name
        if isinstance(self.args.sheduler_wrapper, list):
            metadata["sheduler params"] = [
                [s.get_name(), s.get_params()] for s in self.args.sheduler_wrapper]
        elif isinstance(self.args.sheduler_wrapper, ShedulerWrapper):
            metadata["sheduler params"] = self.args.sheduler_wrapper.get_params()
        else:
            metadata["sheduler params"] = None

        if isinstance(self.args.sheduler_wrapper, list):
            default_flow_style = None
        else:
            default_flow_style = False
        #print(os.path.join(self.args.results_root, "metadata.yaml"))
        with open(os.path.join(self.args.results_root, "metadata.yaml"), "w") as f:
            yaml.dump(metadata, f, default_flow_style=default_flow_style)

        def _export_histories(history: list, file_path: str):
            h = np.array(history)
            file_path = os.path.join(self.args.results_root, file_path+".npy")
            np.save(file_path, h)
        _export_histories(self.train_loss_history, "train_loss_history")
        _export_histories(self.valid_loss_history, "valid_loss_history")
        _export_histories(self.accuracy_history, "accuracy_history")
        _export_histories(self.uar_history, "uar_history")
        _export_histories(self.f1_history, "f1_history")

    def run(self):
        """Run the Configuration
        """
        print("Running NeuralBench with Configuration:")
        print("device:\t\t", self.args.device if isinstance(
            self.args.device, str) else torch.cuda.get_device_name(self.args.device))
        print("state:\t\t", self.args.state)
        print("approach:\t", self.args.approach)
        print("pretrained:\t", self.args.pretrained)
        print("category:\t", self.args.category)
        print("dataset:\t", self.args.dataset)
        print("batch_size:\t", self.args.batch_size)
        print("epochs:\t\t", self.args.epochs)
        print("learning_rate:\t", self.args.learning_rate)
        print("seed:\t\t", self.args.seed)
        print("optimizer:\t", self.args.optimizer_name)
        print("exclude cities:\t", " ".join(self.args.exclude_cities))
        if not self.args.sheduler_wrapper == None:
            print("sheduler:\t", self.args.sheduler_name)
        print("")
        self.accuracy_history, self.uar_history, self.f1_history, self.train_loss_history, self.valid_loss_history = run_training(
            self.args)
        # print("Accuracy history!!!!")
        # print(self.accuracy_history)
        if self.accuracy_history != []:
            self.export_metadata()
            self.plot_accuracy()
        return self.accuracy_history, self.uar_history, self.f1_history, self.train_loss_history, self.valid_loss_history, self.run_name


class GridSearchModule:
    def __init__(self,
                 data_root: str,
                 device: str,
                 features: str,
                 feature_dir: str,
                 pretrained_dir: str = "",
                 results_path: str = None,
                 custom_feature_path: str = None,
                 state: str = None,
                 approach: List[ModelEnum] = [ModelEnum.CNN10],
                 category: List[DataEnum] = [DataEnum.NONE],
                 dataset: List[str] = ["DCASE2020"],
                 batch_size: List[int] = [32],
                 epochs: List[int] = [50],
                 learning_rate: List[float] = [0.001],
                 seed: List[int] = 0,
                 pretrained: List[bool] = [False],
                 optimizer: List[Union[OptimizerEnum, OptimizerWrapper, GDTUOWrapper]] = [
                     OptimizerEnum.SGD],
                 sheduler_wrapper: List[Union[ShedulerWrapper,
                                              List[ShedulerWrapper]]] = None,
                 exclude_cities: List[List[str]] = [None],
                 base_folder: str = None,
                 disable_progress_bar=False,
                 num_gpus: int = 1,
                 ) -> None:
        """Grid Search of NeuralBench over all possible permutations.

        Args:
            data_root (str): Path data has been extracted.
            device (str): CUDA-enabled device to use for training.
            features (str): Path to features.
            results_path (str, optional): Path where results are to be stored, if None the name is generated automatically. Defaults to None.
            custom_feature_path (str, optional): Custom .npy location of features. Defaults to None.
            state (str, optional): _description_. Optional initial state file path. Defaults to None.
            approach (List[ModelEnum], optional): List of model to be used for grid search. Defaults to [ModelEnum.CNN10].
            category (List[DataEnum], optional): List of data categories to be used for grid search. Defaults to [DataEnum.NONE].
            dataset (List[str], optional): List of datasets to be used for grid search. Defaults to ["DCASE2020"].
            batch_size (List[int], optional): List of batch sizes to be used for grid search. Defaults to [32].
            epochs (List[int], optional): List of epochs to be used for grid search. Defaults to [50].
            learning_rate (List[float], optional): List of learning rates to be used for grid search. Defaults to [0.001].
            seed (List[int], optional): List of seeds to be used for grid search. Defaults to 0.
            optimizer (list[Union[OptimizerEnum, OptimizerWrapper, GDTUOWrapper]], optional): List of optimizers to be used for grid search. Defaults to [OptimizerEnum.SGD].
            sheduler_wrapper (List[Union[ShedulerWrapper, List[ShedulerWrapper]]], optional): List of wrapped shedulers or
            list of list of Shedulers to be used for grid search. Defaults to None.
            exclude_cities (List[List[str]], optional): List of List of cities to exclude from training data. Defaults to [None].
            base_folder (str, optional): Base folder for "data_root", "run_name", "features" and "custom_feature_path". Defaults to None.
            disable_progress_bar (bool, optional): Disable tqdm progress bar while training. Defaults to False.
            num_gpus (int, optional): Number of parallel GPUs to be used. Defaults to 1.
        """

        self.data_root = data_root
        self.device = device
        self.features = features
        self.pretrained_dir = pretrained_dir
        self.results_path = results_path
        self.custom_feature_path = custom_feature_path
        self.state = state
        self.approach = approach  # * grid
        self.category = category  # * grid
        self.dataset = dataset # *grid
        self.batch_size = batch_size  # * grid
        self.epochs = epochs  # * grid
        self.learning_rate = learning_rate  # * grid
        self.seed = seed  # * grid
        self.optimizer = optimizer  # * grid
        self.pretrained = pretrained # * grid
        print("Grid search module")
        print(pretrained)
        self.sheduler_wrapper = sheduler_wrapper  # * grid
        self.exclude_cities = exclude_cities  # * grid
        self.base_folder = base_folder
        self.disable_progress_bar = disable_progress_bar
        self.grid = [
            self.dataset, self.category, self.approach, self.batch_size,
            self.epochs, self.learning_rate, self.seed,
            self.optimizer, self.pretrained, self.sheduler_wrapper, exclude_cities
        ]
        print(self.grid)

        os.makedirs(os.path.join(self.base_folder,
                    self.results_path), exist_ok=True)

        self.accuracy_history = []
        self.uar_history = []
        self.f1_history = []
        self.train_loss_history = []
        self.valid_loss_history = []
        self.run_name_history = []
        self.permutations = None
        self.num_gpus = num_gpus

    def generate_permutations(self):
        self.permutations = list(product(*self.grid))

    def exclude_permutations(self, exclude_list: List[ExcludeSearch]):
        if self.permutations == None:
            self.generate_permutations()
        new_permutations = self.permutations
        for exclude in exclude_list:
            new_permutations = exclude.match(new_permutations)
        self.permutations = new_permutations

    def plot_runs(self,
                  plot_type="accuracy",
                  title: str = "Grid Search",
                  file_name: str = None,
                  std_scale=0.1,
                  plot_all=True,
                  overwrite_outpath=None
                  ):
        assert plot_type in ["accuracy", "uar",
                             "f1", "train_loss", "valid_loss"]
        if file_name == None:
            file_name = plot_type

        if title == "Grid Search":
            title += " "+plot_type.upper()

        if plot_type == "accuracy":
            accuracy_history = self.accuracy_history
        elif plot_type == "uar":
            accuracy_history = self.uar_history
        elif plot_type == "f1":
            accuracy_history = self.f1_history
        elif plot_type == "train_loss":
            accuracy_history = self.train_loss_history
        else:
            accuracy_history = self.valid_loss_history

        if plot_all:
            fig, acc_plot, _, lw = generate_plot_base(
                title, plot_type, loss=False)
            colors = matplotlib.colormaps["rainbow"](
                np.linspace(0, 1, len(self.run_name_history))) * 0.8
            for run, history, color in zip(self.run_name_history, accuracy_history, colors):
                plt.plot(range(1, len(history)+1),
                         history, c=color, label=run, linewidth=lw)
            acc_plot.legend(prop={"size": 10})
            outpath = os.path.join(
                self.base_folder, self.results_path, file_name+"_all")
            if overwrite_outpath is not None:
                outpath = overwrite_outpath+"_all"
            fig.savefig(outpath+".png", transparent=False, dpi=100,
                        bbox_inches="tight", pad_inches=1)
            plt.clf()

        fig, acc_plot, _, lw = generate_plot_base(
            title, plot_type, loss=False)

        def group_runs(run_names, data_list):
            identical_groups = {}
            for i, run_name in enumerate(run_names):
                parts = run_name.split("_")
                seed = parts[-2]
                mx = run_name.replace("_"+seed, "")
                if mx not in identical_groups:
                    identical_groups[mx] = []
                identical_groups[mx].append(
                    (run_name, data_list[i]))
            return list(identical_groups.values())

        runs = group_runs(self.run_name_history, accuracy_history)
        colors = matplotlib.colormaps["rainbow"](
            np.linspace(0, 1, len(runs))) * 0.8

        for run, color in zip(runs, colors):
            run_names, histories = zip(*run)
            run_name = run_names[0].replace(
                "_"+run_names[0].split("_")[-2], "")
            # print("-"*50)
            # print("Histories")
            # print(histories)
            run_name = "_".join(
                list(filter(lambda a: a != "None", run_name.split("_"))))
            # print(histories)
            y_mean = np.mean(histories, axis=0)
            y_std = np.std(histories, axis=0) * std_scale
            plt.fill_between(range(1, len(histories[0])+1), y_mean-y_std, y_mean+y_std,
                             color=color, alpha=0.2)
            plt.plot(range(1, len(histories[0])+1),
                     y_mean, c=color, label=run_name, linewidth=lw)
        acc_plot.legend(prop={"size": 10})
        outpath = os.path.join(
            self.base_folder, self.results_path, file_name)
        if overwrite_outpath is not None:
            outpath = overwrite_outpath
        fig.savefig(outpath+".png", transparent=False, dpi=100,
                    bbox_inches="tight", pad_inches=1)
        plt.clf()

    def export_metadata(self, best_type="accuaracy"):
        best_run_name, max_accuracy, max_uar, max_f1, min_valid_loss = self.get_best_run(
            best_type)
        metadata = {
            "total runs": len(self.permutations),
            "best run": best_run_name,
            "min valid loss": min_valid_loss,
            "max valid acc": max_accuracy,
            "max valid uar": max_uar,
            "max valid f1": max_f1,
            "device": self.device,
            "state": self.state,
            "approach": self.approach,
            "category": [c.value for c in self.category],
            "dataset": self.dataset,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "pretrained": self.pretrained,
            "seed": self.seed,
            "exclude_cities": self.exclude_cities,
        }

        optimizer_names = []
        optimizer_params = []
        for o in self.optimizer:
            if isinstance(o, OptimizerEnum):
                optimizer_names.append(o.value)
            else:
                optimizer_names.append(o.get_name())
                optimizer_params.append(o.get_params())
        metadata["optimizer"] = optimizer_names
        metadata["optimizer params"] = optimizer_params

        sheduler_names = []
        sheduler_parameters = []
        for sheduler in self.sheduler_wrapper:
            if isinstance(sheduler, list):
                sheduler_name = "-".join([s.get_name() for s in sheduler])
            elif isinstance(sheduler, ShedulerWrapper):
                sheduler_name = sheduler.get_name()
            else:
                sheduler_name = None

            if isinstance(sheduler, list):
                sheduler_params = [
                    [s.get_name(), s.get_params()] for s in sheduler]
            elif isinstance(sheduler, ShedulerWrapper):
                sheduler_params = sheduler.get_params()
            else:
                sheduler_params = None

            sheduler_names.append(sheduler_name)
            sheduler_parameters.append(sheduler_params)

        metadata["sheduler"] = sheduler_names
        metadata["sheduler params"] = sheduler_parameters
        outpath = os.path.join(
            self.base_folder, self.results_path, "metadata.yaml")
        with open(outpath, "w") as f:
            yaml.dump(metadata, f, default_flow_style=None)

    def get_best_run(self, best_type="accuaracy"):
        assert best_type in ["accuaracy", "uar", "f1", "loss"]
        best_fn = np.argmax
        print(self.accuracy_history)
        if best_type == "loss":
            best_fn = np.argmin
            history = self.valid_loss_history
        elif best_type == "accuaracy":
            history = self.accuracy_history
        elif best_type == "uar":
            history = self.uar_history
        else:
            history = self.f1_history
        best_index = np.unravel_index(best_fn(history), np.shape(history))[0]
        best_run_name = str(self.run_name_history[best_index])
        max_accuracy = float(np.max(self.accuracy_history[best_index]))
        max_uar = float(np.max(self.uar_history[best_index]))
        max_f1 = float(np.max(self.f1_history[best_index]))
        min_valid_loss = float(np.min(self.valid_loss_history[best_index]))
        return best_run_name, max_accuracy, max_uar, max_f1, min_valid_loss

    def _run_single(self):
        run_count = 1
        for experiment in self.permutations:
            print("Running experiment", str(run_count) +
                  "/" + str(len(self.permutations)))
            print("")
            run = NeuralBench(
                data_root=self.data_root,
                device=self.device+":0",
                run_name=None,
                results_path=self.results_path,
                features=self.features,
                pretrained_dir=self.pretrained_dir,
                custom_feature_path=self.custom_feature_path,
                state=self.state,
                dataset=experiment[0],
                category=experiment[1],
                approach=experiment[2],
                batch_size=experiment[3],
                epochs=experiment[4],
                learning_rate=experiment[5],
                seed=experiment[6],
                optimizer=experiment[7],
                pretrained=experiment[8],
                sheduler_wrapper=experiment[9],
                exclude_cities=experiment[10],
                base_folder=self.base_folder,
                disable_progress_bar=self.disable_progress_bar,
            )
            # TODO: for the end; Get back the try except block 
            accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history, run_name_history = run.run()
            self.accuracy_history.append(accuracy_history)
            self.uar_history.append(uar_history)
            self.f1_history.append(f1_history)
            self.train_loss_history.append(train_loss_history)
            self.valid_loss_history.append(valid_loss_history)
            self.run_name_history.append(run_name_history)
            del accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history, run_name_history
            # try:
            #     accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history, run_name_history = run.run()
            #     self.accuracy_history.append(accuracy_history)
            #     self.uar_history.append(uar_history)
            #     self.f1_history.append(f1_history)
            #     self.train_loss_history.append(train_loss_history)
            #     self.valid_loss_history.append(valid_loss_history)
            #     self.run_name_history.append(run_name_history)
            #     del accuracy_history, uar_history, f1_history, train_loss_history, valid_loss_history, run_name_history
            # except torch.cuda.OutOfMemoryError:
            #     log_str = f"\nCUDA: Out of Memory in run {run_count}/{str(len(self.permutations))} runs.\nTimestamp:"
            #     logging.log(level=logging.CRITICAL, msg=log_str)
            # except BaseException as e:
            #     log_str = f"\nException in run {run_count}/{str(len(self.permutations))} runs.\nError: {e}\nTimestamp:"
            #     logging.log(level=logging.CRITICAL, msg=log_str)
            del run
            gc.collect()
            torch.cuda.empty_cache()
            run_count += 1

    def run(self):
        """Run the Grid Search for all possible permutations.
        """
        grid_search_timer = Timer()
        grid_search_timer.start()
        if self.permutations == None:
            self.generate_permutations()
        print("Grid Search Module")
        print("Running", len(self.permutations), "total experiments.")
        print("")

        if self.num_gpus > 1:
            cuda_device_count = torch.cuda.device_count()
            assert self.num_gpus <= torch.cuda.device_count()
            torch.multiprocessing.set_start_method('spawn')
            global_queue = GlobalQueueActor()
            global_queue.set_permutation(self.permutations)
            processes = []
            for actor_num in range(self.num_gpus):
                a: ParallelActor = ParallelActor(
                    global_queue,
                    actor_num,
                    self.data_root,
                    torch.cuda.device(actor_num),
                    None,
                    self.results_path,
                    self.features,
                    self.feature_dir,
                    self.pretrained_dir,
                    self.pretrained,
                    self.custom_feature_path,
                    self.state,
                    self.base_folder,
                    self.disable_progress_bar
                )
                processes.append(Process(target=a.run_parallel))

            [p.start() for p in processes]
            [p.join() for p in processes]
            self.accuracy_history, self.uar_history, self.f1_history, self.train_loss_history, self.valid_loss_history, self.run_name_history = global_queue.get_metrics()

        else:
            self._run_single()
        grid_search_timer.stop()
        log_str = f"\nFinished Grid Search.\nTotal Time: {grid_search_timer.pretty_time()}"
        logging.log(level=logging.CRITICAL, msg=log_str)


class PostProcessing:
    def __init__(self, gridsearch: GridSearchModule, export_path: str, use_base_folder: bool = True) -> None:
        self.gridsearch = gridsearch
        if use_base_folder:
            self.export_path = os.path.join(
                self.gridsearch.base_folder, export_path)
        else:
            self.export_path = export_path
        os.makedirs(self.export_path, exist_ok=True)
        runs = {key: key.split("_")
                for key in os.listdir(
                    os.path.join(self.gridsearch.base_folder,
                                 self.gridsearch.results_path)
        )
            if os.path.isdir(
                    os.path.join(self.gridsearch.base_folder,
                                 self.gridsearch.results_path, key)
        )}

        df = pd.DataFrame.from_dict(
            runs, orient="index")
        df.reset_index(inplace=True)
        # fix for runs that did not include exclude_cities yet
        if len(df.columns) == 9:
            df[8] = "None"
        df.rename(columns={
            "index": "name",
            0: "approach",
            1: "category",
            2: "optimizer",
            3: "learning_rate",
            4: "batch_size",
            5: "epochs",
            6: "pretrained",
            7: "seed",
            8: "sheduler_wrapper",
            9: "exclude_cities",
        }, inplace=True)

        df = df.astype({
            "name": "string",
            "approach": "string",
            "category": "string",
            "optimizer": "string",
            "learning_rate": "string",
            "batch_size": "string",
            "epochs": "string",
            "pretrained": "string",
            "seed": "string",
            "sheduler_wrapper": "string",
            "exclude_cities": "string",
        })
        self.runs_history = df
        self.df = df

    def export_csv(self, args, export_name: str = "grid", acc_type: str = "accuracy"):
        self.runs_history = self.df
        disagg_accuracies = ["all", "indoor", "transportation", "outdoor", "barcelona", "helsinki", "lisbon", "london", "lyon", "milan", "paris", "prague", "stockholm", "vienna", "a", "b",
                             "c", "s1", "s2", "s3", "s4", "s5", "s6"]
        assert acc_type in ["accuracy", "uar",
                            "f1", "train_loss", "valid_loss"]
        yaml_acc_type = self._convert_to_yaml_names(acc_type)
        accs = {k: [] for k in disagg_accuracies}
        mtt = []
        mvt = []
        # print("Runns overview!")
        # print(self.runs_history["name"].tolist())
        for run in self.runs_history["name"].tolist():
            "grid search folder"
            # print(self.gridsearch.base_folder)
            # print(self.gridsearch.results_path)
            # print(run)

            _base = os.path.join(
                self.gridsearch.base_folder,
                self.gridsearch.results_path,
                run
            )
            if args.dataset == "DCASE2020":
                with open(os.path.join(_base, "test_holistic.yaml"), "r") as f:
                    run_accs = yaml.safe_load(f)
                run_accs = run_accs[yaml_acc_type]
                for acc in disagg_accuracies:
                    accs[acc].append(run_accs[acc])
            with open(os.path.join(_base, "metadata.yaml"), "r") as f:
                metadata = yaml.safe_load(f)
            mtt.append(metadata["mean train time"])
            mvt.append(metadata["mean valid time"])
        if args.dataset == "DCASE2020":
            for acc in disagg_accuracies:
                self.runs_history[acc] = accs[acc]
        self.runs_history["mtt"] = mtt
        self.runs_history["mvt"] = mvt
        _export_path = os.path.join(
            self.export_path, export_name+"_"+acc_type+".csv")
        self.runs_history.to_csv(_export_path, index=False)

    def create_results(self, export_name: str = "results"):
        for run in self.runs_history["name"]:
            _base = os.path.join(
                self.gridsearch.base_folder,
                self.gridsearch.results_path,
                run
            )
            _f = os.path.join(
                _base,
                "dev.yaml"
            )
            with open(_f, "r") as f:
                epoch = yaml.safe_load(f)["Epoch"]
            src = os.path.join(
                _base,
                "Epoch_"+str(epoch),
                "dev.csv"
            )
            dst = os.path.join(
                _base,
                export_name+".csv"
            )
            shutil.copyfile(src, dst)

    def group_and_plot(self,
                       group: str,
                       accuracy_types: list = [
                           "accuracy", "uar", "f1", "train_loss", "valid_loss"],
                       std_scale=0.1,
                       plot_all=True):
        self.runs_history = self.df
        groups = self.runs_history.groupby(group)["name"].apply(list).to_dict()
        _export_group_path = os.path.join(self.export_path, group)
        os.makedirs(_export_group_path, exist_ok=True)
        for k, v in groups.items():
            self._load_history(v)
            for acc_type in accuracy_types:
                title = group+": "+str(k)+" | " + acc_type
                file_name = os.path.join(
                    _export_group_path, group+"_"+str(k)+"_"+acc_type)
                self.gridsearch.plot_runs(
                    acc_type, title, None, std_scale, plot_all, overwrite_outpath=file_name)

    def _convert_yaml_names(self, name):
        if name == "ACC":
            return "accuracy"
        elif name == "UAR":
            return "uar"
        elif name == "F1":
            return "f1"
        elif name == "train_loss":
            return "train_loss"
        elif name == "val_loss":
            return "valid_loss"
        return "Epoch"

    def _convert_to_yaml_names(self, name):
        if name == "accuracy":
            return "ACC"
        elif name == "uar":
            return "UAR"
        elif name == "f1":
            return "F1"
        elif name == "train_loss":
            return "train_loss"
        elif name == "valid_loss":
            return "val_loss"

    def plot_best(self,
                  group: str,
                  accuracy_types: list = [
                      "accuracy", "uar", "f1", "train_loss", "valid_loss"],
                  std_scale=0.1
                  ):
        self.runs_history = self.df
        base_accuracies = [
            "accuracy", "uar", "f1", "train_loss", "valid_loss"]
        accs = {k: [] for k in base_accuracies}
        for run in self.runs_history["name"].tolist():
            _f = os.path.join(
                self.gridsearch.base_folder,
                self.gridsearch.results_path,
                run,
                "dev.yaml"
            )
            with open(_f, "r") as f:
                run_accs = yaml.safe_load(f)
            run_accs = {self._convert_yaml_names(
                k): v for k, v in run_accs.items()}
            for acc in base_accuracies:
                accs[acc].append(run_accs[acc])

        for acc_type in accuracy_types:
            self.runs_history[acc_type] = accs[acc_type]

        _export_group_path = os.path.join(self.export_path, "best_"+group)
        os.makedirs(_export_group_path, exist_ok=True)

        groups = self.runs_history.groupby(group)
        for acc_type in accuracy_types:
            idxs = groups.apply(
                lambda x: x[acc_type].idxmax()
                if acc_type in ["accuracy", "uar", "f1"]
                else x[acc_type].idxmin()
            )
            _runs = self.runs_history.loc[idxs]["name"].tolist()
            self._load_history(_runs)

            title = "Best: "+group + " | " + acc_type
            file_name = os.path.join(
                _export_group_path, "best_"+group+"_"+acc_type)
            self.gridsearch.plot_runs(
                acc_type, title, None, std_scale, False, overwrite_outpath=file_name)

    def _import_histories(self, file_path: str):
        grid_base_folder = os.path.join(self.gridsearch.base_folder,
                                        self.gridsearch.results_path)
        file_path = os.path.join(grid_base_folder, file_path+".npy")
        return np.load(file_path).tolist()

    def _load_history(self, runs: list):

        accuracy_history = []
        uar_history = []
        f1_history = []
        train_loss_history = []
        valid_loss_history = []
        run_name_history = []
        for run_name in runs:
            accuracy_history.append(self._import_histories(
                os.path.join(run_name, "accuracy_history")))
            uar_history.append(self._import_histories(
                os.path.join(run_name, "uar_history")))
            f1_history.append(self._import_histories(
                os.path.join(run_name, "f1_history")))
            train_loss_history.append(self._import_histories(
                os.path.join(run_name, "train_loss_history")))
            valid_loss_history.append(self._import_histories(
                os.path.join(run_name, "valid_loss_history")))
            run_name_history.append(run_name)
        self.gridsearch.accuracy_history = accuracy_history
        self.gridsearch.uar_history = uar_history
        self.gridsearch.f1_history = f1_history
        self.gridsearch.train_loss_history = train_loss_history
        self.gridsearch.valid_loss_history = valid_loss_history
        self.gridsearch.run_name_history = run_name_history
