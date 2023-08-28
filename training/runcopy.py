import shutil
from typing import List
import os


class RunCopy:
    def __init__(self,
                 source_folder: str,
                 destination_folder: str,
                 ) -> None:
        self.source_folder = source_folder
        self.destination_folder = destination_folder

    def copy(self, runs: List[str], files=["state.pth.tar", "test_holistic.yaml"]):
        for run in runs:
            os.makedirs(
                os.path.join(self.destination_folder, run),
                exist_ok=True
            )
            for f in files:
                _src = os.path.join(
                    self.source_folder,
                    run,
                    f
                )
                _dst = os.path.join(
                    self.destination_folder,
                    run,
                    f
                )
                shutil.copyfile(_src, _dst)


class DeleteStates:
    def __init__(self, source_folder: str,) -> None:
        self.source_folder = source_folder

    def delete(self):
        runs = [r for r in
                os.listdir(self.source_folder)
                if os.path.isdir(os.path.join(self.source_folder, r))
                ]
        for run in runs:
            epochs = [
                e for e in
                os.listdir(os.path.join(self.source_folder, run))
                if "Epoch" in e and
                os.path.isdir(os.path.join(self.source_folder, run, e))
            ]
            for epoch in epochs:
                f = os.path.join(self.source_folder, run,
                                 epoch, "state.pth.tar")
                if os.path.exists(f):
                    print("Removed:", f)
                    os.remove(f)
