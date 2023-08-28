from neuralbench import NeuralBench, DataEnum, ModelEnum, OptimizerEnum, ShedulerWrapper, PostProcessing
from neuralbench import GridSearchModule, OptimizerWrapper, ExcludeSearch, GDTUOWrapper, SAMWrapper
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD
#from sam import SAM
from KFACPytorch import KFACOptimizer
from gradient_descent_the_ultimate_optimizer import gdtuo
from runcopy import RunCopy, DeleteStates
import argparse

def bench_setup_grid_1(args):
    kfac_optim = OptimizerWrapper(KFACOptimizer, solver="not_symeig")
    gdtuo_optim1 = GDTUOWrapper([
        (gdtuo.Adam, {}, True),
        (gdtuo.Adam, {"alpha": 1e-5}, False),
    ])
    optimizers = []
    if "SGD" in args.optimizer:
        sgd_optim = OptimizerWrapper(SGD, momentum=0.9)
        optimizers.append(sgd_optim)
    if "Adam" in args.optimizer:
        adam_optim = OptimizerWrapper(Adam)
        optimizers.append(adam_optim)
    if "SAM" in args.optimizer:
        sam_optim = SAMWrapper(SGD, momentum=0.9)
        optimizers.append(sam_optim)
    #sgd_optim2 = OptimizerWrapper(SGD, momentum=0.9)
    
    print("main")
    print(args.pretrained)
    grid1 = GridSearchModule(
        #data_root="metadata",
        #data_root=args.data_root + "metadata",
        data_root=args.data_root,
        #data_root="metadata",
        #data_root="",
        device=args.device,
        results_path=args.results_root,
        #features="mel_spectrograms/features.csv",
        features = args.features,
        feature_dir=args.feature_dir,
        pretrained_dir=args.pretrained_dir,
        custom_feature_path=args.feature_dir,
        state=None,
        # TODO: Change approach to commandline! --> why do we need ModelEnum?

        approach=args.approach,
        category=[DataEnum.NONE],
        # batch_size=[16, 32],
        # epochs=[50],
        # learning_rate=[1e-3, 1e-4, 1e-5, 1e-6],
        # seed=[42, 43, 44],
        # optimizer=[kfac_optim, gdtuo_optim1, adam_optim, sgd_optim],
        # batch_size=[32],
        # seed=[42],
        # #epochs=[1],
        # epochs=[100],
        # learning_rate=[1e-3, 5e-4],
        dataset=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        optimizer=optimizers,
        pretrained=args.pretrained,
        #optimizer=[sam],
        sheduler_wrapper=[None],
        #base_folder=args.data_root,
        base_folder="",
        disable_progress_bar=True,
        num_gpus=args.num_gpus
    )

    grid1.exclude_permutations([
        ExcludeSearch(
            batch_size=32,
            optimizer=kfac_optim
        ),
        ExcludeSearch(
            optimizer=gdtuo_optim1,
            sheduler_wrapper="all"
        ),
    ])

    #grid1._run_single()
    grid1.run()
    print("Grid search")
    print(grid1.accuracy_history)
    def is_only_empty_lists(variable):
        if isinstance(variable, list):
            if len(variable) == 0:  # Empty list
                return True
            else:
                return all(is_only_empty_lists(item) for item in variable)
        else:
            return False


    if not is_only_empty_lists(grid1.accuracy_history):
    # if grid1.accuracy_history != []:
        grid1.export_metadata()
        grid1.plot_runs(plot_type="accuracy")
        grid1.plot_runs(plot_type="uar")
        grid1.plot_runs(plot_type="f1")
        grid1.plot_runs(plot_type="train_loss")
        grid1.plot_runs(plot_type="valid_loss")
        postprocessor = PostProcessing(
            gridsearch=grid1, export_path="postprocessing_grid_1"
        )
        postprocessor.export_csv(args, export_name="grid_1")
    # postprocessor.group_and_plot(
    #     group="optimizer"
    # )
    # postprocessor.group_and_plot(
    #     group="learning_rate"
    # )
    # postprocessor.plot_best(
    #     group="optimizer"
    # )
    # postprocessor.plot_best(
    #     group="learning_rate"
    # )
    
    # postprocessor.create_results(export_name="results")


# def bench_setup_grid_2():
#     adam_optim = OptimizerWrapper(Adam)
#     sgd_optim = OptimizerWrapper(SGD, momentum=0.9)

#     grid2 = GridSearchModule(
#         data_root="metadata",
#         device="cuda",
#         results_path="results_grid_2",
#         features="mel_spectrograms/features.csv",
#         custom_feature_path="mel_spectrograms",
#         state=None,
#         approach=[ModelEnum.CNN14],
#         category=[DataEnum.NONE],
#         batch_size=[16, 32],
#         epochs=[50],
#         learning_rate=[1e-3, 1e-4, 1e-5, 1e-6],
#         seed=[42, 43, 44],
#         optimizer=[adam_optim, sgd_optim],
#         sheduler_wrapper=[None],
#         base_folder="/nas/student/SimonRampp/DCASE2020",
#         disable_progress_bar=True,
#         num_gpus=4
#     )

#     # grid2.run()
#     # grid2.export_metadata()
#     # grid2.plot_runs(plot_type="accuracy")
#     # grid2.plot_runs(plot_type="uar")
#     # grid2.plot_runs(plot_type="f1")
#     # grid2.plot_runs(plot_type="train_loss")
#     # grid2.plot_runs(plot_type="valid_loss")
#     postprocessor = PostProcessing(
#         gridsearch=grid2, export_path="postprocessing_grid_2"
#     )
#     # postprocessor.group_and_plot(
#     #     group="optimizer"
#     # )
#     # postprocessor.group_and_plot(
#     #     group="learning_rate"
#     # )
#     # postprocessor.plot_best(
#     #     group="optimizer"
#     # )
#     # postprocessor.plot_best(
#     #     group="learning_rate"
#     # )
#     postprocessor.export_csv(export_name="grid_2")
#     # postprocessor.create_results(export_name="results")


# def bench_setup_grid_3():
#     adam_optim = OptimizerWrapper(Adam)

#     grid3 = GridSearchModule(
#         data_root="metadata",
#         device="cuda",
#         results_path="results_grid_3",
#         features="mel_spectrograms/features.csv",
#         custom_feature_path="mel_spectrograms",
#         state=None,
#         approach=[ModelEnum.CNN10, ModelEnum.CNN14],
#         category=[DataEnum.NONE],
#         batch_size=[16],
#         epochs=[50],
#         learning_rate=[1e-4],
#         seed=[42, 43, 44],
#         optimizer=[adam_optim],
#         sheduler_wrapper=[None],
#         exclude_cities=[
#             ["barcelona"],
#             ["helsinki"],
#             ["lisbon"],
#             ["london"],
#             ["lyon"],
#             ["milan"],
#             ["paris"],
#             ["prague"],
#             ["stockholm"],
#             ["vienna"]
#         ],
#         base_folder="/nas/student/SimonRampp/DCASE2020",
#         disable_progress_bar=True,
#         num_gpus=4
#     )
#     # grid3.run()
#     # grid3.export_metadata()
#     # grid3.plot_runs(plot_type="accuracy")
#     # grid3.plot_runs(plot_type="uar")
#     # grid3.plot_runs(plot_type="f1")
#     # grid3.plot_runs(plot_type="train_loss")
#     # grid3.plot_runs(plot_type="valid_loss")

#     postprocessor = PostProcessing(
#         gridsearch=grid3, export_path="postprocessing_grid_3"
#     )
#     # postprocessor.group_and_plot(
#     #     group="optimizer"
#     # )
#     # postprocessor.group_and_plot(
#     #     group="learning_rate"
#     # )
#     # postprocessor.plot_best(
#     #     group="optimizer"
#     # )
#     # postprocessor.plot_best(
#     #     group="learning_rate"
#     # )
#     postprocessor.export_csv(export_name="grid_3")
#     # postprocessor.create_results(export_name="results")


# def post_run_copy():
#     rc_q1 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_1",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ1_Generalisation"
#     )
#     rc_q1.copy([
#         "cnn10_None_Adam_0-0001_16_50_44_None",
#         "cnn10_None_Adam_0-0001_32_50_44_None",
#         "cnn10_None_SGD_0-001_16_50_42_None",
#         "cnn10_None_SGD_0-001_32_50_43_None"
#     ], files=["results.csv"])

#     rc_q1_2 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_2",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ1_Generalisation"
#     )
#     rc_q1_2.copy([
#         "cnn14_None_Adam_0-0001_16_50_44_None",
#         "cnn14_None_Adam_0-001_32_50_42_None",
#         "cnn14_None_SGD_0-001_16_50_42_None",
#         "cnn14_None_SGD_0-001_32_50_44_None"
#     ], files=["results.csv"])

#     rc_q2 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_1",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ2_2ndOrderOptimizer"
#     )
#     rc_q2.copy([
#         "cnn10_None_Adam_0-0001_16_50_42_None",
#         "cnn10_None_SGD_0-001_16_50_42_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-001_16_50_42_None",
#         "cnn10_None_KFACOptimizer_1e-05_16_50_42_None",
#         "cnn10_None_Adam_0-0001_16_50_43_None",
#         "cnn10_None_SGD_0-001_16_50_43_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-0001_16_50_43_None",
#         "cnn10_None_KFACOptimizer_1e-05_16_50_43_None",
#         "cnn10_None_Adam_0-0001_16_50_44_None",
#         "cnn10_None_SGD_0-001_16_50_44_None",
#         "cnn10_None_GDTUO-Adam-Adam_1e-05_16_50_44_None",
#         "cnn10_None_KFACOptimizer_1e-05_16_50_44_None"
#     ], files=["results.csv"])

#     rc_q3 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_1",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ3_UnseenDevices"
#     )
#     rc_q3.copy([
#         "cnn10_None_SGD_0-001_16_50_42_None",
#         "cnn10_None_Adam_0-0001_16_50_44_None"
#     ], files=["results.csv"])

#     rc_q3_2 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_2",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ3_UnseenDevices"
#     )
#     rc_q3_2.copy([
#         "cnn14_None_SGD_0-001_16_50_42_None",
#         "cnn14_None_Adam_0-0001_16_50_44_None"
#     ], files=["results.csv"])

#     rc_q4 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_3",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ4_UnseenCity"
#     )
#     rc_q4.copy([
#         "cnn10_None_Adam_0-0001_16_50_43_None_barcelona",
#         "cnn10_None_Adam_0-0001_16_50_44_None_helsinki",
#         "cnn10_None_Adam_0-0001_16_50_42_None_lisbon",
#         "cnn10_None_Adam_0-0001_16_50_44_None_london",
#         "cnn10_None_Adam_0-0001_16_50_44_None_lyon",
#         "cnn10_None_Adam_0-0001_16_50_43_None_milan",
#         "cnn10_None_Adam_0-0001_16_50_43_None_paris",
#         "cnn10_None_Adam_0-0001_16_50_44_None_prague",
#         "cnn10_None_Adam_0-0001_16_50_43_None_stockholm",
#         "cnn10_None_Adam_0-0001_16_50_42_None_vienna",
#         "cnn14_None_Adam_0-0001_16_50_42_None_barcelona",
#         "cnn14_None_Adam_0-0001_16_50_42_None_helsinki",
#         "cnn14_None_Adam_0-0001_16_50_43_None_lisbon",
#         "cnn14_None_Adam_0-0001_16_50_42_None_london",
#         "cnn14_None_Adam_0-0001_16_50_42_None_lyon",
#         "cnn14_None_Adam_0-0001_16_50_43_None_milan",
#         "cnn14_None_Adam_0-0001_16_50_43_None_paris",
#         "cnn14_None_Adam_0-0001_16_50_42_None_prague",
#         "cnn14_None_Adam_0-0001_16_50_43_None_stockholm",
#         "cnn14_None_Adam_0-0001_16_50_44_None_vienna"
#     ], files=["results.csv"])


# def post_run_copy_q5():
#     rc_q5 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_1",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ5_Complete"
#     )
#     rc_q5_2 = RunCopy(
#         source_folder="/nas/student/SimonRampp/DCASE2020/results_grid_2",
#         destination_folder="/nas/student/SimonRampp/models_for_interspeech_paper/RQ5_Complete"
#     )

#     rc_q5.copy([
#         "cnn10_None_Adam_0-001_16_50_42_None",
#         "cnn10_None_Adam_0-0001_16_50_42_None",
#         "cnn10_None_Adam_0-001_32_50_42_None",
#         "cnn10_None_Adam_0-0001_32_50_42_None",
#         "cnn10_None_Adam_0-001_16_50_43_None",
#         "cnn10_None_Adam_0-0001_16_50_43_None",
#         "cnn10_None_Adam_0-001_32_50_43_None",
#         "cnn10_None_Adam_0-0001_32_50_43_None",
#         "cnn10_None_SGD_0-001_16_50_42_None",
#         "cnn10_None_SGD_0-0001_16_50_42_None",
#         "cnn10_None_SGD_0-001_32_50_42_None",
#         "cnn10_None_SGD_0-0001_32_50_42_None",
#         "cnn10_None_SGD_0-001_16_50_43_None",
#         "cnn10_None_SGD_0-0001_16_50_43_None",
#         "cnn10_None_SGD_0-001_32_50_43_None",
#         "cnn10_None_SGD_0-0001_32_50_43_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-001_16_50_42_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-0001_16_50_42_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-001_32_50_42_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-0001_32_50_42_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-001_16_50_43_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-0001_16_50_43_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-001_32_50_43_None",
#         "cnn10_None_GDTUO-Adam-Adam_0-0001_32_50_43_None",
#         "cnn10_None_KFACOptimizer_0-001_16_50_42_None",
#         "cnn10_None_KFACOptimizer_0-0001_16_50_42_None",
#         "cnn10_None_KFACOptimizer_0-001_16_50_43_None",
#         "cnn10_None_KFACOptimizer_0-0001_16_50_43_None"
#     ], files=["state.pth.tar", "test_holistic.yaml", "results.csv"])

#     rc_q5_2.copy([
#         "cnn14_None_Adam_0-001_16_50_42_None",
#         "cnn14_None_Adam_0-0001_16_50_42_None",
#         "cnn14_None_Adam_0-001_32_50_42_None",
#         "cnn14_None_Adam_0-0001_32_50_42_None",
#         "cnn14_None_Adam_0-001_16_50_43_None",
#         "cnn14_None_Adam_0-0001_16_50_43_None",
#         "cnn14_None_Adam_0-001_32_50_43_None",
#         "cnn14_None_Adam_0-0001_32_50_43_None",
#         "cnn14_None_SGD_0-001_16_50_42_None",
#         "cnn14_None_SGD_0-0001_16_50_42_None",
#         "cnn14_None_SGD_0-001_32_50_42_None",
#         "cnn14_None_SGD_0-0001_32_50_42_None",
#         "cnn14_None_SGD_0-001_16_50_43_None",
#         "cnn14_None_SGD_0-0001_16_50_43_None",
#         "cnn14_None_SGD_0-001_32_50_43_None",
#         "cnn14_None_SGD_0-0001_32_50_43_None"
#     ], files=["state.pth.tar", "test_holistic.yaml", "results.csv"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DCASE-T1 Training')
    parser.add_argument(
        '--data-root',
        help='Path data has been extracted',
        required=True
    )
    parser.add_argument(
        '--results-root',
        help='Path where results are to be stored',
        required=True
    )
    parser.add_argument(
        '--features',
        help='Path to features',
        required=True
    )
    
    parser.add_argument(
        '--device',
        help='CUDA-enabled device to use for training',
        required=True
    )
    parser.add_argument(
        '--state',
        help='Optional initial state'
    )
    parser.add_argument(
        '--approach',
        nargs='+', 
        type=str,
        default=['cnn10'],
        # resNet50 not yet
        choices=[
            'cnn14',
            'cnn10',
            'sincnet',
            'resnet50',
            # add other efficientnet models
            'efficientnet-b0',
            'efficientnet-b4'
        ]
    )
    parser.add_argument(
        '--category',
        default=None,
        choices=[
            'indoor',
            'outdoor',
            'transportation',
            None
        ]
    )
    # parser.add_argument(
    #     '--batch-size',
    #     type=int,
    #     default=32,
    #     help='Batch size'
    # )
    # parser.add_argument(
    #     '--epochs',
    #     type=int,
    #     default=60
    # )
    # parser.add_argument(
    #     '--learning-rate',
    #     type=float,
    #     default=0.001
    # )
    # parser.add_argument(
    #     '--seed',
    #     type=int,
    #     default=0
    # )
    # parser.add_argument(
    #     '--optimizer',
    #     default='SGD'
    # )
    parser.add_argument(
        '--feature_dir',
        default=''
    )
    parser.add_argument(
        '--pretrained_dir',
        default=''
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=0
    )

    parser.add_argument('--batch_size', nargs='+', type=int, default=[32])
    parser.add_argument('--epochs', nargs='+', type=int, default=[50])
    parser.add_argument('--learning_rate', nargs='+', type=float, default=[1e-4])
    parser.add_argument('--seed', nargs='+', type=int, default=[42])
    parser.add_argument('--optimizer', nargs='+', type=str, default=["SGD"])
    parser.add_argument('--dataset', nargs='+', type=str, default=["DCASE"])
    parser.add_argument('--pretrained', nargs='+', type=str, default=["False"])
    def convert_list_to_bool(args):
        out = []
        for arg in args:
            if arg == "True":
                out.append(True)
            elif arg == "False":
                out.append(False)
            else:
                raise argparse.ArgumentTypeError('Boolean value expected (True/False)')
        return out
    # def parse_boolean(arg):
    #     if arg.lower() == 'true':
    #         return True
    #     elif arg.lower() == 'false':
    #         return False
    #     else:
    #         raise argparse.ArgumentTypeError('Boolean value expected (True/False)')

    
    
    # batch_size=[32],
    #     epochs=[50],
    #     learning_rate=[1e-4, 1e-5],
    #     seed=[42],
    #     optimizer=[sgd_optim, sam],

    parser.add_argument(
        '--custom-feature-path',
        help='Custom .npy location of features',
        required=False
    )

    parser.add_argument(
        '--disable-progress-bar',
        default=False,
        type=bool,
        help='Disable tqdm progress bar while training',
        choices=[
            'True',
            'False',
        ]
    )

    parser.add_argument(
        '--exclude-cities',
        default="None",
        type=str,
        help='Exclude a City from training',
        choices=[
            "barcelona",
            "helsinki",
            "lisbon",
            "london",
            "lyon",
            "milan",
            "paris",
            "prague",
            "stockholm",
            "vienna",
            "None",
        ]
    )

    args = parser.parse_args()
    # pretrained = 
    # print(pretrained)
    args.pretrained = convert_list_to_bool(args.pretrained)
    # print(args.pretrained)
    # print(args.approach)
    bench_setup_grid_1(args)
    #bench_setup_grid_2()
    #bench_setup_grid_3()
    # post_run_copy()
    # post_run_copy_q5()
