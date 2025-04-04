import numpy as np
from PIL import Image

map_image_path = r"data\map2.png"
car_image_path = r"data\car.png"

map_image_path_2 = r"data\map.png"

img = Image.open(map_image_path).convert('L')  # 'L' stands for luminance
map_view = np.array(np.array(img) / 255, dtype=np.bool_)

img2 = Image.open(map_image_path_2).convert('L')  # 'L' stands for luminance
map_view_2 = np.array(np.array(img2) / 255, dtype=np.bool_)

# original:
#             "angle_max_change": 0.3,
#             "car_dimensions": (30, 45),  # width, height
#             "initial_speed": 0.3,
#             "min_speed": 0.3,
#             "max_speed": 1.5,  # 1.5
#             "speed_change": 0.01,
MAX_STEPS = 2000  # 5000
# "angle_max_change": 1.2,
# "car_dimensions": (30, 45),  # width, height
# "initial_speed": 1.2,
# "min_speed": 1.2,
# "max_speed": 6.0,  # 1.5
# "speed_change": 0.04,

CONSTANTS_DICT = {
    "environment": {
        "name": "Basic_Car_Environment",
        "universal_kwargs": {
            "angle_max_change": 1.15,  # 1.15
            "car_dimensions": (30, 45),  # width, height
            "initial_speed": 1.2,  # 1.2
            "min_speed": 1.2,  # 1.2
            "max_speed": 6,
            "speed_change": 0.04,
            "rays_degrees": (-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90),  # (-90, -45, 0, 45, 90),  # (-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90)
            "rays_distances_scale_factor": 100,
            "ray_input_clip": 5,
            "collision_reward": -100,
        },
        "changeable_training_kwargs_list": [
            {
                "map_view": map_view,
                "start_position": (504, 744),
                "start_angle": 0,
                "max_steps": MAX_STEPS,
            },
            {
                "map_view": map_view,
                "start_position": (504, 744),
                "start_angle": 180,
                "max_steps": MAX_STEPS,
            },
            {
                "map_view": map_view,
                "start_position": (425, 337),
                "start_angle": 170,
                "max_steps": MAX_STEPS,
            },
            {
                "map_view": map_view,
                "start_position": (283, 536),
                "start_angle": 200,
                "max_steps": MAX_STEPS,
            },
            {
                "map_view": map_view,
                "start_position": (665, 400),
                "start_angle": 270,
                "max_steps": MAX_STEPS,
            },
            {
                "map_view": map_view,
                "start_position": (366, 173),
                "start_angle": 315,
                "max_steps": MAX_STEPS,
            }
            # {
            #     "map_view": map_view_2,
            #     "start_position": (365, 744),
            #     "start_angle": 0,
            #     "max_steps": MAX_STEPS,
            # },
            # {
            #     "map_view": map_view_2,
            #     "start_position": (365, 744),
            #     "start_angle": 180,
            #     "max_steps": MAX_STEPS,
            # }
        ],
        "changeable_validation_kwargs_list": [
            {
                "map_view": map_view,
                "start_position": (504, 744),
                "start_angle": 0,
                "max_steps": 5_000,
            },
            # {
            #     "map_view": map_view_2,
            #     "start_position": (365, 744),
            #     "start_angle": 0,
            #     "max_steps": 10000,
            # },
        ],
    },
    "neural_network": {
        "input_normal_size": 10,
        "out_actions_number": 6,
        "normal_hidden_layers": 2,
        "normal_hidden_neurons": 256,  # 64
        "dropout": 0.0,
        "normal_activation_function": "relu",  # "relu"
        "last_activation_function": [("softmax", 3), ("softmax", 3)], # [("softmax", 3), ("tanh", 1)],
    },
    "Genetic_Algorithm": {
        "population": 500,
        "max_evaluations": 100_000,
        "epochs": 10000,
        "mutation_factor": 0.1,
        "crosses_per_epoch": 99,
        # "new_individual_every_n_epochs": 2,
        "save_logs_every_n_epochs": 10,
        "max_threads": 0,
        "logs_path": r"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
    },
    "Differential_Evolution": {
        "population": 2000,
        "max_evaluations": 100_000,
        "epochs": 10000,
        "cross_prob": 0.9,
        "diff_weight": 0.8,
        "save_logs_every_n_epochs": 50,
        "max_threads": 0,
        "logs_path": r"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
    },
    "Evolutionary_Mutate_Population": {
        "population": 500,
        "max_evaluations": 10_000_000,
        "epochs": 1000,
        "mutation_controller": {
            # "name": "Mut_Prob",  # "SHADE_single", "Mut_Prob", "SHADE_multiple"
            # "kwargs": {
            #     "mem_size": 10,
            #     "initial_mut_fact_range": (0.1, 0.1), # (0.001, 0.2)
            #     # "mut_change_sigma": 0.1,  # 0.1
            #     "survival_rate": 0.01,
            #     "learning_rate": 0.1,
            # },
            # "name": "Mut_Prob_Epochs",
            # "kwargs": {
            #     "initial_mut_fact_range": (0.08, 0.1),
            #     "change_rate": 0.2,
            # },
            "name": "Mut_One",
            "kwargs": {
                "mutation_factor": 0.1,
                "use_children": False,
            },
        },
        "max_threads": 8,
        "save_logs_every_n_epochs": 50,
        "logs_path": r"logs",
    },
    "Evolutionary_Mutate_Population_Original": {
        "population": 5000,
        "best_base_N": 100,
        "max_evaluations": 100000,
        "epochs": 1000,
        "mutation_controller": {
            "name": "Mut_One",
            "kwargs": {
                "mutation_factor": 0.1,
                "use_children": False,
            },
        },
        "max_threads": 8,
        "save_logs_every_n_epochs": 50,
        "logs_path": r"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
    },
    "GESMR": {
        "population": 100,
        "epochs": 400,
        "k_groups": 10,
        "mut_range": (0.08, 0.1),
        "individual_ratio_breed": 0.5,
        "mutation_ratio_breed": 0.5,
        "mutation_ratio_mutate": 0.5,
        "max_threads": 8,
        "save_logs_every_n_epochs": 5,
        "logs_path": r"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
    },
    "Param_Les_Ev_Mut_Pop": {
        "epochs": 10_000,
        "mutation_controller": {
            "name": "Mut_Prob",  # "SHADE_single", "Mut_Prob", "SHADE_multiple"
            "kwargs": {
                "mem_size": 10,
                "initial_mut_fact_range": (0.05, 0.15), # (0.001, 0.2)
                # "mut_change_sigma": 0.1,  # 0.1
                "survival_rate": 0.01,
                "learning_rate": 0.1,
            },
        },
        "max_threads": 22,
        "save_logs_every_n_epochs": 300,
        "logs_path": r"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
    },
    "Evolutionary_Strategy": {
        "permutations": 1000,
        "max_evaluations": 100_000,
        "epochs": 10000,
        "sigma_change": 0.01,
        "learning_rate": 0.1,
        "save_logs_every_n_epochs": 20,
        "max_threads": 0,
        "logs_path": r"C:\Piotr\AIProjects\Evolutionary_Cars\logs",
    },
    "visualization": {
        "car_image_path": car_image_path,
        "map_image_path": map_image_path,
    },
}
