'''
    ########################################
    ## Sports2D tests                     ##
    ########################################

    Check whether Sports2D still works after each code modification.
    Disable the real-time results and plots to avoid any GUI issues.

    Usage: 
    cd Sports2D/Utilities
    python tests.py
'''


## INIT
import toml
import subprocess
from pathlib import Path


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2023, Sports2D"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.4.0"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def test_workflow():
    '''
    Test the workflow of Sports2D.
    '''

    # From python
    config_path = Path.cwd().parent / 'Demo' / 'Config_demo.toml'
    config_dict = toml.load(config_path)
    config_dict.get("project").update({"video_dir":'../Demo'})
    config_dict.get("process").update({"show_realtime_results":False})
    config_dict.get("post-processing").update({"show_graphs":False})
    
    from Sports2D import Sports2D
    Sports2D.process(config_dict)


    # From command line (CLI)
    # Default
    demo_cmd = ["sports2d", "--show_realtime_results", "False", "--show_graphs", "False"]
    subprocess.run(demo_cmd, check=True, capture_output=True, text=True)

    # From config file
    demo_cmd2 = ["sports2d", "--config", str(config_path), "--show_realtime_results", "False", "--show_graphs", "False"]
    subprocess.run(demo_cmd2, check=True, capture_output=True, text=True)

    # With custom parameters
    demo_cmd3 = ["sports2d", "--to_meters", "False", "--multiperson", "False", "--mode", "lightweight", "--det_frequency", "50", "--time_range", "1.2 2.7",  "--slowmo_factor", "4", "--show_realtime_results", "False", "--show_graphs", "False"]
    subprocess.run(demo_cmd3, check=True, capture_output=True, text=True)
    
    # With inverse kinematics
    # demo_cmd4 = ["sports2d", "--do_ik", "--person_orientation", "front none left", "--show_realtime_results", "False", "--show_graphs", "False"]
    # subprocess.run(demo_cmd4, check=True, capture_output=True, text=True)
  
    
    # TODO: From command line (CLI) with config file 
    cli_config_path = Path(__file__).resolve().parent.parent / 'Demo' / 'Config_demo.toml'
    config_dict = toml.load(cli_config_path)
    cli_video_dir = Path(__file__).resolve().parent.parent / 'Demo'
    config_dict.get("project").update({"video_dir": str(cli_video_dir)})
    with open(cli_config_path, 'w') as f: toml.dump(config_dict, f)

    demo_config_cmd = ["sports2d", "--config", str(cli_config_path), "--show_realtime_results", "False", "--show_graphs", "False"]
    subprocess.run(demo_config_cmd, check=True, capture_output=True, text=True)
