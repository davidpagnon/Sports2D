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
    python_config_path = Path.cwd().parent / 'Demo' / 'Config_demo.toml'
    config_dict = toml.load(python_config_path)
    config_dict.get("project").update({"video_dir":'../Demo'})
    config_dict.get("process").update({"show_realtime_results":False})
    config_dict.get("post-processing").update({"show_graphs":False})
    
    from Sports2D import Sports2D
    Sports2D.process(config_dict)


    # From command line (CLI)
    demo_cmd = ["sports2d", "--show_realtime_results", "False", "--show_graphs", "False"]
    subprocess.run(demo_cmd, check=True, capture_output=True, text=True)
  
    
    # TODO: From command line (CLI) with config file 
    cli_config_path = Path(__file__).resolve().parent.parent / 'Demo' / 'Config_demo.toml'
    config_dict = toml.load(cli_config_path)
    cli_video_dir = Path(__file__).resolve().parent.parent / 'Demo'
    config_dict.get("project").update({"video_dir": str(cli_video_dir)})
    with open(cli_config_path, 'w') as f: toml.dump(config_dict, f)

    demo_config_cmd = ["sports2d", "--config", str(cli_config_path), "--show_realtime_results", "False", "--show_graphs", "False"]
    subprocess.run(demo_config_cmd, check=True, capture_output=True, text=True)
