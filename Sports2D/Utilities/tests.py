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
    config_dict = toml.load('../Demo/Config_demo.toml')
    config_dict.get("project").update({"video_dir":'../Demo'})
    config_dict.get("process").update({"show_realtime_results":False})
    config_dict.get("post-processing").update({"show_plots":False})
    
    from Sports2D import Sports2D
    Sports2D.process(config_dict)

    # From command line (CLI)
    subprocess.run(["python", "sports2d", "--show_realtime_results", "False", "--show_plots", "False"])
    
    # From command line (CLI) with config file
    subprocess.run(["python", "sports2d", "-c", "../Config_demo.toml", "--show_realtime_results", "False", "--show_plots", "False"])

