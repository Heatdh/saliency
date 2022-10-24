import json
import os
from pathlib import  Path
#from test_class import  test
import re
from train_cfgs import train_simpleNet
import torch



def get_config(json_file,json_dir):
    """Read default config and update files based on config file

    Args:
        json_path (dir): directory of jsons
    """
    #print("Loading Json file", json_file)
    with open(str(json_dir)+"/"+json_file, 'r') as j:
        contents = json.loads(j.read())

    default_config_path = Path(__file__).parent.parent/"cfgs/default_cfg.json"
    with open(default_config_path, 'r') as t:
        default_dict = json.loads(t.read())

    default_dict.update(contents)
    
    #print("Updated default directory with values", contents, "resulting in new dict", default_dict)
    return default_dict




if __name__ == '__main__':
    json_dir = Path(__file__).parent.parent/"cfgs"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    json_files = [pos_json for pos_json in os.listdir(json_dir) if pos_json.endswith('.json') and not pos_json.startswith('default')]
    #We want to make sure that the configs are processed in their numerical order
    JSON_FILE_PREFIX = "testbenchmark_"
    json_files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    

    #Enter all numbers which should be excluded:
    skip_experiments = []
    for i in json_files:
        if i.startswith(JSON_FILE_PREFIX):
            experiment_key = int(os.path.splitext(i)[0][len(JSON_FILE_PREFIX):])
        if experiment_key in skip_experiments:
            continue
        
        try:
            s=get_config(i,json_dir)
            simple_obj=train_simpleNet(s,i)
            simple_obj.config_Wb()
            simple_obj.Sal_Dataloader()
            simple_obj.load_encoder()
            simple_obj.set_optimizer()
            simple_obj.train_model()
            #simple_obj.export_model()
        except Exception as e:
            print("Failed run with", e)
