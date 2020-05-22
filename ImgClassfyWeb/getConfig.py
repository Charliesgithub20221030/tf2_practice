import configparser 

def get_config(config_file = 'config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file)
    _conf_ints = [(k,int(v)) for k,v in parser.items('ints')]
    _conf_string = [(k,str(v)) for k,v in parser.items('strings')]
    _conf_floats = [(k,float(v)) for k,v in parser.items('floats')]
    return dict(
            _conf_ints+
            _conf_floats+
            _conf_string)

            
